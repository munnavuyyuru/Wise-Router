import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from app.config import Config


TASK_SYSTEM_PROMPTS: dict[str, str] = {
    "math": "Answer only the number.",
    "sentiment": "One word: positive, negative, or neutral.",
    "code_debug": "Output fixed code only.",
    "code_gen": "Output code only, no explanation.",
    "summarization": "Short summary.",
    "ner": "List entities only.",
    "logic": "Answer only.",
    "factual": "Answer fact only.",
}


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0

    def add(self, other: "TokenUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.cost += other.cost

    def __bool__(self) -> bool:
        return self.total_tokens > 0


MODEL_PRICING: dict[str, dict[str, float]] = {
    "accounts/fireworks/models/gemma-3-1b-it": {"input": 0.10, "output": 0.10},
    "accounts/fireworks/models/gemma-3-4b-it": {"input": 0.20, "output": 0.20},
    "accounts/fireworks/models/minimax-m3": {"input": 0.50, "output": 2.00},
    "accounts/fireworks/models/kimi-k2p7-code": {"input": 1.00, "output": 1.00},
}

DEFAULT_MODEL_COST_PER_1M: dict[str, float] = {"input": 0.50, "output": 0.50}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model, DEFAULT_MODEL_COST_PER_1M)
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)


@dataclass
class UsageRecord:
    timestamp: str = ""
    task_type: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0


class TokenTracker:
    def __init__(self, path: str | Path = "token_usage.json"):
        self.path = Path(path)
        self.cumulative = TokenUsage()
        self._records: list[UsageRecord] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._records = [UsageRecord(**r) for r in data.get("records", [])]
                cum = data.get("cumulative", {})
                self.cumulative = TokenUsage(
                    prompt_tokens=cum.get("prompt_tokens", 0),
                    completion_tokens=cum.get("completion_tokens", 0),
                    total_tokens=cum.get("total_tokens", 0),
                    cost=cum.get("cost", 0.0),
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                self._records = []
                self.cumulative = TokenUsage()

    def record(self, usage: TokenUsage, task_type: str = "", model: str = "") -> None:
        self.cumulative.add(usage)
        cost = estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)
        rec = UsageRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            task_type=task_type,
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost=cost,
        )
        self._records.append(rec)
        self._save()

    def _save(self) -> None:
        data = {
            "cumulative": asdict(self.cumulative),
            "total_calls": len(self._records),
            "records": [asdict(r) for r in self._records],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def reset(self) -> None:
        self.cumulative = TokenUsage()
        self._records = []
        if self.path.exists():
            self.path.unlink()


class FireworksClient:
    def __init__(self, config: Config, token_tracker: Optional[TokenTracker] = None) -> None:
        self.config = config
        self.base_url = config.fireworks_base_url
        self.api_key = config.fireworks_api_key
        self.timeout = 60.0
        self.max_retries = 3
        self.allowed_models = config.allowed_models
        self._last_request: float = 0.0
        self._min_interval = 2.0
        self.token_tracker = token_tracker or TokenTracker()

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()

    def generate(
        self,
        prompt: str,
        model: str,
        task_type: str = "default",
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str:
        if model not in self.allowed_models:
            raise ValueError(
                f"Model {model} not in ALLOWED_MODELS. "
                f"Allowed: {self.allowed_models}"
            )

        self._rate_limit()
        url = f"{self.base_url}/chat/completions"

        system_prompt = TASK_SYSTEM_PROMPTS.get(task_type, "Answer the question concisely.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": {"reasoning_effort": "none"},
        }

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code == 429:
                    time.sleep(min(2 ** attempt * 2, 30))
                    continue
                resp.raise_for_status()
                data = resp.json()

                usage_data = data.get("usage", {})
                if usage_data:
                    pt = usage_data.get("prompt_tokens", 0)
                    ct = usage_data.get("completion_tokens", 0)
                    cost = estimate_cost(model, pt, ct)
                    tu = TokenUsage(
                        prompt_tokens=pt,
                        completion_tokens=ct,
                        total_tokens=usage_data.get("total_tokens", 0),
                        cost=cost,
                    )
                    self.token_tracker.record(tu, task_type=task_type, model=model)

                return str(data["choices"][0]["message"]["content"]).strip()
            except requests.exceptions.ConnectionError as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise ConnectionError(f"Failed to connect to Fireworks: {e}") from e
            except requests.exceptions.Timeout as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise TimeoutError(f"Fireworks request timed out after {self.timeout}s") from e
            except requests.exceptions.HTTPError as e:
                if resp.status_code == 401:
                    raise ValueError("Invalid Fireworks API key") from e
                if resp.status_code == 429 and attempt < self.max_retries - 1:
                    time.sleep(min(2 ** attempt * 2, 30))
                    continue
                raise

        return ""

    def generate_with_fallback(
        self,
        prompt: str,
        models: list[str],
        task_type: str = "default",
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str:
        errors: list[str] = []
        for model in models:
            try:
                return self.generate(prompt, model=model, task_type=task_type, temperature=temperature, max_tokens=max_tokens)
            except (ValueError, ConnectionError, TimeoutError, requests.exceptions.RequestException) as e:
                errors.append(f"{model}: {e}")
                continue
        raise RuntimeError(f"All models failed for task_type={task_type}. Errors: {'; '.join(errors)}")

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)
