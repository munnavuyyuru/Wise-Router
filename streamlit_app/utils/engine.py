import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.classifier.task_classifier import TaskClassifier
from app.config import Config
from app.llm.fireworks_client import FireworksClient, TokenTracker, estimate_cost
from app.router import route_prompt, SOLVERS


TASK_TYPES = ["math", "sentiment", "code_debug", "code_gen", "summarization", "ner", "logic", "factual"]
DET = "deterministic"
DRY = "dry_run"
API = "fireworks_api"


def _matches(answer: str, expected: str) -> bool:
    if not expected:
        return bool(answer.strip()) if answer.strip() else True
    a, e = answer.strip().lower(), expected.strip().lower()
    if a == e:
        return True
    try:
        return abs(float(a) - float(e)) < 0.01
    except (ValueError, TypeError):
        pass
    if len(e) > 3 and e in a:
        return True
    return False


def get_thresholds(config: Config) -> dict[str, float]:
    return dict(config.task_thresholds)


class RoutingEngine:
    def __init__(self, dry_run: bool = False):
        self.config = Config.from_env()
        self.dry_run = dry_run
        self.classifier: Optional[TaskClassifier] = None
        self.tracker = TokenTracker()
        self.tracker.reset()
        self.fireworks: Optional[FireworksClient] = None
        self._initialized = False

    def init_classifier(self, train_data_path: str = "data/training_data.json",
                        model_path: Optional[str] = None) -> None:
        config = self.config
        mp = model_path or config.classifier_model_path
        self.classifier = TaskClassifier(mp)

        if not self.classifier.is_fitted and os.path.exists(train_data_path):
            with open(train_data_path, "r", encoding="utf-8") as f:
                training = json.load(f)
            prompts = [t["prompt"] for t in training if t.get("prompt") and t.get("task_type")]
            labels = [t["task_type"] for t in training if t.get("prompt") and t.get("task_type")]
            if len(set(labels)) >= 2:
                self.classifier.fit(prompts, labels)

    def init_fireworks(self) -> None:
        has_key = bool(self.config.fireworks_api_key) and not self.dry_run
        self.tracker.reset()
        self.fireworks = FireworksClient(self.config, token_tracker=self.tracker) if has_key else None

    def ensure_ready(self, train_data_path: str = "data/training_data.json",
                     model_path: Optional[str] = None) -> None:
        if not self._initialized:
            self.init_classifier(train_data_path, model_path)
            self.init_fireworks()
            self._initialized = True

    def route_single(self, prompt: str) -> dict:
        self.ensure_ready()
        task_type = self.classifier.predict_type(prompt) if self.classifier and self.classifier.is_fitted else "unknown"
        confidence = self.classifier.predict_confidence(prompt) if self.classifier and self.classifier.is_fitted else 0.0
        threshold = self.config.task_thresholds.get(task_type, 0.80)
        solver_fn = SOLVERS.get(task_type)
        solver_name = solver_fn.__name__ if solver_fn else None

        api_before = len(self.tracker._records)
        try:
            answer = route_prompt(prompt, self.config, self.classifier, self.fireworks)
        except Exception as exc:
            answer = f"[error] {exc}"

        api_after = len(self.tracker._records)
        if api_after > api_before:
            source = API
            model_used = self.tracker._records[-1].model if self.tracker._records else ""
        elif answer.startswith("[dry-run]"):
            source = DRY
            model_used = ""
        else:
            source = DET
            model_used = ""

        cost = 0.0
        tokens = 0
        if source == API and self.tracker._records:
            last = self.tracker._records[-1]
            cost = last.cost
            tokens = last.total_tokens

        return {
            "task_type": task_type,
            "confidence": round(confidence, 3),
            "threshold": threshold,
            "solver_name": solver_name,
            "passed_threshold": confidence >= threshold if solver_fn else False,
            "source": source,
            "model_used": model_used,
            "answer": answer,
            "cost": cost,
            "tokens": tokens,
        }

    def route_batch(self, tasks: list[dict]) -> dict:
        self.ensure_ready()
        results = []
        correct = 0
        total = 0
        by_type: dict = {t: {"correct": 0, "total": 0, "by_source": {DET: 0, API: 0, DRY: 0}} for t in TASK_TYPES}
        source_counts = {DET: 0, API: 0, DRY: 0}

        for task in tasks:
            task_id = task.get("task_id", "")
            prompt = task.get("prompt", "")
            expected = task.get("expected_answer", "")

            if not task_id or not prompt:
                continue

            api_before = len(self.tracker._records)
            task_type = self.classifier.predict_type(prompt) if self.classifier and self.classifier.is_fitted else "unknown"

            try:
                answer = route_prompt(prompt, self.config, self.classifier, self.fireworks)
            except Exception:
                answer = ""

            api_after = len(self.tracker._records)
            if api_after > api_before:
                source = API
            elif answer.startswith("[dry-run]"):
                source = DRY
            else:
                source = DET

            source_counts[source] += 1
            if task_type in by_type:
                by_type[task_type]["by_source"][source] += 1

            matched = None
            if expected or answer:
                total += 1
                matched = _matches(answer, expected)
                if matched:
                    correct += 1
                    if task_type in by_type:
                        by_type[task_type]["correct"] += 1
                if task_type in by_type:
                    by_type[task_type]["total"] += 1

            results.append({
                "task_id": task_id,
                "task_type": task_type,
                "prompt": prompt,
                "expected": expected,
                "answer": answer,
                "source": source,
                "correct": matched,
            })

        total_accuracy = (correct / total * 100) if total else 0.0
        cum = self.tracker.cumulative

        for t in TASK_TYPES:
            info = by_type[t]
            info["accuracy"] = round((info["correct"] / info["total"] * 100), 1) if info["total"] else None

        return {
            "results": results,
            "summary": {
                "total_tasks": len(results),
                "correct": correct,
                "incorrect": total - correct,
                "accuracy_pct": round(total_accuracy, 2),
                "source_breakdown": source_counts,
            },
            "cost": {
                "total_cost_usd": round(cum.cost, 6),
                "total_prompt_tokens": cum.prompt_tokens,
                "total_completion_tokens": cum.completion_tokens,
                "total_tokens": cum.total_tokens,
                "api_calls": len(self.tracker._records),
            },
            "per_task_type": by_type,
        }
