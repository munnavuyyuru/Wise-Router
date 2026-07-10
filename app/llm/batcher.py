import re
from collections import defaultdict
from typing import Optional

from app.classifier.task_classifier import TaskClassifier
from app.config import Config
from app.llm.fireworks_client import FireworksClient
from app.llm import model_selector
from app.router import try_deterministic

BATCH_SIZE = 5


class BatchProcessor:
    def __init__(self, config: Config, classifier: TaskClassifier, fireworks: Optional[FireworksClient]):
        self.config = config
        self.classifier = classifier
        self.fireworks = fireworks

    def process_all(self, tasks: list[dict], time_limit: float = 600.0, budget_margin: float = 20.0) -> list[dict]:
        import time
        start_time = time.time()
        results: list[dict] = []
        pending: list[tuple[int, str, str, str, str]] = []

        for task in tasks:
            task_id = task.get("task_id", "")
            prompt = task.get("prompt", "")
            if not task_id or not prompt:
                import sys
                print(f"Warning: skipping task with missing task_id or prompt: {task}", file=sys.stderr)
                continue

            answer = try_deterministic(prompt, self.config, self.classifier)
            if answer is not None:
                results.append({"task_id": task_id, "answer": answer})
            else:
                task_type = self.classifier.predict_type(prompt) if self.classifier.is_fitted else "code_gen"
                model = model_selector.choose(task_type, self.config.allowed_models)
                pending.append((len(results), task_id, prompt, task_type, model or ""))
                results.append(None)

        if pending and self.fireworks:
            elapsed = time.time() - start_time
            remaining = time_limit - elapsed
            if remaining < budget_margin:
                for idx, _, _, _, _ in pending:
                    if idx < len(results):
                        results[idx] = {"task_id": results[idx]["task_id"] if results[idx] else "", "answer": ""}
            else:
                self._flush_batches(pending, results, remaining)

        for i, r in enumerate(results):
            if r is None:
                task_type = self.classifier.predict_type(tasks[i].get("prompt", "")) if self.classifier.is_fitted else "unknown"
                label = f"[dry-run] would route to Fireworks (type={task_type})" if not self.fireworks else ""
                results[i] = {"task_id": tasks[i].get("task_id", ""), "answer": label}

        return results

    def _flush_batches(
        self,
        pending: list[tuple[int, str, str, str, str]],
        results: list[dict],
        remaining: float,
    ) -> None:
        batches: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
        for idx, task_id, prompt, task_type, model in pending:
            key = f"{task_type}:{model}"
            batches[key].append((idx, task_id, prompt))

        import time
        for key, batch in batches.items():
            task_type, model = key.split(":", 1)
            for batch_start in range(0, len(batch), BATCH_SIZE):
                if remaining < 20.0:
                    return
                chunk = batch[batch_start:batch_start + BATCH_SIZE]
                batch_prompt = self._format_batch_prompt(chunk)
                try:
                    response = self.fireworks.generate(batch_prompt, model=model, task_type=task_type, max_tokens=1024)
                    answers = self._parse_batch_response(response, len(chunk))
                except Exception:
                    answers = [""] * len(chunk)
                for (idx, task_id, _), answer in zip(chunk, answers):
                    if idx < len(results):
                        results[idx] = {"task_id": task_id, "answer": answer}
                remaining -= time.time() - (time.time() - 60)

    def _format_batch_prompt(self, chunk: list[tuple[int, str, str]]) -> str:
        lines = []
        for i, (_, _, prompt) in enumerate(chunk, 1):
            lines.append(f"{i}. {prompt}")
        return "\n".join(lines)

    def _parse_batch_response(self, response: str, count: int) -> list[str]:
        lines = [l.strip() for l in response.splitlines() if l.strip()]
        answers: list[str] = []
        for line in lines:
            line = re.sub(r'^\d+[.)]\s*', '', line).strip()
            if line:
                answers.append(line)
        while len(answers) < count:
            answers.append("")
        return answers[:count]
