import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.classifier.task_classifier import TaskClassifier
from app.config import Config
from app.llm.fireworks_client import FireworksClient, TokenTracker, estimate_cost
from app.router import try_deterministic
from app.llm import model_selector

app = FastAPI(title="RoutellM API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = Config.from_env()
_tracker = TokenTracker()
_tracker.reset()
_classifier: Optional[TaskClassifier] = None
_fireworks: Optional[FireworksClient] = None

TASK_TYPES = ["math", "sentiment", "code_debug", "code_gen", "summarization", "ner", "logic", "factual"]
DET = "deterministic"
DRY = "dry_run"
API = "fireworks_api"


def _lazy_init() -> None:
    global _classifier, _fireworks
    if _classifier is None:
        _classifier = TaskClassifier(config.classifier_model_path)
        if not _classifier.is_fitted:
            train_path = os.path.join(os.path.dirname(__file__), "..", "data", "training_data.json")
            if os.path.exists(train_path):
                with open(train_path, "r", encoding="utf-8") as f:
                    training = json.load(f)
                prompts = [t["prompt"] for t in training if t.get("prompt") and t.get("task_type")]
                labels = [t["task_type"] for t in training if t.get("prompt") and t.get("task_type")]
                if len(set(labels)) >= 2:
                    _classifier.fit(prompts, labels)
    if _fireworks is None and config.fireworks_api_key:
        _fireworks = FireworksClient(config, token_tracker=_tracker)


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


class RouteRequest(BaseModel):
    prompt: str
    dry_run: bool = True


class EvaluateRequest(BaseModel):
    tasks: list[dict]
    dry_run: bool = True
    threshold: float = 85.0


class CostCompareRequest(BaseModel):
    tasks: list[dict]
    baseline_model: str = "accounts/fireworks/models/kimi-k2p7-code"
    dry_run: bool = True


@app.get("/api/health")
def health():
    return {"status": "ok", "classifier_loaded": _classifier is not None and _classifier.is_fitted}


@app.get("/api/stats")
def stats():
    return {
        "total_tokens": _tracker.cumulative.total_tokens,
        "total_cost": round(_tracker.cumulative.cost, 6),
        "api_calls": len(_tracker._records),
    }


@app.get("/api/runs")
def list_runs():
    records = _tracker._records
    return [
        {
            "run_id": f"run_{i}",
            "timestamp": r.timestamp,
            "task_type": r.task_type,
            "model": r.model,
            "tokens": r.total_tokens,
            "cost": r.cost,
        }
        for i, r in enumerate(records[-50:])
    ]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    return {"summary": {"total_tasks": 0, "correct": 0, "incorrect": 0, "accuracy_pct": 0.0, "source_breakdown": {DET: 0, API: 0, DRY: 0}}, "cost": {"total_cost_usd": 0, "total_prompt_tokens": 0, "total_completion_tokens": 0, "total_tokens": 0, "api_calls": 0}, "per_task_type": {}, "results": []}


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str):
    return {"status": "deleted"}


@app.post("/api/route")
def route_single(req: RouteRequest):
    _lazy_init()
    api_before = len(_tracker._records)
    task_type = _classifier.predict_type(req.prompt) if _classifier and _classifier.is_fitted else "unknown"
    confidence = _classifier.predict_confidence(req.prompt) if _classifier and _classifier.is_fitted else 0.0
    threshold = config.task_thresholds.get(task_type, 0.80)

    answer = try_deterministic(req.prompt, config, _classifier)
    source = DET
    model_used = ""
    if answer is None:
        if not _fireworks or req.dry_run:
            answer = f"[dry-run] would route to Fireworks (type={task_type})"
            source = DRY
        else:
            models = model_selector.fallback_tiers(task_type, config.allowed_models)
            if models:
                try:
                    answer = _fireworks.generate_with_fallback(req.prompt, models=models, task_type=task_type)
                    source = API
                    if _tracker._records:
                        model_used = _tracker._records[-1].model
                except Exception:
                    answer = ""

    cost = 0.0
    tokens = 0
    api_after = len(_tracker._records)
    if api_after > api_before and _tracker._records:
        last = _tracker._records[-1]
        cost = last.cost
        tokens = last.total_tokens

    return {
        "task_type": task_type,
        "confidence": round(confidence, 3),
        "threshold": threshold,
        "source": source,
        "model_used": model_used,
        "answer": answer,
        "cost": cost,
        "tokens": tokens,
    }


@app.post("/api/evaluate")
def evaluate(req: EvaluateRequest):
    _lazy_init()
    results = []
    correct = 0
    total = 0
    by_type = {t: {"correct": 0, "total": 0, "by_source": {DET: 0, API: 0, DRY: 0}} for t in TASK_TYPES}
    source_counts = {DET: 0, API: 0, DRY: 0}
    api_calls_before = len(_tracker._records)

    for task in req.tasks:
        task_id = task.get("task_id", "")
        prompt = task.get("prompt", "")
        expected = task.get("expected_answer", "")
        if not task_id or not prompt:
            continue

        api_before = len(_tracker._records)
        task_type = _classifier.predict_type(prompt) if _classifier and _classifier.is_fitted else "unknown"

        answer = try_deterministic(prompt, config, _classifier)
        if answer is not None:
            source = DET
        elif not _fireworks or req.dry_run:
            answer = f"[dry-run] would route to Fireworks (type={task_type})"
            source = DRY
        else:
            models = model_selector.fallback_tiers(task_type, config.allowed_models)
            if models:
                try:
                    answer = _fireworks.generate_with_fallback(prompt, models=models, task_type=task_type)
                    source = API
                except Exception:
                    answer = ""
                    source = DRY
            else:
                answer = "[dry-run] would route to Fireworks (type={task_type})"
                source = DRY

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

    api_calls_after = len(_tracker._records)
    cum = _tracker.cumulative

    for t in TASK_TYPES:
        info = by_type[t]
        info["accuracy"] = round((info["correct"] / info["total"] * 100), 1) if info["total"] else None

    return {
        "results": results,
        "summary": {
            "total_tasks": len(results),
            "correct": correct,
            "incorrect": total - correct,
            "accuracy_pct": round((correct / total * 100) if total else 0.0, 2),
            "source_breakdown": source_counts,
        },
        "cost": {
            "total_cost_usd": round(cum.cost, 6),
            "total_prompt_tokens": cum.prompt_tokens,
            "total_completion_tokens": cum.completion_tokens,
            "total_tokens": cum.total_tokens,
            "api_calls": api_calls_after - api_calls_before,
        },
        "per_task_type": by_type,
    }


@app.post("/api/cost-compare")
def cost_compare(req: CostCompareRequest):
    eval_result = evaluate(EvaluateRequest(tasks=req.tasks, dry_run=req.dry_run))
    baseline_total_tokens = 0
    baseline_cost = 0.0
    for task in req.tasks:
        prompt = task.get("prompt", "")
        if prompt:
            pt = max(1, len(prompt) // 4)
            ct = 50
            baseline_total_tokens += pt + ct
            baseline_cost += estimate_cost(req.baseline_model, pt, ct)

    routellm_cost = eval_result["cost"]["total_cost_usd"]
    savings_usd = baseline_cost - routellm_cost
    savings_pct = ((baseline_cost - routellm_cost) / baseline_cost * 100) if baseline_cost > 0 else 0.0

    return {
        "routellm_result": eval_result,
        "baseline": {
            "total_tokens": baseline_total_tokens,
            "total_cost_usd": round(baseline_cost, 6),
            "model": req.baseline_model,
        },
        "savings": {
            "cost_usd": round(savings_usd, 6),
            "cost_pct": round(savings_pct, 1),
        },
    }
