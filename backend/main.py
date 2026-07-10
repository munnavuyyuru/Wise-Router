import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from streamlit_app.utils.engine import RoutingEngine, TASK_TYPES, DET, DRY, API
from app.llm.fireworks_client import estimate_cost


app = FastAPI(title="RoutellM API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


_engine: Optional[RoutingEngine] = None
_runs: list[dict] = []


def get_engine(dry_run: bool = True) -> RoutingEngine:
    global _engine
    if _engine is None or _engine.dry_run != dry_run:
        _engine = RoutingEngine(dry_run=dry_run)
        _engine.ensure_ready()
    return _engine


class RouteRequest(BaseModel):
    prompt: str
    dry_run: bool = True


class RouteResponse(BaseModel):
    task_type: str
    confidence: float
    threshold: float
    solver_name: Optional[str] = None
    passed_threshold: bool
    source: str
    model_used: str
    answer: str
    cost: float
    tokens: int


class TaskItem(BaseModel):
    task_id: str
    prompt: str
    expected_answer: Optional[str] = ""


class EvaluateRequest(BaseModel):
    tasks: list[TaskItem]
    dry_run: bool = True
    threshold: float = 85.0


class EvaluateResponse(BaseModel):
    run_id: str
    created_at: str
    summary: dict
    cost: dict
    per_task_type: dict
    results: list[dict]


class CostCompareRequest(BaseModel):
    tasks: list[TaskItem]
    dry_run: bool = True
    threshold: float = 85.0
    baseline_model: str = "accounts/fireworks/models/kimi-k2p7-code"


@app.on_event("startup")
def startup():
    get_engine(dry_run=True)


@app.get("/api/health")
def health():
    engine = get_engine(dry_run=True)
    return {
        "status": "ok",
        "classifier_fitted": engine.classifier.is_fitted if engine.classifier else False,
        "api_key_present": bool(engine.config.fireworks_api_key),
    }


@app.post("/api/route", response_model=RouteResponse)
def route_single(req: RouteRequest):
    if not req.prompt.strip():
        raise HTTPException(400, "prompt cannot be empty")
    engine = get_engine(dry_run=req.dry_run)
    result = engine.route_single(req.prompt)
    return RouteResponse(**result)


@app.post("/api/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    if not req.tasks:
        raise HTTPException(400, "tasks list cannot be empty")
    engine = get_engine(dry_run=req.dry_run)
    tasks_dict = [t.model_dump() for t in req.tasks]
    result = engine.route_batch(tasks_dict)

    run_id = str(uuid.uuid4())[:8]
    created_at = datetime.now(timezone.utc).isoformat()
    run_record = {
        "run_id": run_id,
        "created_at": created_at,
        "threshold": req.threshold,
        "summary": result["summary"],
        "cost": result["cost"],
        "per_task_type": result["per_task_type"],
        "results": result["results"],
    }
    _runs.append(run_record)

    return EvaluateResponse(
        run_id=run_id,
        created_at=created_at,
        summary=result["summary"],
        cost=result["cost"],
        per_task_type=result["per_task_type"],
        results=result["results"],
    )


@app.get("/api/runs")
def list_runs():
    return [
        {
            "run_id": r["run_id"],
            "created_at": r["created_at"],
            "summary": r["summary"],
            "cost": r["cost"],
            "per_task_type": r["per_task_type"],
        }
        for r in _runs
    ]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    for r in _runs:
        if r["run_id"] == run_id:
            return r
    raise HTTPException(404, f"Run {run_id} not found")


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str):
    global _runs
    before = len(_runs)
    _runs = [r for r in _runs if r["run_id"] != run_id]
    if len(_runs) == before:
        raise HTTPException(404, f"Run {run_id} not found")
    return {"deleted": run_id}


@app.get("/api/stats")
def stats():
    if not _runs:
        return {
            "total_runs": 0,
            "total_tasks": 0,
            "avg_accuracy": 0.0,
            "total_cost_usd": 0.0,
            "total_tokens": 0,
        }
    total_tasks = sum(r["summary"]["total_tasks"] for r in _runs)
    avg_acc = sum(r["summary"]["accuracy_pct"] for r in _runs) / len(_runs)
    total_cost = sum(r["cost"]["total_cost_usd"] for r in _runs)
    total_tokens = sum(r["cost"]["total_tokens"] for r in _runs)
    return {
        "total_runs": len(_runs),
        "total_tasks": total_tasks,
        "avg_accuracy": round(avg_acc, 2),
        "total_cost_usd": round(total_cost, 6),
        "total_tokens": total_tokens,
    }


@app.post("/api/cost-compare")
def cost_compare(req: CostCompareRequest):
    eng = get_engine(dry_run=req.dry_run)
    tasks_dict = [t.model_dump() for t in req.tasks]
    result = eng.route_batch(tasks_dict)

    baseline_cost = 0.0
    baseline_tokens = 0
    for t in req.tasks:
        pt = max(1, len(t.prompt) // 4)
        ct = max(1, pt * 2)
        c = estimate_cost(req.baseline_model, pt, ct)
        baseline_cost += c
        baseline_tokens += pt + ct

    return {
        "routellm_result": result,
        "baseline": {
            "model": req.baseline_model,
            "total_cost_usd": round(baseline_cost, 6),
            "total_tokens": baseline_tokens,
        },
        "savings": {
            "cost_usd": round(baseline_cost - result["cost"]["total_cost_usd"], 6),
            "cost_pct": round((1 - result["cost"]["total_cost_usd"] / baseline_cost) * 100, 1) if baseline_cost > 0 else 0,
            "tokens": baseline_tokens - result["cost"]["total_tokens"],
        },
    }
