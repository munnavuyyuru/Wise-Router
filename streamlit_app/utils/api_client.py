import os
import requests

API_BASE = os.getenv("ROUTELLM_API_URL", "http://localhost:8000/api").rstrip("/")


def _post(path: str, body: dict) -> dict:
    resp = requests.post(f"{API_BASE}{path}", json=body, timeout=300)
    resp.raise_for_status()
    return resp.json()


def _get(path: str) -> dict | list:
    resp = requests.get(f"{API_BASE}{path}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def _delete(path: str) -> dict:
    resp = requests.delete(f"{API_BASE}{path}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def health() -> dict:
    return _get("/health")


def route_single(prompt: str, dry_run: bool = True) -> dict:
    return _post("/route", {"prompt": prompt, "dry_run": dry_run})


def evaluate(tasks: list[dict], dry_run: bool = True, threshold: float = 85.0) -> dict:
    return _post("/evaluate", {"tasks": tasks, "dry_run": dry_run, "threshold": threshold})


def list_runs() -> list:
    result = _get("/runs")
    return result if isinstance(result, list) else []


def get_run(run_id: str) -> dict:
    return _get(f"/runs/{run_id}")


def delete_run(run_id: str) -> dict:
    return _delete(f"/runs/{run_id}")


def stats() -> dict:
    return _get("/stats")


def cost_compare(tasks: list[dict], baseline_model: str = "accounts/fireworks/models/kimi-k2p7-code", dry_run: bool = True) -> dict:
    return _post("/cost-compare", {"tasks": tasks, "dry_run": dry_run, "baseline_model": baseline_model})
