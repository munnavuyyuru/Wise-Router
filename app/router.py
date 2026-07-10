from typing import Optional

from app.classifier.task_classifier import TaskClassifier
from app.config import Config
from app.deterministic import math_solver, sentiment, code_tools, summarizer, ner, logic_solver, factual
from app.llm.fireworks_client import FireworksClient
from app.llm import model_selector

SOLVERS = {
    "math": math_solver.solve,
    "sentiment": sentiment.solve,
    "code_debug": code_tools.solve,
    "code_gen": None,
    "summarization": summarizer.solve,
    "ner": ner.solve,
    "logic": logic_solver.solve,
    "factual": factual.solve,
}


def try_deterministic(prompt: str, config: Config, classifier: TaskClassifier) -> Optional[str]:
    if not classifier.is_fitted:
        return None
    task_type, confidence = classifier.predict(prompt)
    threshold = config.task_thresholds.get(task_type, 0.80)
    solver_fn = SOLVERS.get(task_type)
    if solver_fn and confidence >= threshold:
        answer, solver_confidence = solver_fn(prompt)
        if answer is not None:
            effective_conf = solver_confidence if solver_confidence is not None else confidence
            if effective_conf >= threshold:
                return answer
    return None


def route_prompt(
    prompt: str,
    config: Config,
    classifier: TaskClassifier,
    fireworks: Optional[FireworksClient],
    time_budget_remaining: Optional[float] = None,
) -> str:
    answer = try_deterministic(prompt, config, classifier)
    if answer is not None:
        return answer

    if not classifier.is_fitted:
        task_type = "code_gen"
    else:
        task_type = classifier.predict_type(prompt)

    if fireworks is None:
        return f"[dry-run] would route to Fireworks (type={task_type})"

    if time_budget_remaining is not None and time_budget_remaining < 20.0:
        return ""

    models = model_selector.fallback_tiers(task_type, config.allowed_models)
    if not models:
        return ""

    try:
        return fireworks.generate_with_fallback(prompt, models=models, task_type=task_type)
    except Exception:
        return ""
