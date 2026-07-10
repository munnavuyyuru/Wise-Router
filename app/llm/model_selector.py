from typing import Optional


MODEL_COST_TIERS: dict[str, list[str]] = {
    "math": [
        "accounts/fireworks/models/gemma-3-1b-it",
        "accounts/fireworks/models/gemma-3-4b-it",
        "accounts/fireworks/models/minimax-m3",
    ],
    "sentiment": [
        "accounts/fireworks/models/gemma-3-1b-it",
        "accounts/fireworks/models/gemma-3-4b-it",
    ],
    "code_debug": [
        "accounts/fireworks/models/gemma-3-4b-it",
        "accounts/fireworks/models/kimi-k2p7-code",
    ],
    "code_gen": [
        "accounts/fireworks/models/kimi-k2p7-code",
    ],
    "summarization": [
        "accounts/fireworks/models/gemma-3-1b-it",
        "accounts/fireworks/models/gemma-3-4b-it",
    ],
    "ner": [
        "accounts/fireworks/models/gemma-3-1b-it",
        "accounts/fireworks/models/gemma-3-4b-it",
    ],
    "logic": [
        "accounts/fireworks/models/gemma-3-4b-it",
        "accounts/fireworks/models/minimax-m3",
    ],
    "factual": [
        "accounts/fireworks/models/gemma-3-1b-it",
        "accounts/fireworks/models/gemma-3-4b-it",
    ],
}

DEFAULT_TIER: list[str] = [
    "accounts/fireworks/models/gemma-3-4b-it",
    "accounts/fireworks/models/minimax-m3",
]


def choose(task_type: str, allowed_models: list[str]) -> Optional[str]:
    allowed_set = set(allowed_models)
    tiers = MODEL_COST_TIERS.get(task_type, DEFAULT_TIER)
    for model in tiers:
        if model in allowed_set:
            return model
    for model in DEFAULT_TIER:
        if model in allowed_set:
            return model
    if allowed_models:
        return allowed_models[0]
    return None


def cost_tier(model_name: str) -> int:
    cheap = {"accounts/fireworks/models/gemma-3-1b-it"}
    medium = {"accounts/fireworks/models/gemma-3-4b-it"}
    expensive = {"accounts/fireworks/models/minimax-m3"}
    if model_name in cheap:
        return 0
    if model_name in medium:
        return 1
    if model_name in expensive:
        return 2
    return 3


def fallback_tiers(task_type: str, allowed_models: list[str]) -> list[str]:
    allowed_set = set(allowed_models)
    tiers = MODEL_COST_TIERS.get(task_type, DEFAULT_TIER)
    result: list[str] = []
    seen: set[str] = set()
    for model in tiers + DEFAULT_TIER:
        if model in allowed_set and model not in seen:
            result.append(model)
            seen.add(model)
    if not result:
        result = allowed_models[:1] if allowed_models else []
    return result
