import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

SHORT_MODEL_MAP: dict[str, str] = {}
_cache_normalized: Optional[list[str]] = None

def normalize_model_name(name: str) -> str:
    name = name.strip()
    if name.startswith("accounts/fireworks/models/"):
        return name
    return f"accounts/fireworks/models/{name}"

def parse_allowed_models(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [normalize_model_name(p) for p in parts]


@dataclass
class Config:
    fireworks_api_key: str = field(
        default_factory=lambda: os.getenv("FIREWORKS_API_KEY", "")
    )
    fireworks_base_url: str = field(
        default_factory=lambda: os.getenv(
            "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
        ).rstrip("/")
    )
    allowed_models: list[str] = field(
        default_factory=lambda: parse_allowed_models(
            os.getenv("ALLOWED_MODELS", "")
        )
    )
    input_path: str = "/input/tasks.json"
    output_path: str = "/output/results.json"
    classifier_model_path: str = field(
        default_factory=lambda: os.getenv(
            "CLASSIFIER_MODEL_PATH",
            os.path.join(os.path.dirname(__file__), "..", "data", "task_classifier.pkl"),
        )
    )

    task_thresholds: dict[str, float] = field(default_factory=lambda: {
        "math": 0.85,
        "sentiment": 0.80,
        "code_debug": 0.75,
        "code_gen": 1.0,
        "summarization": 0.60,
        "ner": 0.80,
        "logic": 0.75,
        "factual": 0.80,
    })

    def __post_init__(self) -> None:
        if not self.allowed_models:
            self.allowed_models = [
                "accounts/fireworks/models/gemma-3-4b-it",
                "accounts/fireworks/models/minimax-m3",
                "accounts/fireworks/models/kimi-k2p7-code",
            ]
        if not self.fireworks_api_key:
            import warnings
            warnings.warn("FIREWORKS_API_KEY not set — running in dry-run mode", stacklevel=2)

    @staticmethod
    def from_env(overrides: Optional[dict] = None) -> "Config":
        cfg = Config()
        if overrides:
            for k, v in overrides.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        return cfg
