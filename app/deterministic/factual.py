import re
from typing import Optional

KNOWLEDGE_BASE: dict[str, str] = {
    "capital of france": "Paris",
    "capital of germany": "Berlin",
    "capital of italy": "Rome",
    "capital of spain": "Madrid",
    "capital of united kingdom": "London",
    "capital of japan": "Tokyo",
    "capital of china": "Beijing",
    "capital of india": "New Delhi",
    "capital of canada": "Ottawa",
    "capital of australia": "Canberra",
    "capital of russia": "Moscow",
    "capital of brazil": "Brasilia",
    "inventor of python": "Guido van Rossum",
    "created python": "Guido van Rossum",
    "father of computer": "Charles Babbage",
    "inventor of telephone": "Alexander Graham Bell",
    "inventor of light bulb": "Thomas Edison",
    "first president of the united states": "George Washington",
    "speed of light": "299,792,458 m/s",
    "boiling point of water": "100°C",
    "freezing point of water": "0°C",
    "largest planet": "Jupiter",
    "closest planet to the sun": "Mercury",
    "chemical symbol for water": "H2O",
    "chemical symbol for gold": "Au",
}

QUESTION_PATTERNS = [
    (re.compile(r"what\s+is\s+the\s+(capital|largest|smallest|first|last|speed|boiling|freezing|chemical)\s+(.*)", re.I), 2),
    (re.compile(r"who\s+(invented|created|discovered)\s+(.*)", re.I), 2),
    (re.compile(r"who\s+is\s+the\s+(.*)\s+of\s+(.*)", re.I), 1),
    (re.compile(r"(?:what|which)\s+(?:is|are)\s+(.*)", re.I), 1),
]


def solve(prompt: str) -> tuple[Optional[str], Optional[float]]:
    lower = prompt.lower().strip()

    if any(kw in lower for kw in ("sentiment", "summarize", "calculate", "solve", "write code", "debug", "extract", "fix")):
        return None, None

    for k, v in KNOWLEDGE_BASE.items():
        if k in lower or lower in k:
            return v, 0.92

    if not any(kw in lower for kw in ("what", "who", "which", "where", "when", "how many", "how much", "define", "explain")):
        return None, None

    for pattern, group_idx in QUESTION_PATTERNS:
        m = pattern.search(lower)
        if m:
            key = m.group(group_idx).strip().rstrip("?.,!;:")
            for kb_key, kb_value in KNOWLEDGE_BASE.items():
                if kb_key in key or key in kb_key:
                    return kb_value, 0.92

    return None, None
