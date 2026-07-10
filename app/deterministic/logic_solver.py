import re
from typing import Optional


COMPARATIVE_SIGN: dict[str, int] = {
    "taller": 1, "tallest": 1,
    "older": 1, "oldest": 1,
    "faster": 1, "fastest": 1,
    "heavier": 1, "heaviest": 1,
    "bigger": 1, "biggest": 1,
    "higher": 1, "highest": 1,
    "greater": 1, "greatest": 1,
    "shorter": -1, "shortest": -1,
    "younger": -1, "youngest": -1,
    "slower": -1, "slowest": -1,
    "lighter": -1, "lightest": -1,
    "smaller": -1, "smallest": -1,
    "lower": -1, "lowest": -1,
    "less": -1, "least": -1,
    "before": 1, "after": -1,
}


def solve(prompt: str) -> tuple[Optional[str], Optional[float]]:
    lower = prompt.lower()

    relations = re.findall(
        r"(\w+)\s+is\s+(taller|older|faster|heavier|shorter|younger|slower|lighter|bigger|smaller|higher|lower|greater|less|tallest|oldest|fastest|heaviest|shortest|youngest|slowest|lightest|smallest|highest|lowest|greatest|least|before|after)\s+than\s+(\w+)",
        lower,
    )

    if not relations:
        return None, None

    score_map: dict[str, int] = {}
    for a, rel, b in relations:
        if a not in score_map:
            score_map[a] = 0
        if b not in score_map:
            score_map[b] = 0
        sign = COMPARATIVE_SIGN.get(rel, 0)
        score_map[a] += sign
        score_map[b] -= sign

    if not score_map:
        return None, None

    superlative_match = re.search(
        r"(tallest|oldest|fastest|heaviest|shortest|youngest|slowest|lightest|"
        r"smallest|highest|lowest|greatest|least|biggest)",
        lower,
    )
    if superlative_match:
        sup = superlative_match.group(1)
        sign = COMPARATIVE_SIGN.get(sup, 1)
        if sign == 1:
            best = max(score_map, key=score_map.get)
        else:
            best = min(score_map, key=score_map.get)
        return best.lower(), 0.85

    return None, None
