import re
from typing import Optional

EMAIL_RE = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.]+\b')
URL_RE = re.compile(r'https?://[^\s<>"\'\]\[{}|\\^`]+')
MONEY_RE = re.compile(r'\$\d+(?:,\d{3})*(?:\.\d{2})?')
DATE_RE = re.compile(
    r'\b(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|'
    r'\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}|'
    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})\b',
    re.I
)
PHONE_RE = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
PERCENT_RE = re.compile(r'\b(\d+(?:\.\d+)?)\s*%')
NUMBER_RE = re.compile(r'\b\d+(?:\.\d+)?\b')


def solve(prompt: str) -> tuple[Optional[str], Optional[float]]:
    lower = prompt.lower()
    if "extract" not in lower and "find" not in lower and "identify" not in lower and "ner" not in lower:
        return None, None

    result: dict[str, list[str]] = {}

    emails = EMAIL_RE.findall(prompt)
    if emails:
        result["emails"] = emails

    urls = URL_RE.findall(prompt)
    if urls:
        result["urls"] = urls

    money = MONEY_RE.findall(prompt)
    if money:
        result["money"] = money

    dates = DATE_RE.findall(prompt)
    if dates:
        result["dates"] = dates

    phones = PHONE_RE.findall(prompt)
    if phones:
        result["phones"] = phones

    percents = PERCENT_RE.findall(prompt)
    if percents:
        result["percentages"] = [f"{p}%" for p in percents]

    numbers = NUMBER_RE.findall(prompt)
    if numbers:
        excluded = set()
        for m in money:
            excluded.add(m.replace("$", "").replace(",", ""))
        for p in percents:
            excluded.add(p)
        filtered = [n for n in numbers if n not in excluded]
        if filtered:
            result["numbers"] = filtered

    if not result:
        return None, None

    has_ner_signal = bool(re.search(r'\b(extract|find|identify|list|ner)\b', lower))
    confidence = 0.95 if has_ner_signal and any(len(v) > 0 for v in result.values()) else 0.80

    lines = []
    for k, v in result.items():
        lines.append(f"{k}: {', '.join(v)}")
    return "\n".join(lines), confidence
