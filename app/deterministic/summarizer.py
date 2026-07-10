import re
from typing import Optional


def _split_sentences(text: str) -> list[str]:
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if len(s.strip()) > 10]


def _word_frequencies(sentences: list[str]) -> dict[str, float]:
    freq: dict[str, float] = {}
    for sent in sentences:
        words = re.findall(r'\w+', sent.lower())
        for w in words:
            freq[w] = freq.get(w, 0) + 1
    total = sum(freq.values()) or 1
    return {w: c / total for w, c in freq.items()}


def _score_sentence(sent: str, freq: dict[str, float]) -> float:
    words = re.findall(r'\w+', sent.lower())
    if not words:
        return 0
    return sum(freq.get(w, 0) for w in words) / len(words)


def solve(prompt: str) -> tuple[Optional[str], Optional[float]]:
    lower = prompt.lower()

    if "summarize" not in lower and "summary" not in lower:
        return None, None

    colon_idx = lower.find(":")
    text = prompt[colon_idx + 1:].strip() if colon_idx >= 0 else prompt

    colon_idx2 = text.lower().find(":")
    if colon_idx2 >= 0 and colon_idx2 < len(text) // 2:
        text = text[colon_idx2 + 1:].strip()

    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return None, None

    num_sentences = 3
    num_match = re.search(r"(\d+)\s+sentence", lower)
    if num_match:
        num_sentences = min(int(num_match.group(1)), len(sentences))

    freq = _word_frequencies(sentences)

    scored = [(i, _score_sentence(s, freq)) for i, s in enumerate(sentences)]
    scored.sort(key=lambda x: x[1], reverse=True)

    top_indices = sorted([idx for idx, _ in scored[:num_sentences]])
    summary = " ".join(sentences[i] for i in top_indices)

    confidence = 0.75 if num_match else 0.65

    return summary, confidence
