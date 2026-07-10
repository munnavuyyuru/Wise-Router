import re
from typing import Optional


POSITIVE = {
    "great", "amazing", "wonderful", "fantastic", "excellent", "loved",
    "beautiful", "perfect", "outstanding", "superb", "brilliant",
    "delightful", "pleasure", "impressive", "stunning", "magnificent",
    "terrific", "splendid", "marvelous", "incredible", "fabulous",
    "awesome", "best", "happy", "joy", "wonderful", "good", "nice",
    "pleasant", "enjoyable", "satisfying", "charming", "lovely",
    "positive", "recommend", "worth", "exceeded", "exceeded expectations",
    "flawless", "smooth", "easy", "fast", "reliable", "comfortable",
    "friendly", "helpful", "generous", "thoughtful", "elegant",
}

NEGATIVE = {
    "terrible", "awful", "horrible", "disappointed", "hated", "worst",
    "bad", "poor", "dreadful", "atrocious", "miserable", "ugly",
    "disgusting", "repulsive", "offensive", "frustrating", "annoying",
    "mediocre", "inferior", "pathetic", "lousy", "rotten", "abysmal",
    "negative", "useless", "waste", "boring", "dull", "slow",
    "uncomfortable", "rude", "unhelpful", "expensive", "broken",
    "terrible", "horrible", "disappointing", "frustrated", "angry",
    "sad", "depressing", "painful", "difficult", "problem",
}

NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "hardly", "barely", "n't"}
CONTRAST_WORDS = {"but", "however", "although", "though", "yet", "nevertheless"}
INTENSIFIERS = {"very": 1.5, "extremely": 2.0, "absolutely": 2.0, "really": 1.3, "so": 1.3, "highly": 1.5, "incredibly": 2.0, "remarkably": 1.8, "exceptionally": 1.8, "completely": 1.6, "totally": 1.5, "utterly": 2.0, "deeply": 1.5}
DIMINISHERS = {"slightly": 0.5, "somewhat": 0.7, "barely": 0.3, "hardly": 0.3, "a bit": 0.6, "a little": 0.6, "kind of": 0.5, "sort of": 0.5, "fairly": 0.7, "pretty": 0.8}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z\'\_]+", text.lower())


def solve(prompt: str) -> tuple[Optional[str], Optional[float]]:
    lower = prompt.lower().strip()
    if not any(kw in lower for kw in ("sentiment", "feeling", "opinion", "positive", "negative", "review", "classify the")):
        return None, None

    colon_idx = lower.find(":")
    quote_idx = lower.find('"')
    start = max(colon_idx, quote_idx) if colon_idx >= 0 or quote_idx >= 0 else 0
    text = lower[start:] if start > 0 else lower
    text = text.replace('"', "").replace("'", "").strip()

    if len(text) < 5:
        return None, None

    tokens = _tokenize(text)

    contrast_idx = -1
    for cw in CONTRAST_WORDS:
        words = [w for w in tokens if w == cw]
        if words:
            pos = 0
            for w in tokens:
                if w == cw:
                    contrast_idx = pos
                pos += 1

    if contrast_idx >= 0:
        tokens = tokens[contrast_idx + 1:]

    pos_score = 0.0
    neg_score = 0.0
    negate = False

    for i, token in enumerate(tokens):
        if token in NEGATION_WORDS:
            negate = not negate
            continue
        if token in POSITIVE:
            score = 0.5 if negate else 1.0
            if i > 0:
                prev = tokens[i - 1]
                if prev in INTENSIFIERS:
                    score *= INTENSIFIERS[prev]
                elif prev in DIMINISHERS:
                    score *= DIMINISHERS[prev]
            pos_score += score
        elif token in NEGATIVE:
            score = 0.5 if negate else 1.0
            if i > 0:
                prev = tokens[i - 1]
                if prev in INTENSIFIERS:
                    score *= INTENSIFIERS[prev]
                elif prev in DIMINISHERS:
                    score *= DIMINISHERS[prev]
            neg_score += score
        negate = False

    diff = pos_score - neg_score
    total = pos_score + neg_score

    if total < 1:
        return "neutral", 0.60

    normalized = diff / total
    confidence = min(0.5 + abs(normalized) * 0.4, 0.95)

    if normalized > 0.3:
        return "positive", confidence
    elif normalized < -0.3:
        return "negative", confidence
    else:
        return "neutral", min(confidence, 0.70)
