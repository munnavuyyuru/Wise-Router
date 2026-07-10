from app.config import Config
from app.classifier.task_classifier import TaskClassifier
from app.llm.fireworks_client import FireworksClient, TokenTracker
from app.router import route_prompt


def _make_cfg(task_type: str, threshold: float = 0.5) -> Config:
    cfg = Config()
    cfg.task_thresholds = {t: 0.5 for t in cfg.task_thresholds}
    cfg.task_thresholds[task_type] = threshold
    return cfg


class TestRouter:
    def test_route_math_deterministic(self):
        cfg = _make_cfg("math")
        clf = TaskClassifier()
        prompts = [
            "What is 2+2?",
            "What is the capital of France?",
            "Solve for x: x + 3 = 7",
        ]
        labels = ["math", "code_gen", "math"]
        clf.fit(prompts, labels)

        answer = route_prompt("What is 18 + 24?", cfg, clf, None)
        assert answer == "42"

    def test_route_no_classifier(self):
        cfg = _make_cfg("code_gen")
        clf = TaskClassifier()
        answer = route_prompt("Hello world", cfg, clf, None)
        assert answer

    def test_route_sentiment_deterministic(self):
        cfg = _make_cfg("sentiment")
        clf = TaskClassifier()
        prompts = [
            "What is the sentiment of this review: I love this product",
            "Sentiment analysis: This is terrible, worst product ever",
            "Classify the sentiment: The meeting is scheduled for Tuesday at 3 PM",
            "What is the sentiment of this text: I am so happy today",
            "Sentiment analysis: The food was amazing and delicious",
            "Classify the sentiment: The battery life is disappointing",
        ]
        labels = ["sentiment"] * 6
        clf.fit(prompts + ["Write a Python function to sort a list", "What is 2+2?"], labels + ["code_gen", "math"])

        answer = route_prompt('What is the sentiment: "Great product, loved it!"', cfg, clf, None)
        assert answer in ("positive", "negative", "neutral")

    def test_route_code_debug(self):
        cfg = _make_cfg("code_debug")
        clf = TaskClassifier()
        prompts = [
            "fix this bug in the code",
            "debug this python function",
            "hello world test",
        ]
        labels = ["code_debug", "code_debug", "code_gen"]
        clf.fit(prompts, labels)

        answer = route_prompt("def add(a, b):\n    a + b", cfg, clf, None)
        assert "return" in answer.lower()

    def test_route_ner(self):
        cfg = _make_cfg("ner")
        clf = TaskClassifier()
        prompts = [
            "extract emails from this text",
            "find all phone numbers",
            "hello world test",
        ]
        labels = ["ner", "ner", "code_gen"]
        clf.fit(prompts, labels)

        answer = route_prompt("Extract emails: test@example.com", cfg, clf, None)
        assert "emails" in answer

    def test_route_logic(self):
        cfg = _make_cfg("logic")
        clf = TaskClassifier()
        prompts = [
            "Alice is taller than Bob. Who is tallest?",
            "If it rains, the ground gets wet. It rained. Conclusion?",
            "hello world test",
        ]
        labels = ["logic", "logic", "code_gen"]
        clf.fit(prompts, labels)

        answer = route_prompt("Alice is taller than Bob. Who is tallest?", cfg, clf, None)
        assert answer == "alice"

    def test_route_fireworks_dry_run(self):
        cfg = _make_cfg("code_gen")
        clf = TaskClassifier()
        clf.fit(
            ["write code for sorting", "What is 2+2?"],
            ["code_gen", "math"],
        )
        answer = route_prompt("write sorting function", cfg, clf, None)
        assert "[dry-run]" in answer

    def test_route_with_fireworks_client_passes_tracker(self):
        cfg = _make_cfg("code_gen")
        clf = TaskClassifier()
        clf.fit(
            ["write code for sorting", "What is 2+2?"],
            ["code_gen", "math"],
        )
        tracker = TokenTracker()
        fw = FireworksClient(cfg, token_tracker=tracker)
        route_prompt("write sorting function", cfg, clf, fw)
        assert fw.token_tracker is tracker
