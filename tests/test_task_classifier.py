import pytest
from app.classifier.task_classifier import TaskClassifier, TASK_TYPES


class TestTaskClassifier:
    def test_initialization(self):
        clf = TaskClassifier()
        assert not clf.is_fitted

    def test_fit_and_predict(self):
        clf = TaskClassifier()
        prompts = ["What is 2+2?", "I love this!", "Fix this code", "Write a sort function"]
        labels = ["math", "sentiment", "code_debug", "code_gen"]
        clf.fit(prompts, labels)
        assert clf.is_fitted

        task_type, conf = clf.predict("What is 10 + 20?")
        assert task_type in TASK_TYPES
        assert 0 <= conf <= 1

    def test_save_and_load(self, tmp_path):
        clf = TaskClassifier()
        prompts = ["Hello", "What is 2+2?"]
        labels = ["code_gen", "math"]
        clf.fit(prompts, labels)

        path = tmp_path / "model.pkl"
        clf.save(str(path))
        assert path.exists()

        clf2 = TaskClassifier(str(path))
        assert clf2.is_fitted
        assert clf2.predict("Hello")[0] == clf.predict("Hello")[0]
