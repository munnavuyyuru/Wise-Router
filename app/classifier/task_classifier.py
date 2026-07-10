import os
import pickle
from typing import Any, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


TASK_TYPES = [
    "math",
    "sentiment",
    "code_debug",
    "code_gen",
    "summarization",
    "ner",
    "logic",
    "factual",
]


class TaskClassifier:
    def __init__(
        self,
        model_path: Optional[str] = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        self.model_path = model_path
        self.embedding_model_name = embedding_model_name
        self._encoder: Any = None
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.is_fitted = False

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.embedding_model_name)
        return self._encoder

    def _embed(self, text: str) -> list[float]:
        encoder = self._get_encoder()
        emb = encoder.encode(text, normalize_embeddings=True)
        return emb.tolist()

    def _prepare(self, texts: list[str]) -> np.ndarray:
        embs = [self._embed(t) for t in texts]
        return np.array(embs, dtype=float)

    def fit(self, prompts: list[str], labels: list[str]) -> None:
        if len(prompts) != len(labels):
            raise ValueError("prompts and labels must have same length")
        unique = set(labels)
        for t in unique:
            if t not in TASK_TYPES:
                raise ValueError(f"Unknown label {t!r}, expected one of {TASK_TYPES}")
        X = self._prepare(prompts)
        y = np.array(labels)
        Xs = self.scaler.fit_transform(X)
        self.classifier.fit(Xs, y)
        self.is_fitted = True

    def predict(self, text: str) -> tuple[str, float]:
        if not self.is_fitted:
            return "code_gen", 0.5
        X = self._prepare([text])
        Xs = self.scaler.transform(X)
        probs = self.classifier.predict_proba(Xs)[0]
        idx = int(np.argmax(probs))
        task_type = self.classifier.classes_[idx]
        confidence = float(probs[idx])
        return task_type, confidence

    def predict_type(self, text: str) -> str:
        return self.predict(text)[0]

    def predict_confidence(self, text: str) -> float:
        return self.predict(text)[1]

    def save(self, path: str) -> None:
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before saving")
        data = {
            "scaler": self.scaler,
            "classifier": self.classifier,
            "is_fitted": self.is_fitted,
            "embedding_model_name": self.embedding_model_name,
            "classes": self.classifier.classes_.tolist(),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.scaler = data["scaler"]
        self.classifier = data["classifier"]
        self.is_fitted = data["is_fitted"]
        self.embedding_model_name = data.get("embedding_model_name", "all-MiniLM-L6-v2")
