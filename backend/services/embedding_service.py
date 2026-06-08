import threading

import numpy as np


class EmbeddingService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode_one(self, text: str) -> np.ndarray:
        vector = self.model.encode([text], convert_to_numpy=True)[0]
        return np.asarray(vector, dtype=float)

    @staticmethod
    def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
