from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


@dataclass
class InMemoryVectorIndex:
    vectors: list[np.ndarray] = field(default_factory=list)
    metadata: list[dict[str, Any]] = field(default_factory=list)

    def add(self, vector: np.ndarray, meta: dict[str, Any]) -> None:
        self.vectors.append(_normalize(vector.astype(np.float32)))
        self.metadata.append(meta)

    def search(self, vector: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.vectors:
            return []
        q = _normalize(vector.astype(np.float32))
        scores = [float(np.dot(q, v)) for v in self.vectors]
        order = np.argsort(scores)[::-1][:top_k]
        return [{**self.metadata[i], "score": float(scores[i])} for i in order]


def build_faiss_index(vectors: np.ndarray):
    if faiss is None:
        raise RuntimeError("faiss-cpu is not installed. Install with `pip install .[vector]`.")
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = vectors / norms
    index.add(normalized.astype(np.float32))
    return index
