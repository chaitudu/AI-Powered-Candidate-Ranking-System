"""Local embedding pipeline with sentence-transformers or sklearn fallback."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from core.config import Settings
from core.logging_setup import setup_logging

logger = setup_logging()


class EmbeddingPipeline:
    """Generate and persist dense embeddings for jobs and candidates."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.model = None
        self.backend = "unknown"
        self._vectorizer = None

    def _load_sentence_transformer(self):
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", self.settings.embedding_model)
        self.model = SentenceTransformer(self.settings.embedding_model)
        self.backend = "sentence-transformers"

    def _load_sklearn_vectorizer(self):
        from sklearn.feature_extraction.text import HashingVectorizer

        logger.info("Using sklearn HashingVectorizer fallback (offline, no extra deps)")
        self._vectorizer = HashingVectorizer(
            n_features=8192,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            ngram_range=(1, 2),
        )
        self.backend = "sklearn-hashing"

    def _ensure_backend(self):
        if self.backend != "unknown":
            return
        try:
            self._load_sentence_transformer()
        except Exception as exc:
            logger.warning("sentence-transformers unavailable (%s); using sklearn fallback", exc)
            self._load_sklearn_vectorizer()

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        self._ensure_backend()
        if self.backend == "sentence-transformers":
            batch = batch_size or self.settings.batch_size
            vectors = self.model.encode(
                texts,
                batch_size=batch,
                show_progress_bar=len(texts) > 100,
                normalize_embeddings=True,
            )
            return np.asarray(vectors, dtype=np.float32)

        assert self._vectorizer is not None
        matrix = self._vectorizer.transform(texts)
        return np.asarray(matrix.toarray(), dtype=np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def build_candidate_embeddings(
        self,
        candidate_ids: list[str],
        documents: list[str],
    ) -> np.ndarray:
        if len(candidate_ids) != len(documents):
            raise ValueError("candidate_ids and documents length mismatch")
        return self.encode(documents)

    def save_embedding_store(
        self,
        candidate_ids: list[str],
        embeddings: np.ndarray,
        job_text: str | None = None,
        job_embedding: np.ndarray | None = None,
    ) -> None:
        out_dir = self.settings.embeddings_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "candidate_embeddings.npy", embeddings)
        (out_dir / "candidate_ids.json").write_text(
            json.dumps(candidate_ids, ensure_ascii=False),
            encoding="utf-8",
        )
        meta = {
            "model": self.settings.embedding_model,
            "backend": self.backend,
            "count": len(candidate_ids),
            "dimension": int(embeddings.shape[1]) if len(embeddings) else 0,
        }
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        if self._vectorizer is not None:
            with (out_dir / "vectorizer.pkl").open("wb") as handle:
                pickle.dump(self._vectorizer, handle)

        if job_text is not None and job_embedding is not None:
            (out_dir / "job_description.txt").write_text(job_text, encoding="utf-8")
            np.save(out_dir / "job_embedding.npy", job_embedding)

        logger.info("Saved embedding store with %s vectors to %s", len(candidate_ids), out_dir)

    def load_embedding_store(self) -> tuple[list[str], np.ndarray, np.ndarray | None, str | None]:
        out_dir = self.settings.embeddings_dir
        ids = json.loads((out_dir / "candidate_ids.json").read_text(encoding="utf-8"))
        embeddings = np.load(out_dir / "candidate_embeddings.npy")

        meta_path = out_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.backend = meta.get("backend", self.backend)

        vectorizer_path = out_dir / "vectorizer.pkl"
        if vectorizer_path.exists():
            with vectorizer_path.open("rb") as handle:
                self._vectorizer = pickle.load(handle)
            self.backend = "sklearn-hashing"

        job_embedding = None
        job_text = None
        job_emb_path = out_dir / "job_embedding.npy"
        if job_emb_path.exists():
            job_embedding = np.load(job_emb_path)
        job_text_path = out_dir / "job_description.txt"
        if job_text_path.exists():
            job_text = job_text_path.read_text(encoding="utf-8")
        return ids, embeddings, job_embedding, job_text


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if query.ndim == 1:
        query = query.reshape(1, -1)
    query = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
    matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    return (matrix @ query.T).reshape(-1)


def load_embedding_matrix(settings: Settings | None = None) -> tuple[list[str], np.ndarray]:
    pipeline = EmbeddingPipeline(settings)
    ids, embeddings, _, _ = pipeline.load_embedding_store()
    return ids, embeddings
