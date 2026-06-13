"""Semantic retrieval agent using vector similarity."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.config import Settings
from core.logging_setup import setup_logging
from core.schemas import JobProfile
from embeddings.pipeline import EmbeddingPipeline, cosine_similarity, load_embedding_matrix
from preprocessing.feature_engineering import build_candidate_features
from vector_store.chroma_manager import ChromaManager

logger = setup_logging()


class SemanticRetrievalAgent:
    """Retrieve top candidates using semantic vector search (no keyword matching)."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.embedding_pipeline = EmbeddingPipeline(self.settings)
        self.chroma = ChromaManager(self.settings)

    def retrieve(
        self,
        job_profile: JobProfile,
        candidates: list[dict[str, Any]] | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve top-k candidates by semantic similarity.
        Uses precomputed embeddings when available for speed.
        """
        top_k = top_k or self.settings.retrieval_top_k
        job_text = job_profile.raw_text or " ".join(job_profile.skills + [job_profile.domain])

        try:
            ids, matrix, job_embedding, _ = self.embedding_pipeline.load_embedding_store()
            if job_embedding is None:
                job_embedding = self.embedding_pipeline.encode_single(job_text)
            sims = cosine_similarity(job_embedding, matrix)
            top_indices = np.argpartition(-sims, min(top_k, len(sims) - 1))[:top_k]
            top_indices = top_indices[np.argsort(-sims[top_indices])]
            retrieved_ids = [ids[i] for i in top_indices]

            if candidates is None:
                from core.data_loader import load_candidates_by_ids

                candidate_map = load_candidates_by_ids(
                    self.settings.candidates_path,
                    set(retrieved_ids),
                )
            else:
                candidate_map = {c["candidate_id"]: c for c in candidates}

            results = []
            for idx, cid in zip(top_indices, retrieved_ids):
                cand = candidate_map.get(cid)
                if not cand:
                    continue
                features = build_candidate_features(cand)
                results.append(
                    {
                        "candidate": cand,
                        "features": features,
                        "semantic_score": float(sims[idx]),
                    }
                )
            logger.info("Retrieved %s candidates via embedding store", len(results))
            return results
        except FileNotFoundError:
            logger.info("Embedding store missing; computing on-the-fly retrieval")
            return self._retrieve_live(job_text, candidates, top_k)

    def _retrieve_live(
        self,
        job_text: str,
        candidates: list[dict[str, Any]] | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if candidates is None:
            from core.data_loader import load_candidates

            candidates = load_candidates(self.settings.candidates_path)

        job_vec = self.embedding_pipeline.encode_single(job_text)
        docs = [build_candidate_features(c)["document"] for c in candidates]
        matrix = self.embedding_pipeline.encode(docs)
        sims = cosine_similarity(job_vec, matrix)
        top_indices = np.argpartition(-sims, min(top_k, len(sims) - 1))[:top_k]
        top_indices = top_indices[np.argsort(-sims[top_indices])]

        results = []
        for idx in top_indices:
            cand = candidates[int(idx)]
            features = build_candidate_features(cand)
            results.append(
                {
                    "candidate": cand,
                    "features": features,
                    "semantic_score": float(sims[idx]),
                }
            )
        return results

    def retrieve_with_chroma(self, job_text: str, top_k: int | None = None) -> list[str]:
        """Alternative retrieval path via ChromaDB."""
        top_k = top_k or self.settings.retrieval_top_k
        query_vec = self.embedding_pipeline.encode_single(job_text).tolist()
        if self.chroma.count() == 0:
            raise RuntimeError("Chroma collection is empty. Run precompute_embeddings.py first.")
        result = self.chroma.query(query_vec, top_k=top_k)
        return result.get("ids", [[]])[0]
