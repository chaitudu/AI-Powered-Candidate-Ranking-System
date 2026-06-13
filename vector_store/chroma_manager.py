"""ChromaDB integration for semantic candidate retrieval."""

from __future__ import annotations

from typing import Any

from core.config import Settings
from core.logging_setup import setup_logging

logger = setup_logging()


class ChromaManager:
    """Manage candidate vectors in ChromaDB."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._client = None
        self._collection = None

    def _ensure_client(self):
        if self._client is not None:
            return
        import chromadb

        self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.settings.chroma_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_candidates(
        self,
        candidate_ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Insert or update candidate records."""
        self._ensure_client()
        assert self._collection is not None

        batch_size = 5000
        for start in range(0, len(candidate_ids), batch_size):
            end = start + batch_size
            self._collection.upsert(
                ids=candidate_ids[start:end],
                documents=documents[start:end],
                embeddings=embeddings[start:end],
                metadatas=(metadatas[start:end] if metadatas else None),
            )
        logger.info("Upserted %s candidates into Chroma", len(candidate_ids))

    def query(self, query_embedding: list[float], top_k: int = 50) -> dict[str, Any]:
        """Retrieve top-k candidates by vector similarity."""
        self._ensure_client()
        assert self._collection is not None
        return self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def count(self) -> int:
        self._ensure_client()
        assert self._collection is not None
        return self._collection.count()
