"""Embedding generation and persistence."""

from embeddings.pipeline import EmbeddingPipeline, cosine_similarity, load_embedding_matrix

__all__ = ["EmbeddingPipeline", "cosine_similarity", "load_embedding_matrix"]
