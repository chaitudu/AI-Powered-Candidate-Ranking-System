#!/usr/bin/env python3
"""Precompute candidate embeddings and optional Chroma index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import get_settings
from core.data_loader import iter_candidates, load_job_description
from core.logging_setup import setup_logging
from embeddings.pipeline import EmbeddingPipeline
from preprocessing.feature_engineering import build_candidate_document
from vector_store.chroma_manager import ChromaManager

logger = setup_logging()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute embeddings for offline ranking")
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--job", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick tests")
    parser.add_argument("--sync-chroma", action="store_true", help="Also upsert vectors to ChromaDB")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    if args.candidates:
        settings.candidates_path = args.candidates
    if args.job:
        settings.job_description_path = args.job

    pipeline = EmbeddingPipeline(settings)
    candidate_ids: list[str] = []
    documents: list[str] = []

    for idx, candidate in enumerate(iter_candidates(settings.candidates_path)):
        candidate_ids.append(candidate["candidate_id"])
        documents.append(build_candidate_document(candidate))
        if args.limit and idx + 1 >= args.limit:
            break
        if (idx + 1) % 10000 == 0:
            logger.info("Prepared %s candidate documents", idx + 1)

    logger.info("Encoding %s candidates...", len(documents))
    embeddings = pipeline.build_candidate_embeddings(candidate_ids, documents)

    job_text = load_job_description(settings.job_description_path)
    job_embedding = pipeline.encode_single(job_text)
    pipeline.save_embedding_store(candidate_ids, embeddings, job_text, job_embedding)

    if args.sync_chroma:
        logger.info("Syncing to ChromaDB...")
        chroma = ChromaManager(settings)
        metadatas = [{"candidate_id": cid} for cid in candidate_ids]
        chroma.upsert_candidates(
            candidate_ids,
            documents,
            embeddings.tolist(),
            metadatas=metadatas,
        )

    logger.info("Precomputation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
