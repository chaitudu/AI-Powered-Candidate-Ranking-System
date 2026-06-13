#!/usr/bin/env python3
"""
End-to-end offline ranking CLI for hackathon submission.

Usage:
  python rank.py --candidates ./data/candidates.jsonl --out ./outputs/submission.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import Settings, get_settings
from core.data_loader import load_job_description
from core.logging_setup import setup_logging
from pipeline import (
    CandidateRankingPipeline,
    export_json_report,
    export_ranked_candidates_csv,
    export_submission_csv,
)

logger = setup_logging()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Candidate Ranking System")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="Path to candidates.jsonl",
    )
    parser.add_argument(
        "--job",
        type=Path,
        default=None,
        help="Path to job description text file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "submission.csv",
        help="Submission CSV output path",
    )
    parser.add_argument(
        "--ranked-out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "ranked_candidates.csv",
        help="Extended ranked candidates CSV",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "ranking_report.json",
        help="JSON explainability report",
    )
    parser.add_argument("--retrieval-top-k", type=int, default=2000)
    parser.add_argument("--final-top-k", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()

    if args.candidates:
        settings.candidates_path = args.candidates
    if args.job:
        settings.job_description_path = args.job

    settings.offline_mode = True
    settings.use_llm_agents = False
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    job_text = load_job_description(settings.job_description_path)

    pipeline = CandidateRankingPipeline(settings)
    response = pipeline.run_full_corpus(
        job_text=job_text,
        candidates_path=settings.candidates_path,
        retrieval_top_k=args.retrieval_top_k,
        final_top_k=args.final_top_k,
    )

    export_submission_csv(response.ranked, args.out)
    export_ranked_candidates_csv(response.ranked, args.ranked_out)
    export_json_report(response, args.report_out)

    elapsed = time.time() - start
    logger.info(
        "Ranked %s candidates (retrieved %s) in %.1fs",
        len(response.ranked),
        response.retrieved_count,
        elapsed,
    )
    logger.info("Submission written to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
