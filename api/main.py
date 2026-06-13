"""FastAPI backend for candidate ranking."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise ImportError(
        "FastAPI and pydantic are required for the API. Install via requirements.txt"
    ) from exc

from core.config import get_settings
from core.data_loader import load_job_description
from core.logging_setup import setup_logging
from core.schemas import RankingResponse
from evaluation.metrics import build_evaluation_report
from pipeline import CandidateRankingPipeline, export_ranked_candidates_csv, export_submission_csv

logger = setup_logging()
settings = get_settings()
app = FastAPI(title="AI Candidate Ranking API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RankRequest(BaseModel):
    job_description: str | None = None
    retrieval_top_k: int = Field(default=500, ge=10, le=5000)
    final_top_k: int = Field(default=100, ge=1, le=500)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/dataset/stats")
def dataset_stats() -> dict:
    from core.data_loader import analyze_dataset

    return analyze_dataset(settings.candidates_path, sample_size=5000)


@app.post("/rank")
def rank_candidates(request: RankRequest) -> dict:
    try:
        job_text = request.job_description or load_job_description(settings.job_description_path)
        pipeline = CandidateRankingPipeline(settings)
        response = pipeline.run_full_corpus(
            job_text=job_text,
            candidates_path=settings.candidates_path,
            retrieval_top_k=request.retrieval_top_k,
            final_top_k=request.final_top_k,
        )
        return response.to_dict()
    except Exception as exc:
        logger.exception("Ranking failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/rank/export")
def rank_and_export(request: RankRequest) -> dict:
    job_text = request.job_description or load_job_description(settings.job_description_path)
    pipeline = CandidateRankingPipeline(settings)
    response = pipeline.run_full_corpus(
        job_text=job_text,
        candidates_path=settings.candidates_path,
        retrieval_top_k=request.retrieval_top_k,
        final_top_k=request.final_top_k,
    )
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    submission_path = settings.outputs_dir / "api_submission.csv"
    ranked_path = settings.outputs_dir / "api_ranked_candidates.csv"
    export_submission_csv(response.ranked, submission_path)
    export_ranked_candidates_csv(response.ranked, ranked_path)
    metrics = build_evaluation_report(response.ranked)
    return {
        "submission_csv": str(submission_path),
        "ranked_csv": str(ranked_path),
        "metrics": metrics,
        "top_5": [r.to_dict() for r in response.ranked[:5]],
    }
