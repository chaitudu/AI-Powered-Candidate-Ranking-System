"""Main ranking pipeline orchestration and CSV export."""

from __future__ import annotations

import csv
from pathlib import Path

from agents.workflow import RankingWorkflow
from core.config import Settings
from core.data_loader import iter_candidates, load_job_description
from core.logging_setup import setup_logging
from core.schemas import RankedCandidate, RankingResponse

logger = setup_logging()


class CandidateRankingPipeline:
    """Production pipeline entrypoint."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.workflow = RankingWorkflow(self.settings)

    def run_full_corpus(
        self,
        job_text: str,
        candidates_path: Path,
        retrieval_top_k: int = 2000,
        final_top_k: int = 100,
    ) -> RankingResponse:
        """Rank against full candidate corpus using precomputed semantic index."""
        self.settings.retrieval_top_k = retrieval_top_k
        self.settings.final_top_k = final_top_k

        from agents.job_understanding import JobUnderstandingAgent
        from agents.ranking_agent import RankingAgent
        from agents.semantic_retrieval import SemanticRetrievalAgent

        total = sum(1 for _ in iter_candidates(candidates_path))
        job_profile = JobUnderstandingAgent(self.settings).analyze(job_text)
        retrieved = SemanticRetrievalAgent(self.settings).retrieve(job_profile, top_k=retrieval_top_k)
        ranked = RankingAgent(self.settings).rank(job_profile, retrieved, top_k=final_top_k)

        return RankingResponse(
            job_profile=job_profile,
            total_candidates=total,
            retrieved_count=len(retrieved),
            ranked=ranked,
        )


def export_submission_csv(ranked: list[RankedCandidate], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for item in ranked:
            writer.writerow(
                [item.candidate_id, item.rank, f"{item.submission_score:.4f}", item.reasoning]
            )
    logger.info("Wrote submission CSV: %s", output_path)


def export_ranked_candidates_csv(ranked: list[RankedCandidate], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Rank",
                "Candidate_ID",
                "Name",
                "Score",
                "Skill_Match",
                "Experience_Match",
                "Behavior_Match",
                "Project_Relevance",
                "Learning_Potential",
                "Recommendation",
                "Reason",
            ]
        )
        for item in ranked:
            writer.writerow(
                [
                    item.rank,
                    item.candidate_id,
                    item.name,
                    item.score,
                    item.components.skill_match,
                    item.components.experience_match,
                    item.components.behavior_match,
                    item.components.project_relevance,
                    item.components.learning_potential,
                    item.recommendation,
                    item.reasoning,
                ]
            )
    logger.info("Wrote ranked candidates CSV: %s", output_path)


def export_json_report(response: RankingResponse, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(response.to_json(), encoding="utf-8")
    logger.info("Wrote JSON report: %s", output_path)
