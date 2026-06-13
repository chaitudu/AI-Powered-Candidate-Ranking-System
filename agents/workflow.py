"""LangGraph-style orchestration for the ranking workflow."""

from __future__ import annotations

from typing import Any

from agents.candidate_understanding import CandidateUnderstandingAgent
from agents.job_understanding import JobUnderstandingAgent
from agents.ranking_agent import RankingAgent
from agents.semantic_retrieval import SemanticRetrievalAgent
from core.config import Settings
from core.logging_setup import setup_logging
from core.schemas import CandidateProfileStructured, JobProfile, RankedCandidate, RankingResponse

logger = setup_logging()


class RankingWorkflow:
    """End-to-end agent pipeline (LangGraph-compatible sequential workflow)."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.job_agent = JobUnderstandingAgent(self.settings)
        self.candidate_agent = CandidateUnderstandingAgent(self.settings)
        self.retrieval_agent = SemanticRetrievalAgent(self.settings)
        self.ranking_agent = RankingAgent(self.settings)

        self._graph = None
        try:
            from langgraph.graph import END, StateGraph

            self._graph = self._build_langgraph(StateGraph, END)
        except ImportError:
            logger.info("LangGraph not installed; using sequential workflow engine")

    def _build_langgraph(self, StateGraph, END):
        from typing import TypedDict

        class RankingState(TypedDict, total=False):
            job_text: str
            candidates: list[dict[str, Any]]
            job_profile: JobProfile
            candidate_profiles: list[CandidateProfileStructured]
            retrieved: list[dict[str, Any]]
            ranked: list[RankedCandidate]
            response: RankingResponse

        graph = StateGraph(RankingState)
        graph.add_node("understand_job", lambda s: self._understand_job(s["job_text"]))
        graph.add_node("retrieve_candidates", lambda s: self._retrieve_candidates(s["job_profile"]))
        graph.add_node(
            "understand_candidates",
            lambda s: self._understand_candidates(s.get("retrieved", [])),
        )
        graph.add_node(
            "rank_candidates",
            lambda s: self._rank_candidates(s["job_profile"], s.get("retrieved", [])),
        )
        graph.add_node("finalize", lambda s: self._finalize(s))

        graph.set_entry_point("understand_job")
        graph.add_edge("understand_job", "retrieve_candidates")
        graph.add_edge("retrieve_candidates", "understand_candidates")
        graph.add_edge("understand_candidates", "rank_candidates")
        graph.add_edge("rank_candidates", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    def run(
        self,
        job_text: str,
        candidates: list[dict[str, Any]] | None = None,
        retrieval_top_k: int | None = None,
        final_top_k: int | None = None,
    ) -> RankingResponse:
        if retrieval_top_k:
            self.settings.retrieval_top_k = retrieval_top_k
        if final_top_k:
            self.settings.final_top_k = final_top_k

        if self._graph is not None:
            state = {"job_text": job_text, "candidates": candidates or []}
            result = self._graph.invoke(state)
            response = result.get("response")
            if response is None:
                raise RuntimeError("Ranking workflow did not produce a response")
            return response

        job_profile = self._understand_job(job_text)["job_profile"]
        retrieved = self._retrieve_candidates(job_profile)["retrieved"]
        self._understand_candidates(retrieved)
        ranked = self._rank_candidates(job_profile, retrieved)["ranked"]
        return self._finalize(
            {
                "job_profile": job_profile,
                "retrieved": retrieved,
                "ranked": ranked,
                "candidates": candidates or [],
            }
        )["response"]

    def _understand_job(self, job_text: str) -> dict[str, Any]:
        job_profile = self.job_agent.analyze(job_text)
        logger.info("Job understood: domain=%s skills=%s", job_profile.domain, len(job_profile.skills))
        return {"job_profile": job_profile}

    def _retrieve_candidates(self, job_profile: JobProfile) -> dict[str, Any]:
        retrieved = self.retrieval_agent.retrieve(job_profile, top_k=self.settings.retrieval_top_k)
        logger.info("Semantic retrieval returned %s candidates", len(retrieved))
        return {"retrieved": retrieved}

    def _understand_candidates(self, retrieved: list[dict[str, Any]]) -> dict[str, Any]:
        raw_candidates = [item["candidate"] for item in retrieved]
        profiles = self.candidate_agent.analyze_batch(raw_candidates)
        return {"candidate_profiles": profiles}

    def _rank_candidates(
        self,
        job_profile: JobProfile,
        retrieved: list[dict[str, Any]],
    ) -> dict[str, Any]:
        ranked = self.ranking_agent.rank(
            job_profile,
            retrieved,
            top_k=self.settings.final_top_k,
        )
        return {"ranked": ranked}

    def _finalize(self, state: dict[str, Any]) -> dict[str, Any]:
        response = RankingResponse(
            job_profile=state["job_profile"],
            total_candidates=len(state.get("candidates") or []),
            retrieved_count=len(state.get("retrieved", [])),
            ranked=state.get("ranked", []),
        )
        return {"response": response}
