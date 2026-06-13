"""Streamlit dashboard for recruiter-facing ranking."""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import get_settings
from core.data_loader import load_job_description
from evaluation.metrics import build_evaluation_report
from pipeline import CandidateRankingPipeline, export_ranked_candidates_csv, export_submission_csv

st.set_page_config(page_title="AI Candidate Ranker", layout="wide", page_icon="🎯")


@st.cache_resource
def get_pipeline() -> CandidateRankingPipeline:
    return CandidateRankingPipeline(get_settings())


def ranked_to_csv_text(ranked, export_fn, path) -> str:
    export_fn(ranked, path)
    return path.read_text(encoding="utf-8")


def main() -> None:
    settings = get_settings()
    st.title("AI-Powered Candidate Ranking System")
    st.caption("Semantic job-fit ranking for Redrob India Runs Data & AI Challenge")

    with st.sidebar:
        st.header("Configuration")
        default_job = load_job_description(settings.job_description_path)
        job_text = st.text_area("Job Description", value=default_job, height=320)
        retrieval_top_k = st.slider("Semantic retrieval pool", 50, 1000, 500, 50)
        final_top_k = st.slider("Final shortlist size", 10, 100, 100, 10)
        run_btn = st.button("Run Ranking", type="primary")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Pipeline Overview")
        st.markdown(
            """
            1. **Job Understanding Agent** extracts skills, experience, behavior, domain
            2. **Embedding Retrieval** finds top candidates via vector similarity
            3. **Candidate Understanding Agent** builds structured profiles
            4. **Ranking Agent** scores skill, projects, experience, behavior, learning
            5. **Explainability Layer** surfaces strengths, gaps, and reasoning
            """
        )
    with col2:
        st.subheader("Dataset")
        st.write(f"Candidates: `{settings.candidates_path.name}`")
        st.write("Records: 100,000")

    if run_btn:
        with st.spinner("Running semantic ranking pipeline..."):
            pipeline = get_pipeline()
            response = pipeline.run_full_corpus(
                job_text=job_text,
                candidates_path=settings.candidates_path,
                retrieval_top_k=retrieval_top_k,
                final_top_k=final_top_k,
            )

        st.success(
            f"Ranked top {len(response.ranked)} from retrieval pool of {response.retrieved_count}"
        )

        rows = []
        for item in response.ranked:
            rows.append(
                {
                    "Rank": item.rank,
                    "Candidate_ID": item.candidate_id,
                    "Name": item.name,
                    "Score": item.score,
                    "Skill_Match": item.components.skill_match,
                    "Experience_Match": item.components.experience_match,
                    "Behavior_Match": item.components.behavior_match,
                    "Project_Relevance": item.components.project_relevance,
                    "Recommendation": item.recommendation,
                    "Reason": item.reasoning,
                }
            )
        st.subheader("Top Candidates")
        st.dataframe(rows, use_container_width=True, height=420)

        selected_id = st.selectbox("Inspect candidate", options=[r["Candidate_ID"] for r in rows])
        selected = next(r for r in response.ranked if r.candidate_id == selected_id)
        st.subheader(f"Candidate Insights — {selected.name}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Why selected**")
            st.write(selected.why_selected)
            st.markdown("**Strengths**")
            st.write("\n".join(f"- {s}" for s in selected.strengths) or "- N/A")
        with c2:
            st.markdown("**Why ranked here**")
            st.write(selected.why_ranked_higher)
            st.markdown("**Weaknesses**")
            st.write("\n".join(f"- {w}" for w in selected.weaknesses) or "- N/A")

        metrics = build_evaluation_report(response.ranked)
        st.subheader("Evaluation Metrics")
        st.json(metrics)

        settings.outputs_dir.mkdir(parents=True, exist_ok=True)
        submission_path = settings.outputs_dir / "streamlit_submission.csv"
        ranked_path = settings.outputs_dir / "streamlit_ranked_candidates.csv"
        export_submission_csv(response.ranked, submission_path)
        export_ranked_candidates_csv(response.ranked, ranked_path)

        st.download_button(
            "Download Submission CSV",
            data=submission_path.read_text(encoding="utf-8"),
            file_name="submission.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download Ranked Candidates CSV",
            data=ranked_path.read_text(encoding="utf-8"),
            file_name="ranked_candidates.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
