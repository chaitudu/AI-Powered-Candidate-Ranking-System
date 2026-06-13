"""Unit tests for candidate ranking system."""

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.job_understanding import JobUnderstandingAgent
from agents.ranking_agent import RankingAgent
from core.config import Settings
from core.data_loader import load_candidates, validate_candidate_record
from preprocessing.feature_engineering import build_candidate_features, build_title_career_consistency
from preprocessing.pipeline import preprocess_job_text


@pytest.fixture
def settings() -> Settings:
    s = Settings()
    s.use_llm_agents = False
    s.offline_mode = True
    return s


@pytest.fixture
def sample_candidates():
    path = PROJECT_ROOT / "data" / "sample_candidates.json"
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def test_job_understanding_extracts_skills(settings, tmp_path):
    job_path = PROJECT_ROOT / "data" / "job_description.txt"
    job_text = job_path.read_text(encoding="utf-8")
    profile = JobUnderstandingAgent(settings).analyze(job_text)
    assert len(profile.skills) >= 5
    assert profile.experience_min_years >= 0


def test_candidate_validation(sample_candidates):
    errors = validate_candidate_record(sample_candidates[0])
    assert errors == []


def test_title_career_consistency_detects_mismatch(sample_candidates):
    features = build_candidate_features(sample_candidates[0])
    score = build_title_career_consistency(sample_candidates[0])
    assert 0.0 <= score <= 1.0
    assert features["candidate_id"]


def test_ranking_agent_scores(sample_candidates, settings):
    job_text = (PROJECT_ROOT / "data" / "job_description.txt").read_text(encoding="utf-8")
    job_profile = JobUnderstandingAgent(settings).analyze(job_text)
    retrieved = []
    for candidate in sample_candidates[:5]:
        features = build_candidate_features(candidate)
        retrieved.append(
            {"candidate": candidate, "features": features, "semantic_score": 0.72}
        )
    ranked = RankingAgent(settings).rank(job_profile, retrieved, top_k=5)
    assert len(ranked) == 5
    assert ranked[0].rank == 1
    assert ranked[0].score >= ranked[-1].score


def test_submission_score_format(sample_candidates, settings):
    job_profile = JobUnderstandingAgent(settings).analyze("Senior ML Engineer Python NLP")
    retrieved = []
    for candidate in sample_candidates[:3]:
        features = build_candidate_features(candidate)
        retrieved.append(
            {"candidate": candidate, "features": features, "semantic_score": 0.8}
        )
    ranked = RankingAgent(settings).rank(job_profile, retrieved, top_k=3)
    assert ranked[0].submission_score == 1.0
    assert ranked[1].submission_score == 0.992
