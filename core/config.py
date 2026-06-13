"""Application configuration and paths."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CANDIDATES_PATH = Path(
    os.environ.get(
        "CANDIDATES_PATH",
        r"c:\Users\chaitu chey\Downloads\[PUB] India_runs_data_and_ai_challenge"
        r"\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl",
    )
)


@dataclass
class Settings:
    """Runtime settings for ranking pipeline."""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    outputs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")
    vector_store_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "vector_store")
    chroma_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "vector_store" / "chroma")
    embeddings_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "vector_store" / "embeddings")

    candidates_path: Path = DEFAULT_CANDIDATES_PATH
    job_description_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "job_description.txt"
    )

    embedding_model: str = os.environ.get(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    gemini_model: str = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    gemini_api_key: str = os.environ.get("GOOGLE_API_KEY", "")

    retrieval_top_k: int = 50
    final_top_k: int = 100
    batch_size: int = 256
    offline_mode: bool = os.environ.get("OFFLINE_MODE", "true").lower() == "true"
    use_llm_agents: bool = os.environ.get("USE_LLM_AGENTS", "false").lower() == "true"

    chroma_collection: str = "candidate_profiles"

    score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "skill_match": 30.0,
            "project_relevance": 20.0,
            "experience_match": 20.0,
            "behavior_match": 15.0,
            "learning_potential": 15.0,
        }
    )


def get_settings() -> Settings:
    """Return singleton-like settings instance."""
    return Settings()
