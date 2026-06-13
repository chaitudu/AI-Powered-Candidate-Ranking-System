"""Data models for structured agent outputs and API payloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class JobProfile:
    skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    experience: str = ""
    experience_min_years: float = 0.0
    experience_max_years: float = 15.0
    behavior: list[str] = field(default_factory=list)
    domain: str = ""
    leadership_signals: list[str] = field(default_factory=list)
    communication_requirements: list[str] = field(default_factory=list)
    raw_text: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobProfile":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateProfileStructured:
    candidate_id: str
    name: str = ""
    skills: list[str] = field(default_factory=list)
    experience_years: float = 0.0
    projects: list[str] = field(default_factory=list)
    behavior: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    domain: str = ""
    narrative: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ComponentScores:
    skill_match: float = 0.0
    project_relevance: float = 0.0
    experience_match: float = 0.0
    behavior_match: float = 0.0
    learning_potential: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RankedCandidate:
    candidate_id: str
    name: str = ""
    score: float = 0.0
    submission_score: float = 0.0
    rank: int = 0
    components: ComponentScores = field(default_factory=ComponentScores)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    recommendation: str = ""
    reasoning: str = ""
    why_selected: str = ""
    why_ranked_higher: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["components"] = self.components.to_dict()
        return data


@dataclass
class RankingResponse:
    job_profile: JobProfile
    total_candidates: int
    retrieved_count: int
    ranked: list[RankedCandidate]

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_profile": self.job_profile.to_dict(),
            "total_candidates": self.total_candidates,
            "retrieved_count": self.retrieved_count,
            "ranked": [r.to_dict() for r in self.ranked],
        }

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
