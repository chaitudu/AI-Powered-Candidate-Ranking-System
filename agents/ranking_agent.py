"""AI ranking agent with explainable multi-signal scoring."""

from __future__ import annotations

from typing import Any

from core.config import Settings
from core.logging_setup import setup_logging
from core.schemas import ComponentScores, JobProfile, RankedCandidate
from preprocessing.feature_engineering import (
    AI_CORE_SKILLS,
    AI_TITLE_KEYWORDS,
    TECHNICAL_CAREER_KEYWORDS,
    normalize_skill,
)

logger = setup_logging()


class RankingAgent:
    """
    Score candidates on five components (100 points total):
    - Skill Match (30)
    - Project Relevance (20)
    - Experience Match (20)
    - Behavioral Match (15)
    - Learning Potential (15)
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.weights = self.settings.score_weights

    def rank(
        self,
        job_profile: JobProfile,
        retrieved: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[RankedCandidate]:
        top_k = top_k or self.settings.final_top_k
        required = {normalize_skill(s) for s in job_profile.skills}
        preferred = {normalize_skill(s) for s in job_profile.preferred_skills}

        scored: list[RankedCandidate] = []
        for item in retrieved:
            features = item["features"]
            semantic = float(item.get("semantic_score", 0.0))
            components = self._score_components(
                job_profile, features, required, preferred, semantic
            )
            total = (
                components.skill_match
                + components.project_relevance
                + components.experience_match
                + components.behavior_match
                + components.learning_potential
            )

            trust = float(features.get("skill_trust", 0.5))
            consistency = float(features.get("title_career_consistency", 0.5))
            title_relevance = self._title_relevance(features, job_profile)
            multiplier = (0.55 + 0.25 * trust + 0.20 * consistency) * title_relevance
            total = min(100.0, total * multiplier)

            strengths, weaknesses, missing = self._explain(
                job_profile, features, components, required, preferred
            )
            reasoning = self._build_reasoning(features, components, total, missing)

            scored.append(
                RankedCandidate(
                    candidate_id=features["candidate_id"],
                    name=features.get("name", ""),
                    score=round(total, 2),
                    components=components,
                    strengths=strengths,
                    weaknesses=weaknesses,
                    missing_skills=missing,
                    recommendation=self._recommendation(total),
                    reasoning=reasoning,
                    why_selected=(
                        f"Strong semantic fit ({semantic:.2f}) with verified career signals "
                        f"and {features.get('ai_core_skill_count', 0)} core ML/AI skills."
                    ),
                    why_ranked_higher=(
                        f"Composite score driven by skill match ({components.skill_match:.1f}/30) "
                        f"and project relevance ({components.project_relevance:.1f}/20)."
                    ),
                )
            )

        scored.sort(key=lambda x: (-x.score, x.candidate_id))
        for idx, item in enumerate(scored[:top_k], start=1):
            item.rank = idx
            item.submission_score = round(1.0 - (idx - 1) * 0.008, 4)
        return scored[:top_k]

    def _score_components(
        self,
        job_profile: JobProfile,
        features: dict[str, Any],
        required: set[str],
        preferred: set[str],
        semantic: float,
    ) -> ComponentScores:
        candidate_skills = set(features.get("skills", []))
        ai_core = set(features.get("ai_core_skills", []))

        required_hits = self._skill_overlap(candidate_skills, required)
        preferred_hits = self._skill_overlap(candidate_skills, preferred)
        ai_hits = len(ai_core)

        skill_ratio = 0.0
        if required:
            skill_ratio = required_hits / max(len(required), 1)
        else:
            skill_ratio = min(1.0, ai_hits / 6.0)

        skill_ratio += min(0.2, preferred_hits / max(len(preferred) or 1, 1) * 0.2)
        skill_ratio = min(1.0, skill_ratio)
        skill_match = self.weights["skill_match"] * skill_ratio * float(features.get("skill_trust", 0.7))

        project_relevance = self.weights["project_relevance"] * min(
            1.0, 0.45 + semantic * 0.55
        ) * float(features.get("title_career_consistency", 0.6))

        years = float(features.get("years_of_experience", 0))
        exp_min = job_profile.experience_min_years or 0
        exp_max = job_profile.experience_max_years or 15
        if exp_min <= years <= exp_max:
            exp_score = 1.0
        elif years < exp_min:
            exp_score = max(0.2, 1.0 - (exp_min - years) / max(exp_min, 1))
        else:
            exp_score = max(0.35, 1.0 - (years - exp_max) / 10.0)

        title = features.get("current_title", "").lower()
        if any(k in title for k in ["ml", "machine learning", "ai", "data scien", "nlp"]):
            exp_score = min(1.0, exp_score + 0.15)
        elif not TECHNICAL_CAREER_KEYWORDS.search(
            " ".join(features.get("projects", []))
        ):
            exp_score *= 0.35
        experience_match = self.weights["experience_match"] * exp_score

        behavior_match = self.weights["behavior_match"] * self._behavior_score(features)
        learning_potential = self.weights["learning_potential"] * self._learning_score(features)

        return ComponentScores(
            skill_match=round(skill_match, 2),
            project_relevance=round(project_relevance, 2),
            experience_match=round(experience_match, 2),
            behavior_match=round(behavior_match, 2),
            learning_potential=round(learning_potential, 2),
        )

    def _skill_overlap(self, candidate_skills: set[str], target: set[str]) -> int:
        if not target:
            return 0
        hits = 0
        for skill in target:
            if skill in candidate_skills:
                hits += 1
                continue
            for cs in candidate_skills:
                if skill in cs or cs in skill:
                    hits += 1
                    break
        return hits

    def _title_relevance(self, features: dict[str, Any], job_profile: JobProfile) -> float:
        """Penalize profiles whose title/career do not match an ML engineering role."""
        title = features.get("current_title", "")
        raw = features.get("raw") or {}
        career_text = " ".join(
            f"{r.get('title', '')} {r.get('description', '')}"
            for r in raw.get("career_history", [])
        )
        summary = raw.get("profile", {}).get("summary", "")

        title_match = bool(AI_TITLE_KEYWORDS.search(title))
        career_match = bool(TECHNICAL_CAREER_KEYWORDS.search(career_text))
        summary_match = bool(TECHNICAL_CAREER_KEYWORDS.search(summary))
        ai_skills = int(features.get("ai_core_skill_count", 0))

        if title_match and career_match:
            return 1.0
        if title_match or (career_match and ai_skills >= 3):
            return 0.92
        if career_match or summary_match:
            return 0.75
        if ai_skills >= 5 and float(features.get("title_career_consistency", 0)) < 0.45:
            return 0.25
        if ai_skills >= 3:
            return 0.45
        return 0.2

    def _behavior_score(self, features: dict[str, Any]) -> float:
        score = 0.35
        if features.get("open_to_work"):
            score += 0.1
        response = float(features.get("recruiter_response_rate", 0))
        score += min(0.25, response * 0.25)
        interview = float(features.get("interview_completion_rate", 0))
        score += min(0.15, interview * 0.15)
        if int(features.get("saved_by_recruiters_30d", 0)) >= 3:
            score += 0.1
        score *= float(features.get("title_career_consistency", 0.6))
        return min(1.0, score)

    def _learning_score(self, features: dict[str, Any]) -> float:
        score = 0.25
        github = float(features.get("github_activity", -1))
        if github >= 0:
            score += min(0.35, github / 100.0)
        completeness = float(features.get("profile_completeness", 0))
        score += min(0.2, completeness / 100.0 * 0.2)
        if features.get("ai_core_skill_count", 0) >= 2:
            score += 0.15
        score *= float(features.get("skill_trust", 0.6))
        return min(1.0, score)

    def _explain(
        self,
        job_profile: JobProfile,
        features: dict[str, Any],
        components: ComponentScores,
        required: set[str],
        preferred: set[str],
    ) -> tuple[list[str], list[str], list[str]]:
        strengths: list[str] = []
        weaknesses: list[str] = []
        candidate_skills = set(features.get("skills", []))

        if components.skill_match >= 20:
            strengths.append("Strong required skill alignment with credible proficiency signals")
        if components.project_relevance >= 14:
            strengths.append("Career projects semantically aligned with ML engineering work")
        if components.experience_match >= 15:
            strengths.append("Experience band matches role expectations")
        if features.get("ai_core_skill_count", 0) >= 4:
            strengths.append(f"{features['ai_core_skill_count']} verified AI/ML core skills")
        if float(features.get("recruiter_response_rate", 0)) >= 0.5:
            strengths.append("High recruiter response rate indicates engagement")

        missing = []
        for skill in sorted(required):
            if skill not in candidate_skills and not any(skill in cs or cs in skill for cs in candidate_skills):
                missing.append(skill)

        if components.skill_match < 15:
            weaknesses.append("Limited overlap with required skills")
        if float(features.get("title_career_consistency", 1)) < 0.5:
            weaknesses.append("Title/career narrative inconsistency suggests keyword stuffing risk")
        if float(features.get("skill_trust", 1)) < 0.55:
            weaknesses.append("Skill claims not supported by assessments or tenure")
        if missing:
            weaknesses.append(f"Missing key skills: {', '.join(missing[:5])}")

        return strengths, weaknesses, missing

    def _build_reasoning(
        self,
        features: dict[str, Any],
        components: ComponentScores,
        total: float,
        missing: list[str],
    ) -> str:
        title = features.get("current_title", "Candidate")
        years = features.get("years_of_experience", 0)
        ai_count = features.get("ai_core_skill_count", 0)
        response = features.get("recruiter_response_rate", 0)
        base = (
            f"{title} with {years:.1f} yrs; {ai_count} AI core skills; "
            f"response rate {response:.2f}; score {total:.1f}/100."
        )
        if missing:
            base += f" Gaps: {', '.join(missing[:3])}."
        return base

    def _recommendation(self, score: float) -> str:
        if score >= 85:
            return "Strong Hire"
        if score >= 70:
            return "Interview"
        if score >= 55:
            return "Consider"
        return "Pass"
