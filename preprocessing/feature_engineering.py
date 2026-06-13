"""Feature engineering for semantic matching and ranking."""

from __future__ import annotations

import re
from typing import Any

PROFICIENCY_WEIGHTS = {
    "beginner": 0.35,
    "intermediate": 0.65,
    "advanced": 0.85,
    "expert": 1.0,
}

AI_CORE_SKILLS = {
    "machine learning",
    "deep learning",
    "pytorch",
    "tensorflow",
    "keras",
    "scikit-learn",
    "sklearn",
    "nlp",
    "natural language processing",
    "computer vision",
    "transformers",
    "llm",
    "llms",
    "fine-tuning llms",
    "fine tuning llms",
    "rag",
    "retrieval augmented generation",
    "mlops",
    "feature engineering",
    "statistical modeling",
    "xgboost",
    "object detection",
    "image classification",
    "speech recognition",
    "reinforcement learning",
    "neural networks",
    "data science",
    "pandas",
    "numpy",
    "pytorch lightning",
    "hugging face",
    "langchain",
    "langgraph",
    "vector database",
    "chromadb",
    "faiss",
    "milvus",
    "model deployment",
    "bentoml",
    "weights & biases",
    "mlflow",
}

AI_TITLE_KEYWORDS = re.compile(
    r"(machine learning|ml engineer|ai engineer|data scien|deep learning|nlp|applied ml|ai research|ai specialist)",
    re.IGNORECASE,
)

TECHNICAL_CAREER_KEYWORDS = re.compile(
    r"(model|pipeline|ml|machine learning|deep learning|nlp|llm|pytorch|tensorflow|embedding|vector|rag|training|inference|feature|data science|algorithm|neural|classification|regression|forecast|recommendation|search ranking)",
    re.IGNORECASE,
)


def normalize_skill(skill: str) -> str:
    """Normalize skill names for matching."""
    return re.sub(r"\s+", " ", skill.strip().lower())


def extract_projects_from_career(career_history: list[dict[str, Any]]) -> list[str]:
    """Extract project-like signals from career descriptions."""
    projects: list[str] = []
    for role in career_history:
        title = role.get("title", "")
        description = role.get("description", "")
        snippet = f"{title}: {description[:180]}"
        if description.strip():
            projects.append(snippet.strip())
    return projects


def build_candidate_document(candidate: dict[str, Any]) -> str:
    """Build rich text document for embedding from all candidate signals."""
    profile = candidate.get("profile", {})
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        profile.get("current_industry", ""),
        f"Experience years: {profile.get('years_of_experience', 0)}",
    ]

    for role in candidate.get("career_history", []):
        parts.append(
            f"Role {role.get('title', '')} at {role.get('company', '')}: {role.get('description', '')}"
        )

    skill_lines = []
    for skill in candidate.get("skills", []):
        skill_lines.append(
            f"{skill.get('name', '')} ({skill.get('proficiency', '')}, "
            f"{skill.get('duration_months', 0)} months, "
            f"{skill.get('endorsements', 0)} endorsements)"
        )
    parts.append("Skills: " + "; ".join(skill_lines))

    for cert in candidate.get("certifications", []):
        parts.append(f"Certification: {cert.get('name', '')} by {cert.get('issuer', '')}")

    for edu in candidate.get("education", []):
        parts.append(
            f"Education: {edu.get('degree', '')} in {edu.get('field_of_study', '')} "
            f"from {edu.get('institution', '')}"
        )

    signals = candidate.get("redrob_signals", {})
    parts.append(
        "Platform signals: "
        f"profile_completeness={signals.get('profile_completeness_score', 0)}, "
        f"github_activity={signals.get('github_activity_score', -1)}, "
        f"recruiter_response_rate={signals.get('recruiter_response_rate', 0)}, "
        f"open_to_work={signals.get('open_to_work_flag', False)}"
    )

    return "\n".join(part for part in parts if part)


def build_skill_trust_score(candidate: dict[str, Any]) -> float:
    """
    Detect keyword stuffing: high claimed skill proficiency without assessments,
    endorsements, or duration support.
    """
    skills = candidate.get("skills", [])
    if not skills:
        return 0.5

    signals = candidate.get("redrob_signals", {})
    assessments = {
        normalize_skill(k): float(v)
        for k, v in (signals.get("skill_assessment_scores") or {}).items()
    }

    trust_scores: list[float] = []
    for skill in skills:
        name = normalize_skill(skill.get("name", ""))
        proficiency = skill.get("proficiency", "beginner")
        endorsements = int(skill.get("endorsements", 0))
        duration = int(skill.get("duration_months", 0))

        base = PROFICIENCY_WEIGHTS.get(proficiency, 0.5)
        if name in assessments:
            base = min(1.0, base * 0.5 + assessments[name] / 100.0 * 0.5)
        elif proficiency in {"advanced", "expert"} and endorsements < 3 and duration < 12:
            base *= 0.45
        elif endorsements >= 10 and duration >= 18:
            base = min(1.0, base + 0.1)

        trust_scores.append(base)

    return sum(trust_scores) / len(trust_scores)


def build_title_career_consistency(candidate: dict[str, Any]) -> float:
    """
    Measure alignment between current title, career history, and claimed skills.
    Low score indicates honeypot / keyword-stuffed profiles.
    """
    profile = candidate.get("profile", {})
    current_title = profile.get("current_title", "")
    summary = profile.get("summary", "")
    career_text = " ".join(
        f"{r.get('title', '')} {r.get('description', '')}" for r in candidate.get("career_history", [])
    )
    combined = f"{current_title} {summary} {career_text}"

    title_ai = bool(AI_TITLE_KEYWORDS.search(current_title))
    career_ai = bool(TECHNICAL_CAREER_KEYWORDS.search(career_text))
    summary_ai = bool(TECHNICAL_CAREER_KEYWORDS.search(summary))

    ai_skill_count = sum(
        1 for s in candidate.get("skills", []) if normalize_skill(s.get("name", "")) in AI_CORE_SKILLS
    )

    score = 0.55
    if title_ai and career_ai:
        score += 0.25
    elif career_ai or summary_ai:
        score += 0.15
    elif title_ai and not career_ai:
        score -= 0.2

    if ai_skill_count >= 5 and not (career_ai or title_ai or summary_ai):
        score -= 0.35
    elif ai_skill_count >= 3 and (career_ai or title_ai):
        score += 0.1

    career_titles = [r.get("title", "").lower() for r in candidate.get("career_history", [])]
    if current_title and any(current_title.lower() in t or t in current_title.lower() for t in career_titles):
        score += 0.05

    return max(0.15, min(1.0, score))


def build_candidate_features(candidate: dict[str, Any]) -> dict[str, Any]:
    """Extract engineered features used by ranking heuristics."""
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills = [normalize_skill(s.get("name", "")) for s in candidate.get("skills", []) if s.get("name")]

    ai_core_hits = [s for s in skills if s in AI_CORE_SKILLS or any(k in s for k in AI_CORE_SKILLS)]
    return {
        "candidate_id": candidate.get("candidate_id", ""),
        "name": profile.get("anonymized_name", ""),
        "current_title": profile.get("current_title", ""),
        "years_of_experience": float(profile.get("years_of_experience", 0) or 0),
        "skills": skills,
        "ai_core_skills": ai_core_hits,
        "ai_core_skill_count": len(ai_core_hits),
        "projects": extract_projects_from_career(candidate.get("career_history", [])),
        "skill_trust": build_skill_trust_score(candidate),
        "title_career_consistency": build_title_career_consistency(candidate),
        "profile_completeness": float(signals.get("profile_completeness_score", 0) or 0),
        "github_activity": float(signals.get("github_activity_score", -1) or -1),
        "recruiter_response_rate": float(signals.get("recruiter_response_rate", 0) or 0),
        "open_to_work": bool(signals.get("open_to_work_flag", False)),
        "interview_completion_rate": float(signals.get("interview_completion_rate", 0) or 0),
        "saved_by_recruiters_30d": int(signals.get("saved_by_recruiters_30d", 0) or 0),
        "document": build_candidate_document(candidate),
        "raw": candidate,
    }
