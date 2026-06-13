"""Preprocessing pipeline for job and candidate data."""

from __future__ import annotations

import re
from typing import Any

from preprocessing.feature_engineering import AI_CORE_SKILLS, normalize_skill


BEHAVIORAL_KEYWORDS = {
    "ownership": ["ownership", "owned", "accountability", "end-to-end"],
    "communication": ["communication", "stakeholder", "present", "document", "explain"],
    "collaboration": ["collaborate", "cross-functional", "team", "mentor"],
    "self learner": ["self-directed", "self learner", "curious", "learning", "kaggle", "side project"],
    "leadership": ["led", "lead", "mentor", "managed team", "drove"],
}


def preprocess_job_text(job_text: str) -> dict[str, Any]:
    """Rule-based job parsing used in offline mode."""
    text = job_text.strip()
    lower = text.lower()

    required_skills = _extract_section_items(text, ["required skills", "required:"])
    preferred_skills = _extract_section_items(text, ["preferred skills", "preferred:"])
    if not required_skills:
        required_skills = sorted(
            skill for skill in AI_CORE_SKILLS if skill in lower
        )[:12]

    exp_min, exp_max = _extract_experience_range(lower)
    domain = _extract_after_label(text, "domain knowledge") or "Machine Learning / HR Tech"
    behavior = _extract_behavior_traits(lower)
    leadership = _extract_section_items(text, ["leadership signals"])
    communication = _extract_section_items(text, ["communication requirements"])

    return {
        "skills": required_skills,
        "preferred_skills": preferred_skills,
        "experience": f"{exp_min}-{exp_max} years",
        "experience_min_years": exp_min,
        "experience_max_years": exp_max,
        "behavior": behavior,
        "domain": domain,
        "leadership_signals": leadership,
        "communication_requirements": communication,
        "raw_text": text,
    }


def preprocess_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach preprocessed documents and basic cleanup flags."""
    processed: list[dict[str, Any]] = []
    for candidate in candidates:
        from preprocessing.feature_engineering import build_candidate_document, build_candidate_features

        features = build_candidate_features(candidate)
        processed.append(
            {
                "candidate": candidate,
                "features": features,
                "document": build_candidate_document(candidate),
            }
        )
    return processed


def _extract_section_items(text: str, headers: list[str]) -> list[str]:
    lines = text.splitlines()
    items: list[str] = []
    capture = False
    section_headers = {
        "responsibilities",
        "preferred skills",
        "experience requirements",
        "behavioral traits",
        "domain knowledge",
        "leadership signals",
        "communication requirements",
        "about the role",
    }
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower().rstrip(":")
        if any(h in lower for h in headers):
            capture = True
            continue
        if capture:
            if stripped.startswith("- "):
                item = stripped[2:].strip()
                if item:
                    for part in re.split(r",|/| and ", item):
                        part = part.strip(" .")
                        if part and len(part) > 2:
                            items.append(normalize_skill(part))
            elif lower in section_headers or (stripped.endswith(":") and lower.replace(" ", "") != ""):
                if items:
                    break
            elif not stripped and items:
                break
    return items


def _extract_experience_range(lower_text: str) -> tuple[float, float]:
    match = re.search(r"experience:\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)", lower_text)
    if match:
        return float(match.group(1)), float(match.group(2))
    match = re.search(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*years", lower_text)
    if match:
        return float(match.group(1)), float(match.group(2))
    return 3.0, 8.0


def _extract_after_label(text: str, label: str) -> str:
    pattern = re.compile(rf"{re.escape(label)}\s*\n(.+)", re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return match.group(1).strip().splitlines()[0].strip("- ").strip()
    return ""


def _extract_behavior_traits(lower_text: str) -> list[str]:
    found = []
    for trait, keywords in BEHAVIORAL_KEYWORDS.items():
        if any(k in lower_text for k in keywords):
            found.append(trait.title())
    if not found:
        found = ["Ownership", "Communication", "Self Learner"]
    return found
