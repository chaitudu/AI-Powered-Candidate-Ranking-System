"""Candidate understanding agent."""

from __future__ import annotations

import json
import re
from typing import Any

from core.config import Settings
from core.logging_setup import setup_logging
from core.schemas import CandidateProfileStructured
from preprocessing.feature_engineering import build_candidate_features, extract_projects_from_career

logger = setup_logging()


class CandidateUnderstandingAgent:
    """Build structured candidate profiles from raw JSON records."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()

    def analyze(self, candidate: dict[str, Any]) -> CandidateProfileStructured:
        """Return structured candidate profile."""
        if self.settings.use_llm_agents and self.settings.gemini_api_key:
            try:
                return self._analyze_with_gemini(candidate)
            except Exception as exc:
                logger.warning(
                    "Gemini candidate analysis failed for %s: %s",
                    candidate.get("candidate_id"),
                    exc,
                )
        return self._analyze_with_rules(candidate)

    def analyze_batch(self, candidates: list[dict[str, Any]]) -> list[CandidateProfileStructured]:
        return [self.analyze(c) for c in candidates]

    def _analyze_with_rules(self, candidate: dict[str, Any]) -> CandidateProfileStructured:
        features = build_candidate_features(candidate)
        profile = candidate.get("profile", {})
        signals = candidate.get("redrob_signals", {})

        behavior = []
        summary = profile.get("summary", "").lower()
        if "self" in summary and "learn" in summary:
            behavior.append("Self Learner")
        if signals.get("recruiter_response_rate", 0) >= 0.5:
            behavior.append("Responsive")
        if signals.get("github_activity_score", -1) > 20:
            behavior.append("Consistent Contributor")
        if not behavior:
            behavior = ["Professional"]

        certs = [
            f"{c.get('name', '')} ({c.get('issuer', '')})"
            for c in candidate.get("certifications", [])
        ]

        return CandidateProfileStructured(
            candidate_id=candidate.get("candidate_id", ""),
            name=profile.get("anonymized_name", ""),
            skills=features["skills"],
            experience_years=features["years_of_experience"],
            projects=extract_projects_from_career(candidate.get("career_history", [])),
            behavior=behavior,
            certifications=certs,
            domain=profile.get("current_industry", ""),
            narrative=features["document"][:1200],
            metadata={
                "skill_trust": features["skill_trust"],
                "title_career_consistency": features["title_career_consistency"],
                "ai_core_skill_count": features["ai_core_skill_count"],
            },
        )

    def _analyze_with_gemini(self, candidate: dict[str, Any]) -> CandidateProfileStructured:
        import google.generativeai as genai

        genai.configure(api_key=self.settings.gemini_api_key)
        model = genai.GenerativeModel(self.settings.gemini_model)
        prompt = f"""
Analyze this candidate profile and return ONLY JSON with keys:
candidate_id, name, skills, experience_years, projects, behavior, certifications, domain.

Candidate:
{json.dumps(candidate, ensure_ascii=False)[:12000]}
"""
        response = model.generate_content(prompt)
        text = response.text or "{}"
        match = re.search(r"\{.*\}", text, re.DOTALL)
        payload = json.loads(match.group(0) if match else text)
        payload.setdefault("candidate_id", candidate.get("candidate_id", ""))
        payload.setdefault("name", candidate.get("profile", {}).get("anonymized_name", ""))
        payload.setdefault("narrative", build_candidate_features(candidate)["document"][:1200])
        payload.setdefault("metadata", {})
        return CandidateProfileStructured(**payload)
