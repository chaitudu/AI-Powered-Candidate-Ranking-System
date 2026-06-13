"""Job understanding agent."""

from __future__ import annotations

import json
import re
from typing import Any

from core.config import Settings
from core.logging_setup import setup_logging
from core.schemas import JobProfile
from preprocessing.pipeline import preprocess_job_text

logger = setup_logging()


class JobUnderstandingAgent:
    """Extract structured job requirements from raw job description text."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()

    def analyze(self, job_text: str) -> JobProfile:
        """Return structured job profile using LLM or offline parser."""
        if self.settings.use_llm_agents and self.settings.gemini_api_key:
            try:
                return self._analyze_with_gemini(job_text)
            except Exception as exc:
                logger.warning("Gemini job analysis failed, falling back to rules: %s", exc)
        return self._analyze_with_rules(job_text)

    def _analyze_with_rules(self, job_text: str) -> JobProfile:
        parsed = preprocess_job_text(job_text)
        return JobProfile(**parsed)

    def _analyze_with_gemini(self, job_text: str) -> JobProfile:
        import google.generativeai as genai

        genai.configure(api_key=self.settings.gemini_api_key)
        model = genai.GenerativeModel(self.settings.gemini_model)
        prompt = f"""
You are a senior technical recruiter. Read the job description and return ONLY valid JSON with keys:
skills, preferred_skills, experience, experience_min_years, experience_max_years, behavior, domain,
leadership_signals, communication_requirements.

Job Description:
{job_text}
"""
        response = model.generate_content(prompt)
        text = response.text or "{}"
        match = re.search(r"\{.*\}", text, re.DOTALL)
        payload = json.loads(match.group(0) if match else text)
        payload["raw_text"] = job_text
        return JobProfile(**payload)
