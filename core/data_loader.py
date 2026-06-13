"""Candidate dataset loading and validation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

from core.logging_setup import setup_logging

logger = setup_logging()

CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")


def load_job_description(path: Path) -> str:
    """Load job description text from file."""
    if not path.exists():
        raise FileNotFoundError(f"Job description not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def validate_candidate_record(record: dict[str, Any]) -> list[str]:
    """Validate a candidate record against expected schema fields."""
    errors: list[str] = []
    cid = record.get("candidate_id", "")
    if not CANDIDATE_ID_PATTERN.match(str(cid)):
        errors.append(f"Invalid candidate_id: {cid}")

    required_roots = [
        "profile",
        "career_history",
        "education",
        "skills",
        "redrob_signals",
    ]
    for key in required_roots:
        if key not in record:
            errors.append(f"Missing required field: {key}")

    profile = record.get("profile", {})
    for key in [
        "anonymized_name",
        "headline",
        "summary",
        "years_of_experience",
        "current_title",
    ]:
        if key not in profile:
            errors.append(f"Missing profile.{key}")

    return errors


def iter_candidates(path: Path) -> Iterator[dict[str, Any]]:
    """Stream candidates from JSONL file."""
    logger.info("Loading candidates from %s", path)
    if not path.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")

    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON at line %s: %s", line_no, exc)
                continue

            errors = validate_candidate_record(record)
            if errors:
                logger.debug("Candidate validation warnings line %s: %s", line_no, errors)

            count += 1
            yield record

    logger.info("Loaded %s candidate records", count)


def load_candidates(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load all candidates into memory (optionally limited)."""
    records: list[dict[str, Any]] = []
    for idx, record in enumerate(iter_candidates(path)):
        records.append(record)
        if limit is not None and idx + 1 >= limit:
            break
    return records


def load_candidates_index(path: Path) -> dict[str, dict[str, Any]]:
    """Load candidates keyed by candidate_id."""
    return {c["candidate_id"]: c for c in iter_candidates(path)}


def load_candidates_by_ids(path: Path, candidate_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Load only the requested candidate records (memory-efficient)."""
    if not candidate_ids:
        return {}
    wanted = set(candidate_ids)
    found: dict[str, dict[str, Any]] = {}
    for record in iter_candidates(path):
        cid = record.get("candidate_id", "")
        if cid in wanted:
            found[cid] = record
            if len(found) == len(wanted):
                break
    return found


def analyze_dataset(path: Path, sample_size: int = 5000) -> dict[str, Any]:
    """Compute dataset statistics for EDA and preprocessing."""
    from collections import Counter

    titles: Counter[str] = Counter()
    missing_certifications = 0
    missing_assessments = 0
    skill_counts: list[int] = []
    total = 0

    for record in iter_candidates(path):
        total += 1
        profile = record.get("profile", {})
        titles[profile.get("current_title", "unknown")] += 1
        skill_counts.append(len(record.get("skills", [])))
        if not record.get("certifications"):
            missing_certifications += 1
        signals = record.get("redrob_signals", {})
        if not signals.get("skill_assessment_scores"):
            missing_assessments += 1
        if total >= sample_size:
            break

    return {
        "sampled_records": total,
        "top_titles": titles.most_common(15),
        "avg_skill_count": sum(skill_counts) / max(len(skill_counts), 1),
        "missing_certifications_pct": missing_certifications / max(total, 1) * 100,
        "missing_assessments_pct": missing_assessments / max(total, 1) * 100,
    }
