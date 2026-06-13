"""Evaluation metrics for ranking quality."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.schemas import RankedCandidate


def score_distribution(ranked: list[RankedCandidate]) -> dict[str, float]:
    scores = [r.score for r in ranked]
    if not scores:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(scores, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def component_averages(ranked: list[RankedCandidate]) -> dict[str, float]:
    if not ranked:
        return {}
    keys = [
        "skill_match",
        "project_relevance",
        "experience_match",
        "behavior_match",
        "learning_potential",
    ]
    totals = {k: 0.0 for k in keys}
    for item in ranked:
        for key in keys:
            totals[key] += getattr(item.components, key)
    return {k: round(v / len(ranked), 2) for k, v in totals.items()}


def honeypot_penalty_rate(ranked: list[RankedCandidate], threshold: float = 0.5) -> float:
    """Estimate rate of low-consistency profiles in top ranks (should be low)."""
    flagged = 0
    for item in ranked:
        if any("inconsistency" in w.lower() or "keyword stuffing" in w.lower() for w in item.weaknesses):
            flagged += 1
    return flagged / max(len(ranked), 1)


def diversity_by_title(ranked: list[RankedCandidate], candidate_map: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in ranked:
        title = candidate_map.get(item.candidate_id, {}).get("profile", {}).get("current_title", "Unknown")
        counts[title] = counts.get(title, 0) + 1
    return counts


def build_evaluation_report(ranked: list[RankedCandidate]) -> dict[str, Any]:
    return {
        "ranked_count": len(ranked),
        "score_distribution": score_distribution(ranked),
        "component_averages": component_averages(ranked),
        "honeypot_penalty_rate": honeypot_penalty_rate(ranked),
        "recommendation_mix": {
            rec: sum(1 for r in ranked if r.recommendation == rec)
            for rec in sorted(set(r.recommendation for r in ranked))
        },
    }
