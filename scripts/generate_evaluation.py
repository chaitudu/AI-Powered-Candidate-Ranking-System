#!/usr/bin/env python3
"""Generate evaluation metrics from ranking outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import get_settings
from core.data_loader import load_candidates_by_ids
from evaluation.metrics import build_evaluation_report, diversity_by_title
from pipeline import CandidateRankingPipeline


def load_ranked_from_csv(path: Path):
    import csv

    from core.schemas import ComponentScores, RankedCandidate

    ranked = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ranked.append(
                RankedCandidate(
                    candidate_id=row["Candidate_ID"],
                    name=row.get("Name", ""),
                    score=float(row["Score"]),
                    rank=int(row["Rank"]),
                    components=ComponentScores(
                        skill_match=float(row.get("Skill_Match", 0)),
                        experience_match=float(row.get("Experience_Match", 0)),
                        behavior_match=float(row.get("Behavior_Match", 0)),
                        project_relevance=float(row.get("Project_Relevance", 0)),
                        learning_potential=float(row.get("Learning_Potential", 0)),
                    ),
                    recommendation=row.get("Recommendation", ""),
                    reasoning=row.get("Reason", ""),
                    weaknesses=[],
                )
            )
    return ranked


def main() -> int:
    settings = get_settings()
    ranked_path = settings.outputs_dir / "ranked_candidates.csv"
    if not ranked_path.exists():
        print("Run rank.py first to generate outputs.")
        return 1

    ranked = load_ranked_from_csv(ranked_path)
    ids = {r.candidate_id for r in ranked}
    candidate_map = load_candidates_by_ids(settings.candidates_path, ids)

    report = build_evaluation_report(ranked)
    report["title_diversity_top100"] = diversity_by_title(ranked, candidate_map)

    out = settings.outputs_dir / "evaluation_metrics.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
