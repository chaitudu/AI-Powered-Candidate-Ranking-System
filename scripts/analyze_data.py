#!/usr/bin/env python3
"""Dataset exploration script for EDA and schema inspection."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import get_settings
from core.data_loader import analyze_dataset, iter_candidates, validate_candidate_record


def main() -> int:
    settings = get_settings()
    path = settings.candidates_path

    print("=" * 60)
    print("AI Candidate Ranking — Dataset Analysis")
    print("=" * 60)
    print(f"Path: {path}")

    stats = analyze_dataset(path, sample_size=10000)
    print("\nSample stats (10K records):")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    titles: Counter[str] = Counter()
    ai_titles = 0
    honeypot = 0
    total = 0

    from preprocessing.feature_engineering import build_title_career_consistency

    for record in iter_candidates(path):
        total += 1
        title = record.get("profile", {}).get("current_title", "unknown")
        titles[title] += 1
        if any(k in title.lower() for k in ["ml", "machine learning", "ai", "data scien"]):
            ai_titles += 1
        if build_title_career_consistency(record) < 0.35:
            honeypot += 1
        if total >= 50000:
            break

    print(f"\nScanned records: {total}")
    print(f"AI-related titles: {ai_titles}")
    print(f"Low consistency profiles (potential honeypots): {honeypot}")
    print(f"Top titles: {titles.most_common(10)}")

    out = settings.outputs_dir / "dataset_analysis.json"
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "sample_stats": stats,
                "scanned_records": total,
                "ai_related_titles": ai_titles,
                "low_consistency_profiles": honeypot,
                "top_titles": titles.most_common(20),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
