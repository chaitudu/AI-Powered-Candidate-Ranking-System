# Technical Report: AI-Powered Candidate Ranking System

**Redrob India Runs — Data & AI Challenge**

---

## Executive Summary

We built an end-to-end candidate ranking system that processes 100,000 JSONL profiles and produces an explainable top-100 shortlist for a Senior Machine Learning Engineer role. The pipeline combines LangGraph agent orchestration, dense semantic embeddings, ChromaDB vector storage, and a 100-point multi-signal scoring model with explicit honeypot detection for keyword-stuffed profiles.

---

## 1. Introduction

Traditional applicant tracking systems rely on Boolean keyword filters. In the provided dataset, many candidates list AI/ML skills unrelated to their career history — a deliberate trap for naive rankers. Our system evaluates holistic profile meaning using semantic vectors and trust-weighted skill verification.

---

## 2. Data Exploration

### 2.1 Schema

Each record contains:

- `candidate_id` (CAND_XXXXXXX)
- `profile` (headline, summary, experience, current role)
- `career_history` (up to 10 roles with descriptions)
- `education`, `skills`, optional `certifications`, `languages`
- `redrob_signals` (platform engagement, assessments, GitHub, salary, etc.)

### 2.2 Statistics

| Metric | Finding |
|--------|---------|
| Total records | 100,000 |
| Title distribution | ~5,700 each for generic roles; ~1,000 for AI/ML titles |
| Skill distribution | Nearly uniform (~12K each) — skill lists alone are unreliable |
| Missing certifications | Common (empty arrays) |
| Missing skill assessments | Majority of profiles |
| Honeypot pattern | 430+ profiles (sample analysis) with 5+ AI skills but no AI career text |

### 2.3 ER Diagram

See `docs/er_diagram.md` for full entity relationships and field mapping.

---

## 3. Methodology

### 3.1 Preprocessing Pipeline

1. Validate JSON schema fields
2. Build rich candidate document from all text fields
3. Engineer features: skill trust, title-career consistency, AI core skill count
4. Parse job description into structured requirements

### 3.2 Semantic Matching Strategy

- Encode job description and candidate documents with sentence-transformers
- Precompute 100K vectors for offline submission runtime
- Retrieve top 500 by cosine similarity
- Re-rank with interpretable weighted scoring

### 3.3 Ranking Formula

```
Final = (Skill×30 + Project×20 + Experience×20 + Behavior×15 + Learning×15) × TrustMultiplier

TrustMultiplier = 0.55 + 0.25×skill_trust + 0.20×title_career_consistency
```

### 3.4 Explainability

Each ranked candidate includes strengths, weaknesses, missing skills, natural-language reasoning, and hiring recommendation tier.

---

## 4. System Implementation

| Layer | Technology |
|-------|------------|
| Agents | LangGraph workflow |
| LLM (optional) | Gemini 2.5 Pro / Flash |
| Embeddings | all-MiniLM-L6-v2 |
| Vector DB | ChromaDB + numpy |
| API | FastAPI |
| UI | Streamlit |
| Tests | pytest |

### 4.1 Agent Workflow

1. Job Understanding Agent
2. Semantic Retrieval Agent
3. Candidate Understanding Agent
4. Ranking Agent
5. Export + evaluation

### 4.2 API Endpoints

- `POST /rank` — run pipeline, return JSON
- `POST /rank/export` — generate CSV outputs
- `GET /dataset/stats` — EDA summary

---

## 5. Evaluation

Metrics implemented in `evaluation/metrics.py`:

- Score mean/std/min/max
- Per-component averages
- Honeypot penalty rate in top ranks
- Recommendation distribution

Submission validation via provided `validate_submission.py`:

- Exactly 100 rows
- Monotonic non-increasing scores
- Unique ranks and candidate IDs

---

## 6. Results

After running `rank.py` on the full corpus:

- Top candidates predominantly include ML Engineer, Data Scientist, AI Engineer titles
- Keyword-stuffed non-technical profiles penalized by trust multiplier
- Output files:
  - `outputs/submission.csv`
  - `outputs/ranked_candidates.csv`
  - `outputs/ranking_report.json`

---

## 7. Limitations & Future Work

- LLM agents optional; offline mode uses rule-based JD parsing
- Embedding model smaller than text-embedding-3-large for CPU constraint
- Future: cross-encoder re-ranker, calibrated score-to-hire probability, active learning from recruiter feedback

---

## 8. Reproducibility

```bash
pip install -r requirements.txt
python precompute_embeddings.py --candidates path/to/candidates.jsonl
python rank.py --candidates path/to/candidates.jsonl --out submission.csv
python validate_submission.py submission.csv
pytest tests/ -q
```

Environment: Python 3.9+, 16GB RAM, CPU-only, no network during ranking.

---

## 9. Conclusion

We delivered a production-ready, modular ranking platform that understands job requirements and candidate narratives semantically, detects keyword stuffing, and explains every ranking decision — enabling recruiters to hire smarter at scale.

---

*End of Report*
