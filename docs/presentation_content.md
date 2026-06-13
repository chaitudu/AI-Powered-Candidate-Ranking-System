# AI Candidate Ranking System — Presentation Deck

> Convert this document to PDF for submission. Each `##` section maps to one slide.

---

## Slide 1: Title

**AI-Powered Candidate Ranking System**

Redrob India Runs — Data & AI Challenge

Semantic hiring intelligence beyond keyword filters

---

## Slide 2: The Problem

- Recruiters review hundreds of profiles; keyword filters miss real fit
- Candidate data is messy: title/skill mismatches, keyword stuffing, sparse assessments
- Hiring needs **meaning**, not string matching

**Goal:** Rank 100K candidates for a Senior ML Engineer role with explainable shortlists

---

## Slide 3: Dataset Understanding

- **100,000** anonymized JSONL candidate profiles
- Rich signals: profile, career history, education, skills, Redrob platform activity
- Schema-validated fields (`candidate_id`, `redrob_signals`, etc.)
- Detected honeypots: profiles with many AI skills but non-AI career narratives

---

## Slide 4: Our Approach

1. Understand the job (skills, experience, behavior, domain)
2. Embed job + full candidate narratives
3. Retrieve top candidates by **vector similarity** (not keywords)
4. Re-rank with multi-signal scoring + anti-stuffing trust model
5. Explain every decision for recruiters

---

## Slide 5: Architecture

```
Job Description → Job Agent → Vector Retrieval → Candidate Agent → Ranking Agent → CSV + UI
                     ↓              ↓
              Gemini (optional)   ChromaDB + Embeddings
```

- **LangGraph** orchestrates agent workflow
- **FastAPI** serves ranking API
- **Streamlit** provides recruiter dashboard

---

## Slide 6: Job Understanding Agent

Extracts structured JSON:

```json
{
  "skills": ["Python", "Machine Learning", "NLP"],
  "experience": "3-8 years",
  "behavior": ["Ownership", "Communication"],
  "domain": "HR Tech / ML Systems"
}
```

Offline fallback: rule-based JD parser for submission reproducibility

---

## Slide 7: Candidate Understanding Agent

Analyzes resume + skills + certifications + Redrob signals:

```json
{
  "candidate_id": "CAND_0001234",
  "skills": ["PyTorch", "NLP", "RAG"],
  "experience_years": 5.2,
  "behavior": ["Self Learner", "Responsive"]
}
```

Computes **skill trust** and **title-career consistency** to catch keyword stuffing

---

## Slide 8: Embedding & Semantic Retrieval

- Model: `sentence-transformers/all-MiniLM-L6-v2` (offline-safe)
- Documents: headline + summary + career descriptions + skills + signals
- Storage: numpy cache + ChromaDB
- Retrieval: cosine similarity → top 500 pool → deep re-rank to top 100

**No keyword matching in retrieval stage**

---

## Slide 9: Ranking Formula (100 pts)

| Component | Weight |
|-----------|--------|
| Skill Match | 30 |
| Project Relevance | 20 |
| Experience Match | 20 |
| Behavioral Match | 15 |
| Learning Potential | 15 |

Final score × trust multiplier (skill assessments + career consistency)

---

## Slide 10: Explainable AI

For each candidate:

- **Why selected** — semantic fit + verified signals
- **Why ranked higher** — component score breakdown
- **Missing skills** — explicit gaps vs JD
- **Strengths / weaknesses** — recruiter-readable bullets
- **Recommendation** — Strong Hire / Interview / Consider / Pass

---

## Slide 11: Anti-Honeypot Strategy

Challenge dataset includes misleading profiles:

- Accountant listing advanced LLM skills with no ML career evidence
- Penalties: low title-career consistency, missing assessments, weak tenure
- Title relevance weighted against **career history**, not headline alone

Result: genuine ML engineers rise above keyword stuffers

---

## Slide 12: Evaluation Metrics

- Score distribution & component averages
- Honeypot flag rate in top-100
- Recommendation mix
- Submission validator compliance (monotonic scores, unique ranks)
- Runtime: offline ranking within challenge compute limits

---

## Slide 13: Demo & Deliverables

**Deliverables:**

- GitHub repo with production code
- `rank.py` one-command submission pipeline
- Streamlit dashboard + FastAPI
- `submission.csv` + `ranked_candidates.csv`
- Architecture docs + this deck (PDF)

**Live demo:** Upload JD → Run Ranking → Inspect candidate insights → Download CSV

---

## Slide 14: Why This Works

- Semantic understanding beats keyword filters
- Multi-signal scoring mirrors recruiter judgment
- Explainability builds trust in AI shortlists
- Offline reproducibility meets hackathon constraints
- Modular agents allow LLM upgrade path (Gemini 2.5 Pro)

---

## Slide 15: Thank You

**Team:** [Your Team Name]

**Repo:** [GitHub URL]

**Contact:** [Email]

Questions?
