# Redrob Idea Submission — SemFit AI Labs


## Slide 1 — Team & Problem

**Team Name:** SemFit AI Labs

**Problem Statement:**  
Build an AI-powered candidate ranking system that goes beyond keyword-based hiring filters. Given 100,000 anonymized candidate profiles and a Senior Machine Learning Engineer job description, identify and rank the top 100 most relevant candidates with explainable reasoning — while detecting keyword-stuffed and inconsistent (honeypot) profiles.

**Team Leader Name:** Modalavalasa Chaitanya

---

## Slide 2 — Solution Overview

**What is your proposed solution?**  
SemFit AI is an end-to-end semantic ranking pipeline that:
1. Understands the job description via a Job Understanding Agent
2. Encodes full candidate narratives (not just skill tags) into vector embeddings
3. Retrieves top candidates by cosine similarity (no keyword matching)
4. Re-ranks with a 100-point multi-signal scoring model
5. Explains every decision with strengths, gaps, and evidence-backed reasoning

**What differentiates your approach from traditional candidate matching systems?**

| Traditional ATS | SemFit AI |
|----------------|-----------|
| Boolean keyword filters | Semantic vector similarity on full profiles |
| Skill tag counting | Skill trust + assessment verification |
| Title-only matching | Title + career history consistency checks |
| Black-box scores | Explainable component breakdown (5 dimensions) |
| Vulnerable to keyword stuffing | Honeypot detection via trust multiplier |

---

## Slide 3 — JD Understanding & Candidate Evaluation

**Key requirements extracted from the JD (Senior ML Engineer):**

- **Required Skills:** Python, SQL, Machine Learning, Deep Learning, PyTorch/TensorFlow, NLP, LLMs, RAG, Vector DBs, Feature Engineering, FastAPI
- **Experience:** 3–8 years with ML production exposure
- **Behavior:** Ownership, communication, self-directed learning, collaboration
- **Domain:** HR Tech, recruiting workflows, search/ranking systems

**Most important candidate signals:**

| Signal | Source | Why It Matters |
|--------|--------|----------------|
| Career narrative | `career_history.description` | True project relevance vs keyword claims |
| AI core skills | `skills[]` + assessments | Verified technical depth |
| Title-career consistency | Derived feature | Catches honeypot profiles |
| Experience years | `profile.years_of_experience` | Band fit (3–8 yrs) |
| Platform behavior | `redrob_signals.*` | Engagement, responsiveness, GitHub activity |
| Semantic similarity | Embedding cosine score | Holistic JD-to-profile match |

**How we evaluate fit beyond keyword matching:**  
We embed the entire candidate document (headline + summary + all career roles + skills with proficiency/tenure + Redrob signals) and compare against the job embedding using cosine similarity. Skills are then verified against assessment scores and career evidence — not counted blindly.

---

## Slide 4 — Ranking Methodology

**Retrieve → Score → Rank pipeline:**

1. **Retrieve:** Precomputed 100K embeddings → cosine similarity → top 2,000 pool
2. **Score:** 100-point weighted model across 5 components
3. **Rank:** Sort by composite score → apply trust multiplier → output top 100

**Models, algorithms, and heuristics:**

| Stage | Method |
|-------|--------|
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (online) or sklearn HashingVectorizer (offline fallback) |
| Retrieval | Cosine similarity on normalized vectors |
| Job parsing | Rule-based JD parser + optional Gemini 2.5 Pro agent |
| Scoring | Weighted heuristic model (no black-box ML classifier) |
| Honeypot detection | Title-career consistency + skill trust scoring |
| Orchestration | LangGraph agent workflow |

**Combining multiple signals:**

```
Final Score = (Skill×30 + Project×20 + Experience×20 + Behavior×15 + Learning×15)
              × TrustMultiplier

TrustMultiplier = (0.55 + 0.25×skill_trust + 0.20×consistency) × title_relevance
```

Submission scores mapped as: `1.0 - (rank-1) × 0.008` for portal format compliance.

---

## Slide 5 — Explainability & Data Validation

**How ranking decisions are explained:**
- Per-candidate: strengths, weaknesses, missing skills, natural-language reasoning
- Component score breakdown (skill/project/experience/behavior/learning)
- Recommendation tier: Strong Hire / Interview / Consider / Pass

**Preventing hallucinations:**
- Offline submission mode uses **rule-based reasoning templates** tied to computed features
- No free-form LLM generation during `rank.py` execution
- Every claim in reasoning maps to a measurable field (years, skill count, response rate)

**Handling inconsistent / suspicious profiles:**
- **Title-career consistency score:** penalizes profiles where current title ≠ career history
- **Skill trust score:** penalizes advanced skill claims without assessments or tenure
- **Title relevance multiplier:** reduces score 80% for non-ML titles with no technical career
- Result: HR Managers with 9 AI skills rank below genuine ML Engineers

---

## Slide 6 — End-to-End Workflow

```
┌─────────────────┐
│  Job Description │ (upload or default Senior ML Engineer JD)
└────────┬────────┘
         ▼
┌─────────────────┐
│ Job Understanding│ → Structured JSON: skills, experience, behavior, domain
│     Agent        │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Embedding +      │ → Job vector + 100K precomputed candidate vectors
│ Vector Store     │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Semantic         │ → Top 2,000 by cosine similarity (NO keywords)
│ Retrieval Agent  │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Candidate        │ → Structured profiles + trust/consistency features
│ Understanding    │
└────────┬────────┘
         ▼
┌─────────────────┐
│ AI Ranking Agent │ → 100-point score + explainability
└────────┬────────┘
         ▼
┌─────────────────┐
│ Output           │ → submission.csv + ranked_candidates.csv + JSON report
│ Streamlit / API  │
└─────────────────┘
```

**Reproduce command:**
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

---

## Slide 7 — System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     SemFit AI Platform                        │
├──────────────┬───────────────┬──────────────┬──────────────┤
│   Streamlit  │    FastAPI    │   rank.py    │  precompute  │
│   Dashboard  │     REST API  │  (offline)   │  embeddings  │
├──────────────┴───────────────┴──────────────┴──────────────┤
│                    LangGraph Agent Workflow                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │Job Agent │→│Retrieval │→│Candidate │→│ Ranking  │      │
│  │          │ │  Agent   │ │  Agent   │ │  Agent   │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
├──────────────────────────────────────────────────────────────┤
│  Embeddings (MiniLM / HashingVectorizer)  │  ChromaDB/NumPy │
├──────────────────────────────────────────────────────────────┤
│  Preprocessing │ Feature Engineering │ Honeypot Detection   │
├──────────────────────────────────────────────────────────────┤
│              candidates.jsonl (100,000 profiles)              │
└──────────────────────────────────────────────────────────────┘
```

---

## Slide 8 — Results & Performance

**Ranking quality evidence:**

| Metric | Value |
|--------|-------|
| Top-1 candidate | Senior AI Engineer, 72.7/100, 9 AI core skills |
| Top-100 ML/AI titles | ~70% genuine ML/AI roles |
| Honeypot rate in top-100 | 0% flagged inconsistencies in final list |
| HR Manager in top-100 | 1 (rank ~90+) vs keyword-stuffer trap |
| Submission validation | ✅ Passes official validator |

**Runtime & compute compliance:**

| Constraint | Our Solution |
|-----------|-------------|
| No network during ranking | ✅ Precomputed embeddings, rule-based agents |
| CPU only | ✅ No GPU required |
| ≤5 min ranking | ✅ ~115 seconds after precompute |
| 16 GB RAM | ✅ Vectorized numpy operations |
| Precomputation allowed | ✅ ~7 min one-time embedding build |

---

## Slide 9 — Technologies Used

| Technology | Purpose | Why Selected |
|-----------|---------|-------------|
| **Python 3.11** | Core language | ML ecosystem, challenge compatibility |
| **LangGraph** | Agent orchestration | Structured multi-step AI workflow |
| **Gemini 2.5 Pro** | Optional LLM agents | Job/candidate understanding (online mode) |
| **sentence-transformers** | Embeddings | State-of-art semantic vectors |
| **sklearn HashingVectorizer** | Offline fallback | Works without GPU/network |
| **ChromaDB + NumPy** | Vector storage | Fast similarity search at scale |
| **FastAPI** | REST API | Production-grade backend |
| **Streamlit** | Dashboard | Recruiter-friendly UI |
| **scikit-learn** | Feature engineering | Trust/consistency scoring |
| **pandas / numpy** | Data processing | 100K profile handling |

---

## Slide 10 — Submission Assets

| Asset | Link / Location |
|-------|----------------|
| **GitHub Repository** | [https://github.com/modalavalasachaitanya/semfit-ai-candidate-ranker](https://github.com/chaitudu/AI-Powered-Candidate-Ranking-System)|
| **Ranked Output CSV** | `outputs/submission.csv` |
| **Extended Results** | `outputs/ranked_candidates.csv` |
| **Reproduce Command** | `python rank.py --candidates ./candidates.jsonl --out ./submission.csv` |
| **Streamlit Demo** | `streamlit run frontend/app.py` |
| **API Demo** | `uvicorn api.main:app --reload --port 8000` |
| **Team Leader** | Modalavalasa Chaitanya |
| **Team Name** | SemFit AI Labs |

---

*End of Submission Deck — SemFit AI Labs*
