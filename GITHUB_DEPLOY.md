# Deploy to GitHub — Step by Step

**Team:** SemFit AI Labs  
**Leader:** Modalavalasa Chaitanya  
**Suggested repo name:** `semfit-ai-candidate-ranker`

---

## Step 1 — Create GitHub Repository

1. Go to [https://github.com/new](https://github.com/new)
2. Repository name: `semfit-ai-candidate-ranker`
3. Description: `AI-powered semantic candidate ranking for Redrob India Runs Challenge`
4. Set to **Public**
5. Do **NOT** initialize with README (we already have one)
6. Click **Create repository**

---

## Step 2 — Push from Your PC

Open **Command Prompt** in the project folder:

```cmd
cd C:\Users\chaitu chey\Documents\ai-candidate-ranking-system

git init
git add .
git commit -m "Initial commit: SemFit AI candidate ranking system for Redrob challenge"

git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/semfit-ai-candidate-ranker.git
git push -u origin main
```

Replace `YOUR_GITHUB_USERNAME` with your actual GitHub username (e.g. `modalavalasachaitanya`).

---

## Step 3 — Upload Submission PDF

1. Open `docs\REDDROB_SUBMISSION_DECK.md` in **Word** or **Google Docs**
2. Format each `---` section as a slide
3. Export as **PDF** (must be under 5 MB)
4. Upload to Redrob portal

**Alternative:** Open `docs\presentation_content.md` — same content, different format.

---

## Step 4 — Upload Ranked CSV

Upload: `outputs\submission.csv` to the Redrob portal (CSV/XLSX field).

Validate first:
```cmd
.venv\Scripts\python.exe validate_submission.py outputs\submission.csv
```

---

## Step 5 — Update Links

After creating the repo, update these files with your real GitHub URL:
- `README.md` (Team section)
- `submission_metadata.yaml` (github_repo field)
- `docs/REDDROB_SUBMISSION_DECK.md` (Slide 10)

Also update `primary_contact.email` in `submission_metadata.yaml` with your real email.

---

## What Gets Pushed vs Excluded

| Included in Git | Excluded (too large) |
|-----------------|---------------------|
| All source code | `.venv/` |
| README, docs, configs | `vector_store/embeddings/*.npy` |
| `outputs/submission.csv` | `candidates.jsonl` (100K dataset) |
| Sample data in `data/` | `ranking_report.json` |

Reviewers reproduce embeddings via `python precompute_embeddings.py`.
