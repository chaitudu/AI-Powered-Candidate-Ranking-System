# Troubleshooting (Windows)

## 1. Streamlit / SSL error with Anaconda

**Error:**
```
ImportError: DLL load failed while importing _ssl
```

**Cause:** Anaconda Python 3.9 on your machine has a broken OpenSSL DLL (`libssl-3-x64.dll` missing).

**Fix:** Use the project virtual environment (Python 3.11):

```cmd
cd C:\Users\chaitu chey\Documents\ai-candidate-ranking-system
setup_env.bat
run_streamlit.bat
```

Ranking (`rank.py`) still works with Anaconda, but **Streamlit and FastAPI require SSL**.

---

## 2. CMD vs PowerShell syntax

| Wrong in CMD | Correct in CMD |
|--------------|------------------|
| `& "python.exe" script.py` | `"C:\...\python.exe" script.py` |
| `# comment` | `REM comment` |
| `outputs/` | `dir outputs` or `cd outputs` |

Use the provided `.bat` files to avoid syntax issues.

---

## 3. uvicorn not found

Run `setup_env.bat` once to install all dependencies into `.venv`.

---

## 4. View outputs folder

```cmd
dir outputs
type outputs\submission.csv
```

---

## 5. Validate submission

```cmd
.venv\Scripts\python.exe validate_submission.py outputs\submission.csv
```

Expected: `Submission is valid.`
