@echo off
setlocal
cd /d "%~dp0"
call "%~dp0.venv\Scripts\activate.bat" 2>nul
if errorlevel 1 (
  echo Virtual env not found. Run setup_env.bat first.
  exit /b 1
)
python rank.py --out outputs\submission.csv
python validate_submission.py outputs\submission.csv
endlocal
