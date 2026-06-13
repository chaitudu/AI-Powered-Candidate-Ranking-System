@echo off
setlocal
cd /d "%~dp0"
call "%~dp0.venv\Scripts\activate.bat" 2>nul
if errorlevel 1 (
  echo Virtual env not found. Run setup_env.bat first.
  exit /b 1
)
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
endlocal
