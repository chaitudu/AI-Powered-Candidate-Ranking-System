@echo off
setlocal
cd /d "%~dp0"
set PY311=C:\Users\chaitu chey\AppData\Local\Programs\Python\Python311\python.exe

if not exist "%PY311%" (
  echo Python 3.11 not found at %PY311%
  echo Install Python 3.11 from https://www.python.org/downloads/
  exit /b 1
)

echo Creating virtual environment...
"%PY311%" -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import ssl; import streamlit; import fastapi; print('Setup OK')"
echo.
echo Done. Use run_streamlit.bat or run_api.bat next.
endlocal
