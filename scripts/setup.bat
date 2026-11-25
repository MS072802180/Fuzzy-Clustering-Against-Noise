@echo off
REM Creates .venv in the project folder and installs everything (Windows).
cd /d "%~dp0\.."
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ is required'" || exit /b 1
%PY% -m venv .venv || exit /b 1
.venv\Scripts\python -m pip install --upgrade pip
REM CPU-only PyTorch (much smaller download than the default CUDA build)
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu || .venv\Scripts\python -m pip install torch
.venv\Scripts\python -m pip install -r requirements.txt || exit /b 1
.venv\Scripts\python -m pytest -q tests
echo.
echo Setup complete. In VS Code: Ctrl+Shift+P -^> "Python: Select Interpreter" -^> .venv, then press F5.
