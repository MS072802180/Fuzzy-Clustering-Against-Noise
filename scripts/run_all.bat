@echo off
REM Runs tests then the whole study.  Usage: scripts\run_all.bat [configs\full.yaml]
cd /d "%~dp0\.."
set CFG=%1
if "%CFG%"=="" set CFG=configs\full.yaml
python -m pytest -q tests || exit /b 1
python main.py --config %CFG%
