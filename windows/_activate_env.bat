@echo off
REM Shared helper: activates the conda env and cd's to the repo root.
REM Called by every other .bat in this folder via `call`. Not meant to be
REM run directly.

REM ==== EDIT ME: name of the conda environment with this project's deps ====
set CONDA_ENV=eeg-visual

call conda activate %CONDA_ENV%
if errorlevel 1 (
    echo [ERROR] Could not activate conda environment "%CONDA_ENV%".
    echo         Edit CONDA_ENV at the top of windows\_activate_env.bat, or
    echo         run "conda env list" to see the correct name.
    exit /b 1
)

REM %~dp0 = directory this .bat lives in (windows\). Repo root is one level up.
cd /d "%~dp0.."
