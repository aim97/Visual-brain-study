@echo off
REM Runs every seeded experiment set back to back: standard split, semantic
REM relabeling, session/isolated-classes split, and the unseen-class test.
REM This is the "just rerun everything the reviewers asked for" button - it
REM can take a very long time (models x datasets x seeds). Safe to Ctrl+C and
REM re-launch later: already-finished (model, dataset, seed) combinations are
REM skipped automatically.
REM
REM Usage:
REM   run_all_seeded_experiments.bat
REM   run_all_seeded_experiments.bat --dry-run     -> see the full plan first
REM   (any extra arguments are forwarded to run_experiments.py)

call "%~dp0_activate_env.bat" || exit /b 1

python -m eeg_visual_classification.scripts.run_experiments --seeds 1 2 3 4 5 %*
