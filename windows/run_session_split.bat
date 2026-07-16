@echo off
REM Session-based / isolated-classes split (Table: tab:eeg_filtering_results,
REM "semantic relabeling + isolated classes" rows; sample distribution in
REM tab:semantic_distribution, "Session Split (Isolated Classes)" rows).
REM Same 10-way semantic head as the relabeling split, but held-out classes
REM come from entirely unseen sessions. 5 seeds each.
REM
REM Usage:
REM   run_session_split.bat
REM   run_session_split.bat --dry-run
REM   (any extra arguments are forwarded to run_experiments.py)

call "%~dp0_activate_env.bat" || exit /b 1

python -m eeg_visual_classification.scripts.run_experiments --only semantic_isolated_classes --seeds 1 2 3 4 5 %*
