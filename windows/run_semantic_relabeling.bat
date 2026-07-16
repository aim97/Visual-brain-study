@echo off
REM Semantic relabeling split (Table: "Testing accuracy of EEG classification
REM models across filtering methods using semantic relabeling" / tab:eeg_filtering_results,
REM "semantic relabeling" rows). LSTM, EEGConformer, NeuroStream4D across all
REM 4 frequency-band dataset versions, 5 seeds each.
REM
REM Usage:
REM   run_semantic_relabeling.bat
REM   run_semantic_relabeling.bat --dry-run
REM   (any extra arguments are forwarded to run_experiments.py)

call "%~dp0_activate_env.bat" || exit /b 1

python -m eeg_visual_classification.scripts.run_experiments --only semantic --seeds 1 2 3 4 5 %*
