@echo off
REM Standard split (Table: "Best validation accuracy..." / tab:baselineAcc,
REM and Table: "Maximum testing accuracy..." / tab:VisualRepresentations).
REM All baselines + NeuroStream + NeuroStream4D (NeuroStream-SST), across all
REM 4 frequency-band dataset versions, 5 seeds each.
REM
REM Usage:
REM   run_standard_split.bat                     -> full sweep, seeds 1-5
REM   run_standard_split.bat --dry-run            -> print commands, don't run
REM   run_standard_split.bat --models NeuroStream4D --seeds 1 2 3
REM   (any extra arguments are forwarded to run_experiments.py)

call "%~dp0_activate_env.bat" || exit /b 1

python -m eeg_visual_classification.scripts.run_experiments --only standard --seeds 1 2 3 4 5 %*
