@echo off
REM Table: "Comparison of EEG Classification Models in Terms of Parameter
REM Count and Evaluation Time (ms)" / tab:model-comparison.
REM Single deterministic run - no data files or seeds needed, just profiles
REM each architecture's params/MACs/FLOPs/inference time on random input of
REM the right shape. Edit the models_to_test dict in evaluate_models.py to
REM add/remove entries.

call "%~dp0_activate_env.bat" || exit /b 1

python -m eeg_visual_classification.scripts.evaluate_models
