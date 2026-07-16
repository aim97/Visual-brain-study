@echo off
REM Scans checkpoints and writes mean/std accuracy, macro-F1, and 95%% CI
REM across seeds, straight from each checkpoint's stored confusion matrix -
REM no re-inference needed. Run this after a seeded sweep finishes (or
REM partway through, to check progress).
REM
REM Writes two files into the repo root:
REM   standard_split_summary.json   (from "stored models\": standard split +
REM                                   unseen-class checkpoints)
REM   semantic_summary.json         (from "semantic models\": semantic
REM                                   relabeling + session-split checkpoints)

call "%~dp0_activate_env.bat" || exit /b 1

python -m eeg_visual_classification.scripts.aggregate_results --models-dir "stored models" --output standard_split_summary.json
python -m eeg_visual_classification.scripts.aggregate_results --models-dir "semantic models" --output semantic_summary.json

echo Wrote standard_split_summary.json and semantic_summary.json
