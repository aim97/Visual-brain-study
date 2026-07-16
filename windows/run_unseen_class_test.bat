@echo off
REM Unseen-class generalization test (t-SNE figure, fig:tsne; the "reached up
REM to 68%% ... classifier reached 99%% accuracy" numbers in the "Unseen
REM classes test" subsection). Trains on 36 classes, holds out 4, then probes
REM the held-out representations with an SVM. High-gamma (55-95 Hz) band only,
REM matching the paper.
REM
REM Only NeuroStream and NeuroStream4D implement forward_features(), which
REM this protocol needs for the post-training SVM probe - other baselines
REM will fail at that step.
REM
REM The paper reports this as a single demonstrative run rather than a
REM seeded/CI'd metric, so this defaults to seed 1 only. Pass --seeds to
REM repeat it if you want variance here too.
REM
REM Usage:
REM   run_unseen_class_test.bat
REM   run_unseen_class_test.bat --seeds 1 2 3
REM   (any extra arguments are forwarded to run_experiments.py)

call "%~dp0_activate_env.bat" || exit /b 1

python -m eeg_visual_classification.scripts.run_experiments --only unseen_class --seeds 1 %*
