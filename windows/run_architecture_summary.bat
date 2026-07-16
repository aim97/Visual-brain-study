@echo off
REM Table: "Detailed layer-by-layer hyperparameter settings and tensor
REM dimension transformations of the optimal NeuroStream-SST architecture" /
REM tab:neurostream4d_architecture (architecture-details.tex).
REM Prints a torchinfo layer-by-layer summary for the C0 / best-found
REM NeuroStream4D configuration (30 aggregated frequencies, 48 stem kernels,
REM 3 R(2+1)D blocks - see tab:ablation, row C0).

call "%~dp0_activate_env.bat" || exit /b 1

python -m eeg_visual_classification.scripts.summarize_model -mt NeuroStream4D -mp bin_size=30 base=48 n_blocks=3
