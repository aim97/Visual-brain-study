@echo off
REM Table: "The accuracy per subject achieved by our proposed NeuroStream
REM model..." / tab:Subject-acc, and Figure: "The confusion matrix of our
REM NeuroStream model across the three versions of the dataset" / fig:confusion
REM (images/conf_mats.png in the paper).
REM
REM Loads the BEST checkpoint per dataset for the given model+params (must
REM match how it was trained - same -mp options), re-evaluates it on the
REM validation split with real per-subject IDs, and plots side-by-side
REM confusion matrices.
REM
REM Requires standard-split training to have already produced checkpoints
REM (run_standard_split.bat) for the model/params you pass here.
REM
REM Usage:
REM   run_subject_bias_analysis.bat NeuroStream4D
REM   run_subject_bias_analysis.bat NeuroStream4D base=48 n_blocks=3
REM (first arg = model type, remaining args = -mp key=value pairs, forwarded
REM  as-is - must match the -mp options used when training that checkpoint)

call "%~dp0_activate_env.bat" || exit /b 1

set MODEL=%1
if "%MODEL%"=="" (
    echo Usage: run_subject_bias_analysis.bat MODEL_TYPE [key=value ...]
    echo Example: run_subject_bias_analysis.bat NeuroStream4D base=48 n_blocks=3
    exit /b 1
)
shift

python -m eeg_visual_classification.scripts.bias_analysis -mt %MODEL% -mp %1 %2 %3 %4 %5 %6 --savefig confusion_matrices.png
