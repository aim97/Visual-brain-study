@echo off
REM Table: "Ablation study of NeuroStream architectural and representation
REM components" / tab:ablation. Standard split, high-gamma (55-95 Hz) band
REM only, single run per row (the paper does not report seed variance for
REM the ablation table). ~18 short runs.
REM
REM Row -> model_params mapping is inferred from the NeuroStream4D/NeuroStream
REM constructors (see eeg_visual_classification/models/NeuroStream4D.py and
REM NeuroStream.py) matched against each row's description in the paper.
REM R6 ("No Spectral Component") uses the NeuroStream model (no CWT/STFT
REM front-end at all) instead of NeuroStream4D - its 67.08%% number should
REM match the "NeuroStream-R6" row in tab:VisualRepresentations as a
REM cross-check. Double-check this mapping before trusting the numbers.

call "%~dp0_activate_env.bat" || exit /b 1

set DATA=data/block/eeg_55_95_std.pth
set SPLIT=data/block/block_splits_by_image_all.pth

echo === B0: baseline (bin_size=10, base=32, n_blocks=4 - NeuroStream4D defaults) ===
python -m eeg_visual_classification.scripts.standard_classification -mt NeuroStream4D -ed %DATA% -sp %SPLIT% -expn ablation_B0 -sd 1

echo === R1: F_r=5 ===
python -m eeg_visual_classification.scripts.standard_classification -mt NeuroStream4D -ed %DATA% -sp %SPLIT% -expn ablation_R1 -sd 1 -mp bin_size=5

echo === R2: F_r=20 ===
python -m eeg_visual_classification.scripts.standard_classification -mt NeuroStream4D -ed %DATA% -sp %SPLIT% -expn ablation_R2 -sd 1 -mp bin_size=20

echo === R3: F_r=30 ===
python -m eeg_visual_classification.scripts.standard_classification -mt NeuroStream4D -ed %DATA% -sp %SPLIT% -expn ablation_R3 -sd 1 -mp bin_size=30

echo === R4: F_r=35 ===
python -m eeg_visual_classification.scripts.standard_classification -mt NeuroStream4D -ed %DATA% -sp %SPLIT% -expn ablation_R4 -sd 1 -mp bin_size=35

echo === R5: STFT spectrogram instead of CWT ===
python -m eeg_visual_classification.scripts.standard_classification -mt NeuroStream4D -ed %DATA% -sp %SPLIT% -expn ablation_R5 -sd 1 -mp spec_type=stft

echo === R6: No spectral component (plain NeuroStream, ScalpMaps only) ===
python -m eeg_visual_classification.scripts.standard_classification -mt NeuroStream -ed %DATA% -sp %SPLIT% -expn ablation_R6 -sd 1

echo === A1: 48 Conv3D stem kernels ===
python -m eeg_visual_classification.scripts.standard_classification -mt NeuroStream4D -ed %DATA% -sp %SPLIT% -expn ablation_A1 -sd 1 -mp base=48

echo === A2: 48 stem kernels + 3 R(2+1)D blocks ===
python -m eeg_visual_classification.scripts.standard_classification -mt NeuroStream4D -ed %DATA% -sp %SPLIT% -expn ablation_A2 -sd 1 -mp base=48 n_blocks=3

echo === A3: Reduced depth (2 R(2+1)D blocks) ===
python -m eeg_visual_classification.scripts.standard_classification -mt NeuroStream4D -ed %DATA% -sp %SPLIT% -expn ablation_A3 -sd 1 -mp n_blocks=2

echo === A4: Increased depth (6 R(2+1)D blocks) ===
python -m eeg_visual_classification.scripts.standard_classification -mt NeuroStream4D -ed %DATA% -sp %SPLIT% -expn ablation_A4 -sd 1 -mp n_blocks=6

echo === A5: No initial 3D convolution stem ===
python -m eeg_visual_classification.scripts.standard_classification -mt NeuroStream4D -ed %DATA% -sp %SPLIT% -expn ablation_A5 -sd 1 -mp use_stem=False

echo === A6: Large 3D stem kernel (7x7x7) ===
python -m eeg_visual_classification.scripts.standard_classification -mt NeuroStream4D -ed %DATA% -sp %SPLIT% -expn ablation_A6 -sd 1 -mp "conv3d_kernel_size=(7,7,7)"

echo === A7: Spatial-first processing instead of temporal-first ===
python -m eeg_visual_classification.scripts.standard_classification -mt NeuroStream4D -ed %DATA% -sp %SPLIT% -expn ablation_A7 -sd 1 -mp is_temporal_first=False

echo === C1: 64 kernels, 30 aggregated frequencies, 2 blocks ===
python -m eeg_visual_classification.scripts.standard_classification -mt NeuroStream4D -ed %DATA% -sp %SPLIT% -expn ablation_C1 -sd 1 -mp base=64 bin_size=30 n_blocks=2

echo === C2: 48 kernels, 25 aggregated frequencies, 3 blocks ===
python -m eeg_visual_classification.scripts.standard_classification -mt NeuroStream4D -ed %DATA% -sp %SPLIT% -expn ablation_C2 -sd 1 -mp base=48 bin_size=25 n_blocks=3

echo === C3: 30 aggregated frequencies, 3 blocks ===
python -m eeg_visual_classification.scripts.standard_classification -mt NeuroStream4D -ed %DATA% -sp %SPLIT% -expn ablation_C3 -sd 1 -mp bin_size=30 n_blocks=3

echo === C0: Full proposed NeuroStream-SST (30 freq, 48 kernels, 3 blocks) ===
python -m eeg_visual_classification.scripts.standard_classification -mt NeuroStream4D -ed %DATA% -sp %SPLIT% -expn ablation_C0 -sd 1 -mp base=48 bin_size=30 n_blocks=3

echo Ablation sweep complete.
