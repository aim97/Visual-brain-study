# NeuroStream: EEG Visual Stimulus Classification

Code for the *NeuroStream: Spectral-Spatio-Temporal Deep Learning for Visual
Stimulus Classification from EEG* paper. This README explains how to
reproduce every table/figure in the paper and how to run the multi-seed
reruns requested in review.

## 1. Setup

```bash
conda create -n eeg-visual python=3.10
conda activate eeg-visual
pip install -r requirements.txt
```

No `pip install -e .` is required — every script under
`eeg_visual_classification/scripts/` adds the repo root to `sys.path`
itself, so it can be run directly from anywhere:

```bash
python -m eeg_visual_classification.scripts.standard_classification -mt EEGNET -expn my_run
```

On Windows, see **Section 5** for `.bat` launchers that activate the conda
env and run the sweeps for you.

## 2. Repo layout

```
eeg_visual_classification/
  models/         model definitions + MODEL_REGISTRY (models/registery.py)
  utils/          data loading, the shared training loop, checkpoint I/O
  scripts/        every runnable entry point (see table below)
  resources/      small, checked-in split index files (see Section 3)
windows/          .bat launchers wrapping the scripts above
data/             NOT checked in - raw/filtered EEG .pth files you supply
stored models/    NOT checked in - checkpoints from standard/unseen-class runs
semantic models/  NOT checked in - checkpoints from semantic/session runs
```

## 3. Data layout

`data/` is gitignored (the EEG recordings are large and not ours to
redistribute). Scripts resolve dataset/split paths relative to
`eeg_visual_classification/`, so place files as:

```
eeg_visual_classification/data/block/eeg_signals_raw_with_mean_std.pth
eeg_visual_classification/data/block/eeg_5_95_std.pth
eeg_visual_classification/data/block/eeg_14_70_std.pth
eeg_visual_classification/data/block/eeg_55_95_std.pth
eeg_visual_classification/data/block/block_splits_by_image_all.pth   # from the original EEGCVPR40 release
eeg_visual_classification/data/processed/semantic_eeg_raw.pth
eeg_visual_classification/data/processed/semantic_eeg_5_95_std.pth
eeg_visual_classification/data/processed/semantic_eeg_14_70_std.pth
eeg_visual_classification/data/processed/semantic_eeg_55_95_std.pth
```

`eeg_visual_classification/resources/` **is** checked in and already has the
split-index files this repo generated (see `scripts/generate_*.py` /
`scripts/build_semantic_dataset.py` if you need to regenerate them):

- `resources/SemanticSplits/semantic_splits.pth` - semantic relabeling split
- `resources/SemanticSplits/session_splits.pth` - session-based/isolated-classes split
- `resources/unseenclasses_path.pth` - unseen-class (4 held-out classes) split

These are already the defaults the scripts point at, so you generally only
need to supply the `data/` EEG files.

> These filenames are the naming convention used throughout the code
> (`bias_analysis.py`, `run_experiments.py`) but weren't verified against
> real files on this machine - if yours differ, either rename them or edit
> the `*_DATASETS` lists at the top of
> `eeg_visual_classification/scripts/run_experiments.py`.

## 4. Table-by-table reproduction guide

| Paper table / figure | Script | What it does |
|---|---|---|
| *"Detailed layer-by-layer hyperparameter settings..."* (`tab:neurostream4d_architecture`) | `summarize_model.py` | Prints a torchinfo summary of the optimal NeuroStream-SST config. `windows\run_architecture_summary.bat` |
| *"The Experimental hyperparameters..."* (`tab:TraningParams`) | n/a | Fixed values already hardcoded as CLI defaults in `utils/lib.py::create_parser` (lr=1e-3, decay 50% every 30 epochs, Adam, 200 epochs, batch 16) |
| *"Best validation accuracy..."* (`tab:baselineAcc`) and *"Maximum testing accuracy..."* (`tab:VisualRepresentations`) | `standard_classification.py` | Standard 70/15/15 split, all 40 classes/subjects, across the 4 frequency-band dataset versions. `windows\run_standard_split.bat` |
| *"Comparison of EEG Classification Models in Terms of Parameter Count and Evaluation Time"* (`tab:model-comparison`) | `evaluate_models.py` | Profiles params/MACs/FLOPs/latency on random input - no data files needed. `windows\run_latency_comparison.bat` |
| *"Ablation study of NeuroStream architectural and representation components"* (`tab:ablation`) | `standard_classification.py` with `-mp` overrides | High-gamma band only, one run per row (B0, R1-R6, A1-A7, C0-C3). `windows\run_ablation_study.bat` runs all rows; see its comments for the exact param → row mapping |
| *"The accuracy per subject..."* (`tab:Subject-acc`) and confusion-matrix figure (`fig:confusion`) | `bias_analysis.py` | Post-hoc: loads the **best checkpoint per dataset** for a given model+params from a standard-split run, re-evaluates with real per-subject IDs. Requires `run_standard_split.bat` to have produced checkpoints first. `windows\run_subject_bias_analysis.bat NeuroStream4D base=48 n_blocks=3` |
| *"Testing accuracy of EEG classification models across filtering methods using semantic relabeling"* (`tab:eeg_filtering_results`), *"semantic relabeling"* rows | `semantic_classification.py` with `-sp resources/SemanticSplits/semantic_splits.pth` | 10-way semantic-category split. `windows\run_semantic_relabeling.bat` |
| Same table, *"semantic relabeling + isolated classes"* rows; sample distribution (`tab:semantic_distribution`, "Session Split" rows) | `semantic_classification.py` with `-sp resources/SemanticSplits/session_splits.pth` | Same 10-way head, held-out classes come from unseen sessions. `windows\run_session_split.bat` |
| *"The t-SNE graph of the EEG features of four unseen classes"* (`fig:tsne`); "reached up to 68%... classifier reached 99% accuracy" (Unseen classes test) | `separate_classification.py` | Trains on 36 classes, holds out 4, probes `forward_features()` output with an SVM + t-SNE. High-gamma band only. Only `NeuroStream`/`NeuroStream4D` implement `forward_features()`. `windows\run_unseen_class_test.bat` |
| Spectrogram example figure (STFT vs CWT) | `generate_spec_samples.py` | One-off plot from a single channel/sample |
| Sample-distribution table (`tab:semantic_distribution`) | `semantic_distribution.py` / `semantic_categories_rel_analysis.py` | Inspects the split files directly, no training |

## 5. Windows `.bat` launchers (multi-seed)

All in `windows/`. Every one of them:
1. Activates the conda env named in `windows\_activate_env.bat` (**edit
   `CONDA_ENV` at the top of that file once**, to match your env name).
2. `cd`s to the repo root regardless of where you double-click/run it from.
3. Forwards any extra arguments you type after the `.bat` name straight to
   the underlying script.

| Batch file | Runs |
|---|---|
| `run_standard_split.bat` | Standard split, all baselines + NeuroStream(-SST), all 4 dataset versions, seeds 1-5 |
| `run_semantic_relabeling.bat` | Semantic relabeling split, seeds 1-5 |
| `run_session_split.bat` | Session/isolated-classes split, seeds 1-5 |
| `run_unseen_class_test.bat` | Unseen-class SVM/t-SNE probe, seed 1 (single demonstrative run by default - see its comments) |
| `run_all_seeded_experiments.bat` | All of the above, back to back |
| `run_ablation_study.bat` | All 18 ablation rows, single seed each |
| `run_latency_comparison.bat` | Table 4 profiling (no seeds/data needed) |
| `run_architecture_summary.bat` | Architecture table (no seeds/data needed) |
| `run_subject_bias_analysis.bat MODEL [key=value ...]` | Per-subject accuracy + confusion-matrix figure for an already-trained model |
| `aggregate_seed_results.bat` | Summarizes accuracy/macro-F1/95% CI across all seeds run so far |

Examples:

```bat
windows\run_standard_split.bat --dry-run
windows\run_standard_split.bat --models NeuroStream4D EEGNET --seeds 1 2 3
windows\run_semantic_relabeling.bat
windows\run_subject_bias_analysis.bat NeuroStream4D base=48 n_blocks=3
```

All the seeded launchers go through
`eeg_visual_classification/scripts/run_experiments.py`, which **skips any
(model, dataset, seed) combination that already has a checkpoint** - safe
to Ctrl+C a multi-day sweep and re-launch the same command later to resume.
`--dry-run` prints the full command matrix without running anything, useful
to sanity-check before committing to a long run.

To change which models/datasets/seeds are swept by default, edit the lists
at the top of `run_experiments.py` (`STANDARD_BASELINES`,
`SEMANTIC_BASELINES`, `STANDARD_DATASETS`, `SEMANTIC_DATASETS`, etc.).

## 6. Aggregating results across seeds

```bash
python -m eeg_visual_classification.scripts.aggregate_results --models-dir "stored models" --output standard_summary.json
python -m eeg_visual_classification.scripts.aggregate_results --models-dir "semantic models" --output semantic_summary.json
```

Reads every checkpoint's stored confusion matrix (no re-inference) and
reports, per (model, dataset): mean/std top-1 accuracy, mean/std macro-F1,
mean 95% CI width, and the list of seeds found. This is what feeds the
"repeated-run variability" numbers reviewers asked for.

## 7. Manually running a single experiment

Every training script shares the same core flags:

```
-mt / --model_type      registry key, e.g. NeuroStream4D (see models/registery.py)
-mp / --model_params     key=value pairs, e.g. -mp bin_size=30 base=48 n_blocks=3
-ed / --eeg-dataset      path to the .pth EEG file
-sp / --splits-path      path to the .pth split file
-sd / --seed             random seed (default 1)
-expn / --experiment-name   free-form tag, becomes part of the checkpoint path
-b / --batch_size, -e / --epochs, -lr / --learning-rate, --no-cuda, ...
```

```bash
python -m eeg_visual_classification.scripts.standard_classification \
    -mt NeuroStream4D -mp bin_size=30 base=48 n_blocks=3 \
    -ed data/block/eeg_55_95_std.pth -sp data/block/block_splits_by_image_all.pth \
    -sd 1 -expn my_run
```

`summarize_model.py` (torchinfo summary) and `evaluate_models.py`
(params/MACs/FLOPs/latency) don't need any data files - they just build the
model and run it on random input of the right shape.

## 8. Known caveats

- `braindecode`'s model API (`ShallowConvNet`, `DeepConvNet`, `EEGInception`,
  `EEGITNet`, `ATCNet`, `EEGConformer`) was verified against the latest PyPI
  release at the time this was written, not against whatever version is
  pinned in your conda env. Run `run_architecture_summary.bat` (or
  `summarize_model.py -mt <name>`) on each new baseline once before a long
  run to catch any API drift early.
- `bias_analysis.py` finds a checkpoint by re-computing its hash from
  `model_type` + `-mp` params, so those must **exactly** match what you
  passed when training (it always adds `n_classes=40` itself, same as
  `standard_classification.py`).
- The ablation → `model_params` mapping in `run_ablation_study.bat` is
  inferred from the `NeuroStream4D`/`NeuroStream` constructors matched
  against each row's description in the paper, not from original run logs.
  The R6 row's 67.08% should match the "NeuroStream-R6" row in
  `tab:VisualRepresentations` as a cross-check - worth confirming before
  trusting the rest of the mapping.
