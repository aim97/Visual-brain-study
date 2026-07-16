"""Cross-platform (Windows/Linux/Mac) driver for the multi-seed experiment
sweeps requested by reviewers: N seeds x every model x every dataset
version, for each evaluation protocol. Replaces run.sh, which is bash-only
and therefore cannot run in a native Windows terminal / plain `python`
conda environment.

Usage (from anywhere, this script finds the repo root itself):
    python eeg_visual_classification/scripts/run_experiments.py --dry-run
    python eeg_visual_classification/scripts/run_experiments.py --only standard
    python eeg_visual_classification/scripts/run_experiments.py --seeds 1 2 3 4 5

Each (script, model, dataset, seed) combination is skipped if a checkpoint
for it already exists, so an interrupted multi-day sweep can just be
re-launched to pick up where it left off. Failures are logged and do not
abort the rest of the sweep; a summary prints at the end.

EDIT THE `EXPERIMENTS` LIST BELOW to match your exact filenames - the
dataset paths here are the repo's documented naming convention
(see methods.tex / bias_analysis.py) but were never verified against real
data files on this machine, so double check them before a long run.
"""
import argparse
import glob
import subprocess
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parent

# ---------------------------------------------------------------------------
# Baselines to sweep. Registry keys - see eeg_visual_classification/models/registery.py
# ---------------------------------------------------------------------------
STANDARD_BASELINES = [
    "lstm",
    "BLSTM",
    "EEGNET",
    "EEGChannelNet",
    "AttnSleep",
    "EEGConformer",
    "ATCNet",
    # Newly added baselines addressing the missing multi-branch / multi-scale /
    # deep-convolutional-pathway CNN comparisons Reviewer 2 asked for:
    "ShallowConvNet",
    "DeepConvNet",
    "EEGInception",
    "EEGITNet",
    "NeuroStream",
    "NeuroStream4D",  # NeuroStream-SST in the paper
]

SEMANTIC_BASELINES = ["lstm", "EEGConformer", "NeuroStream4D"]

STANDARD_DATASETS = [
    "data/block/eeg_signals_raw_with_mean_std.pth",
    "data/block/eeg_5_95_std.pth",
    "data/block/eeg_14_70_std.pth",
    "data/block/eeg_55_95_std.pth",
]

SEMANTIC_DATASETS = [
    "data/processed/semantic_eeg_raw.pth",
    "data/processed/semantic_eeg_5_95_std.pth",
    "data/processed/semantic_eeg_14_70_std.pth",
    "data/processed/semantic_eeg_55_95_std.pth",
]

# The unseen-class protocol probes the trained encoder's forward_features()
# output with an SVM, so it only makes sense for models that implement that
# method (currently NeuroStream and NeuroStream4D). The paper reports this
# on the high-gamma (55-95 Hz) band only.
UNSEEN_CLASS_MODELS = ["NeuroStream", "NeuroStream4D"]
UNSEEN_CLASS_DATASETS = ["data/block/eeg_55_95_std.pth"]

# ---------------------------------------------------------------------------
# Each entry: (name, script module, model list, dataset list, extra CLI args)
# `extra_args` is a list of strings appended verbatim to every invocation.
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    (
        "standard",
        "eeg_visual_classification.scripts.standard_classification",
        STANDARD_BASELINES,
        STANDARD_DATASETS,
        ["-sp", "data/block/block_splits_by_image_all.pth"],
    ),
    (
        "semantic",
        "eeg_visual_classification.scripts.semantic_classification",
        SEMANTIC_BASELINES,
        SEMANTIC_DATASETS,
        ["-sp", "resources/SemanticSplits/semantic_splits.pth"],
    ),
    (
        "semantic_isolated_classes",
        "eeg_visual_classification.scripts.semantic_classification",
        SEMANTIC_BASELINES,
        SEMANTIC_DATASETS,
        ["-sp", "resources/SemanticSplits/session_splits.pth"],
    ),
    (
        "unseen_class",
        "eeg_visual_classification.scripts.separate_classification",
        UNSEEN_CLASS_MODELS,
        UNSEEN_CLASS_DATASETS,
        ["-sp", "resources/unseenclasses_path.pth"],
    ),
]


def already_ran(models_dir: Path, model_type: str, seed: int, dataset_stem: str) -> bool:
    if not models_dir.exists():
        return False
    pattern = str(models_dir / f"{model_type}_*_seed{seed}" / dataset_stem / "*.pth")
    return len(glob.glob(pattern)) > 0


def build_command(script_module, model, dataset, seed, extra_args, experiment_name):
    return [
        sys.executable,
        "-m",
        script_module,
        "-mt",
        model,
        "-ed",
        dataset,
        "-sd",
        str(seed),
        "-expn",
        experiment_name,
        *extra_args,
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="Random seeds to sweep"
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[e[0] for e in EXPERIMENTS],
        help="Restrict to a subset of experiment sets (default: all)",
    )
    parser.add_argument(
        "--models", nargs="+", help="Restrict to a subset of model names (default: all in each set)"
    )
    parser.add_argument(
        "--experiment-name",
        default="reviewer_reruns",
        help="Passed through as -expn / used to namespace saved checkpoints",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip a (model, dataset, seed) combo if a matching checkpoint already exists (default: on)",
    )
    parser.add_argument(
        "--no-skip-existing", dest="skip_existing", action="store_false"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the commands without running them"
    )
    args = parser.parse_args()

    sets = [e for e in EXPERIMENTS if args.only is None or e[0] in args.only]

    total, skipped, failed, ran = 0, 0, [], 0
    for name, script_module, models, datasets, extra_args in sets:
        model_list = [m for m in models if args.models is None or m in args.models]
        models_dir = REPO_ROOT / (
            "semantic models" if "semantic" in script_module else "stored models"
        )

        for seed in args.seeds:
            for model in model_list:
                for dataset in datasets:
                    total += 1
                    dataset_stem = Path(dataset).stem

                    if args.skip_existing and already_ran(models_dir, model, seed, dataset_stem):
                        print(f"[skip] {name}: {model} / {dataset_stem} / seed={seed} (checkpoint exists)")
                        skipped += 1
                        continue

                    cmd = build_command(
                        script_module, model, dataset, seed, extra_args, args.experiment_name
                    )
                    print(f"[run ] {name}: {' '.join(cmd)}")

                    if args.dry_run:
                        continue

                    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
                    if result.returncode != 0:
                        failed.append((name, model, dataset_stem, seed))
                        print(f"[FAIL] {name}: {model} / {dataset_stem} / seed={seed} "
                              f"(exit code {result.returncode}) - continuing with the rest of the sweep")
                    else:
                        ran += 1

    print("\n=== Sweep summary ===")
    print(f"Total combinations: {total}  |  Ran: {ran}  |  Skipped (already done): {skipped}  |  Failed: {len(failed)}")
    if failed:
        print("Failed combinations:")
        for name, model, dataset_stem, seed in failed:
            print(f"  - {name}: {model} / {dataset_stem} / seed={seed}")


if __name__ == "__main__":
    main()
