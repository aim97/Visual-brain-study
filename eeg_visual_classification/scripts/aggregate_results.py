"""Scan saved checkpoints, group them by (model, dataset), and compute
mean/std top-1 accuracy, macro-F1, and a 95% CI across the repeated-seed
runs directly from each checkpoint's stored confusion matrix - no
re-inference needed.

Run directly: python eeg_visual_classification/scripts/aggregate_results.py \
    --models-dir "semantic_models" --output semantic_summary.json
"""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from eeg_visual_classification.utils import load_checkpoint
from eeg_visual_classification.utils.evaluation_metrics import compute_macro_f1_and_ci

PKG_ROOT = Path(__file__).resolve().parents[1]


def aggregate_checkpoints(models_dir):
    """
    Scans the directory for all checkpoints, groups them by Model and Dataset,
    and computes aggregated statistics across the random seeds.
    """
    model_files = glob.glob(str(Path(models_dir) / "**" / "*.pth"), recursive=True)

    # Dictionary to hold grouping: [Model][Dataset] -> list of run dicts
    grouped_results = {}

    for file_path in model_files:
        # We don't need the model state dict, just the metadata
        _, checkpoint = load_checkpoint(file_path, device="cpu")

        model_name = checkpoint.get("model_name", "Unknown_Model")
        dataset_name = Path(checkpoint["dataset_options"]["eeg_dataset"]).stem

        conf_mat = checkpoint.get("confusion_matrx")
        if conf_mat is None:
            continue  # Skip if older checkpoint without matrix

        subject_acc = checkpoint["dataset_options"].get("subject_validation_acc", {})
        seed = checkpoint["dataset_options"].get("seed")

        # Calculate true metrics from raw matrix
        acc, macro_f1, ci = compute_macro_f1_and_ci(conf_mat)

        run_data = {
            "seed": seed,
            "model_hash": checkpoint["model_hash"],
            "epoch": checkpoint["epoch"],
            "top1_accuracy": acc,
            "macro_f1": macro_f1,
            "confidence_interval": ci,
            "subject_bias": subject_acc,
            "checkpoint": file_path,
        }

        grouped_results.setdefault(model_name, {}).setdefault(dataset_name, []).append(
            run_data
        )

    return generate_summary_report(grouped_results)


def generate_summary_report(grouped_results):
    final_report = {}

    for model, datasets in grouped_results.items():
        final_report[model] = {}
        for dataset, runs in datasets.items():

            accs = [r["top1_accuracy"] for r in runs]
            f1s = [r["macro_f1"] for r in runs]
            cis = [r["confidence_interval"] for r in runs]

            # Extract average per-subject accuracy across seeds
            all_subjects = set(sub for r in runs for sub in r["subject_bias"].keys())
            subject_avg = {}
            for sub in all_subjects:
                sub_scores = [r["subject_bias"][sub] for r in runs if sub in r["subject_bias"]]
                subject_avg[f"Subject_{sub}"] = round(sum(sub_scores) / len(sub_scores), 4)

            final_report[model][dataset] = {
                "total_seeds_run": len(runs),
                "seeds": sorted({r["seed"] for r in runs if r["seed"] is not None}),
                "metrics": {
                    "top1_accuracy_mean": round(float(np.mean(accs)), 4),
                    "top1_accuracy_std": round(float(np.std(accs)), 4),
                    "macro_f1_mean": round(float(np.mean(f1s)), 4),
                    "macro_f1_std": round(float(np.std(f1s)), 4),
                    "confidence_interval_mean": round(float(np.mean(cis)), 4),
                },
                "average_subject_accuracy": subject_avg,
                "raw_runs": runs,
            }

    return final_report


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate results across seeds")
    parser.add_argument(
        "--models-dir",
        default=str(PKG_ROOT / "semantic_models"),
        help="Directory to scan recursively for *.pth checkpoints",
    )
    parser.add_argument(
        "--output",
        default="results_summary.json",
        help="Path to write the aggregated JSON report to",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Aggregating checkpoint data from {args.models_dir} (no re-inference)...")
    report = aggregate_checkpoints(args.models_dir)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=4)

    print(f"Aggregation complete! {len(report)} model(s) summarized to {args.output}.")
