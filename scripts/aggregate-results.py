import glob
import os
import json
import pandas as pd
from pathlib import Path
from ..utils import load_checkpoint
import numpy as np
# Import the metric calculation logic
from ..utils.evaluation_metrics import compute_macro_f1_and_ci

PKG_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PKG_ROOT / "semantic_models"

def aggregate_checkpoints():
    """
    Scans the directory for all checkpoints, groups them by Model and Dataset,
    and computes aggregated statistics across the random seeds.
    """
    model_files = glob.glob(f"{MODELS_DIR}/**/*.pth", recursive=True)

    # Dictionary to hold grouping: [Model][Dataset] -> list of run dicts
    grouped_results = {}

    for file_path in model_files:
        # We don't need the model state dict, just the metadata
        _, checkpoint = load_checkpoint(file_path, map_location="cpu")

        model_name = checkpoint.get("model_name", "Unknown_Model")
        dataset_name = checkpoint["dataset_options"]["eeg_dataset"].split("\\")[-1].split(".")[0]

        conf_mat = checkpoint.get("confusion_matrx")
        if conf_mat is None:
            continue # Skip if older checkpoint without matrix

        subject_acc = checkpoint["dataset_options"].get("subject_validation_acc", {})

        # Calculate true metrics from raw matrix
        acc, macro_f1, ci = compute_macro_f1_and_ci(conf_mat)

        run_data = {
            "seed_hash": checkpoint["model_hash"],
            "epoch": checkpoint["epoch"],
            "top1_accuracy": acc,
            "macro_f1": macro_f1,
            "confidence_interval": ci,
            "subject_bias": subject_acc
        }

        if model_name not in grouped_results:
            grouped_results[model_name] = {}
        if dataset_name not in grouped_results[model_name]:
            grouped_results[model_name][dataset_name] = []

        grouped_results[model_name][dataset_name].append(run_data)

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
                "metrics": {
                    "top1_accuracy_mean": round(float(np.mean(accs)), 4),
                    "top1_accuracy_std": round(float(np.std(accs)), 4),
                    "macro_f1_mean": round(float(np.mean(f1s)), 4),
                    "macro_f1_std": round(float(np.std(f1s)), 4),
                    "confidence_interval_mean": round(float(np.mean(cis)), 4)
                },
                "average_subject_accuracy": subject_avg,
                "raw_runs": runs
            }

    return final_report

# TODO: make it accept the output as as a parameter
if __name__ == "__main__":
    print("Aggregating checkpoint data without running inference...")
    report = aggregate_checkpoints()

    output_file = "bias_analysis_summary.json"
    with open(output_file, "w") as f:
        json.dump(report, f, indent=4)

    print(f"Aggregation complete! Clean data written to {output_file}.")
