import pickle
import argparse
import glob
import os
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import logging

from ..utils.ValidationOnlySplitter import ValidationOnlySplitter
from ..utils.EEGDataset import EEGDataset


orig_torch_load = torch.load


def torch_wrapper(*args, **kwargs):
    logging.warning(
        "[comfyui-unsafe-torch] I have unsafely patched `torch.load`.  The `weights_only` option of `torch.load` is forcibly disabled."
    )
    kwargs["weights_only"] = False

    return orig_torch_load(*args, **kwargs)


torch.load = torch_wrapper

NODE_CLASS_MAPPINGS = {}
__all__ = ["NODE_CLASS_MAPPINGS"]

PKG_ROOT = Path(__file__).resolve().parents[1]  # .../eeg_visual_classification
MODELS_DIR = PKG_ROOT / "stored models"
DATA_DIR = PKG_ROOT / "data" / "block"


datasets = [
    "eeg_signals_raw_with_mean_std",
    "eeg_55_95_std",
    "eeg_14_70_std",
    "eeg_5_95_std",
]


def get_pretrained_model_paths(model_name):
    model_path_pattern = f"{MODELS_DIR}/*{model_name}*/{model_name}_*.pth"

    # find all files matching the pattern
    model_files = glob.glob(model_path_pattern)

    acc_files = [
        os.path.join(os.path.dirname(model_file), "accuracies_per_epoch.pkl")
        for model_file in model_files
    ]

    # group all models trained on the same dataset version together
    dataset_models = {dataset: [] for dataset in datasets}
    for model_file, acc_file in zip(model_files, acc_files):
        dataset_name = (
            model_file.split("\\")[-2][len(model_name) + 1 :].split("std")[0] + "std"
        )
        if dataset_name in dataset_models:
            dataset_models[dataset_name].append((model_file, acc_file))
        else:
            print(
                f"Warning: Dataset {dataset_name} not recognized in model file {model_file}"
            )

    return dataset_models


def get_max_val_acc(acc_file):
    """Load accuracy data from a JSON file.

    Args:
        acc_file (str): Path to the accuracy JSON file.
    Returns:

    """
    with open(acc_file, "rb") as f:
        accuracy_data = pickle.load(f)
        acc = max(accuracy_data["val"])
        if type(acc) == tuple:
            acc = acc[0]
    return acc


def get_best_model_paths(dataset_models):
    # get the paths of the best models for each dataset
    best_model_paths = {dataset: (None, 0) for dataset in datasets}
    for dataset, models in dataset_models.items():
        max_acc = -1
        best_model = None
        for model_file, acc_file in models:
            acc = get_max_val_acc(acc_file)
            if acc > max_acc:
                max_acc = acc
                best_model = model_file
        best_model_paths[dataset] = (best_model, max_acc)
    return best_model_paths


def evaluate_model(model_type, model_path, dataset_path):
    dataset = EEGDataset(
        {
            "eeg_dataset": dataset_path,
            "subject": 0,
            "time_low": 20,
            "time_high": 460,
            "model_type": model_type,
        }
    )
    dataset_loader = DataLoader(
        ValidationOnlySplitter(
            dataset,
            split_path=f"{DATA_DIR}/block_splits_by_image_all.pth",
            split_num=0,
        ),
        batch_size=32,
        drop_last=True,
        shuffle=True,
    )

    # load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.load(model_path, map_location="cpu", weights_only=True)
    model.to(device)
    model.eval()

    per_subject_counts = {}  # subject_id -> [correct, total]
    confusion = None
    total_correct = 0
    total_samples = 0

    # evaluate the model on the validation set
    with torch.no_grad():
        for batch in dataset_loader:
            # support batches of (eeg,label,subject) or (eeg,label)
            if len(batch) == 3:
                eeg, label, subject = batch
            else:
                eeg, label = batch
                # if subject not provided set to -1 so it will be grouped separately
                subject = torch.full_like(label, -1)

            eeg = eeg.to(device)
            label = label.to(device)
            subject = subject.to(device)

            outputs = model(eeg)

            # determine predictions for binary or multi-class outputs
            if outputs.ndim == 1 or (outputs.ndim == 2 and outputs.size(1) == 1):
                probs = torch.sigmoid(outputs.view(-1))
                preds = (probs > 0.5).long()
                num_classes = 2
            else:
                preds = outputs.argmax(dim=1)
                num_classes = outputs.size(1)

            # init confusion matrix as [true_label, predicted_label]
            if confusion is None:
                confusion = torch.zeros(
                    (num_classes, num_classes), dtype=torch.int64, device="cpu"
                )

            # iterate element-wise to update per-subject stats and confusion
            for t, p, s in zip(
                label.view(-1).cpu().tolist(),
                preds.view(-1).cpu().tolist(),
                subject.view(-1).cpu().tolist(),
            ):
                s = int(s)
                t = int(t)
                p = int(p)
                counts = per_subject_counts.setdefault(s, [0, 0])
                counts[1] += 1
                if p == t:
                    counts[0] += 1
                    total_correct += 1
                total_samples += 1
                # clamp to valid class indices if needed
                if 0 <= t < confusion.shape[0] and 0 <= p < confusion.shape[1]:
                    confusion[t, p] += 1

    # compute per-subject accuracy
    per_subject_accuracy = {
        sub: (c[0] / c[1] if c[1] > 0 else 0.0) for sub, c in per_subject_counts.items()
    }

    overall_accuracy = (total_correct / total_samples) if total_samples > 0 else 0.0
    print(f"Overall validation accuracy: {overall_accuracy:.4f}")

    # return per-subject accuracy dict and confusion matrix (numpy array)
    return per_subject_accuracy, confusion.cpu().numpy()  # type: ignore


# take model name as command line argument
parser = argparse.ArgumentParser()
parser.add_argument(
    "-mt",
    "--model-type",
    type=str,
    required=True,
    help="Name of the model to analyze bias for",
)

opt = parser.parse_args()
model_name = opt.model_type

dataset_models = get_pretrained_model_paths(model_name)

# display the number of models found for each dataset
for dataset, models in dataset_models.items():
    print(f"Dataset: {dataset}, Number of models found: {len(models)}")

best_model_paths = get_best_model_paths(dataset_models)

# display the best model paths and their accuracies
for dataset, (model_path, acc) in best_model_paths.items():
    print(f"Dataset: {dataset}")
    print(f"Best Model Path: {model_path}")
    print(f"Max Validation Accuracy: {acc:.4f}\n")

# evaluate each best model and display per-subject accuracies
for dataset, (model_path, acc) in best_model_paths.items():
    if model_path is None:
        print(f"No model found for dataset {dataset}, skipping evaluation.")
        continue
    print(f"Evaluating model for dataset {dataset} with path {model_path}")
    dataset_path = f"{DATA_DIR}/{dataset}.pth"
    per_subject_accuracy, confusion = evaluate_model(
        model_name, model_path, dataset_path
    )
    print(f"Per-subject accuracies for dataset {dataset}:")
    for subject, accuracy in per_subject_accuracy.items():
        print(f"Subject {subject}: Accuracy {accuracy:.4f}")
    print(f"Confusion Matrix for dataset {dataset}:\n{confusion}\n")
