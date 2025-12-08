import pickle
import argparse
import glob
import os
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import pandas as pd

from ..utils import EEGDataset, load_checkpoint, ValidationOnlySplitter, get_model_hash

# orig_torch_load = torch.load


# def torch_wrapper(*args, **kwargs):
#     logging.warning(
#         "[comfyui-unsafe-torch] I have unsafely patched `torch.load`.  The `weights_only` option of `torch.load` is forcibly disabled."
#     )
#     kwargs["weights_only"] = False

#     return orig_torch_load(*args, **kwargs)


# torch.load = torch_wrapper

# NODE_CLASS_MAPPINGS = {}
# __all__ = ["NODE_CLASS_MAPPINGS"]

PKG_ROOT = Path(__file__).resolve().parents[1]  # .../eeg_visual_classification
MODELS_DIR = PKG_ROOT / "stored models"
DATA_DIR = PKG_ROOT / "data" / "block"


def get_pretrained_model_paths(model_names, dataset_names):
    models = []
    for model_name in model_names:
        for dataset_name in dataset_names:
            model_path_pattern = (
                f"{MODELS_DIR}/{model_name}*/{dataset_name}/{model_name}_*.pth"
            )

            # find all files matching the pattern
            model_files = glob.glob(model_path_pattern)

            models.extend([(model_name, dataset_name, f) for f in model_files])
    return models


def get_model_path_by_hash(model_hash, dataset_names):
    model_path_pattern = f"{MODELS_DIR}/*{model_hash}*/*/*.pth"
    model_path_pattern = model_path_pattern.replace("\\", "\\/")

    # print(f"Searching for models with pattern: {model_path_pattern}")
    model_files = glob.glob(model_path_pattern)
    model_datasets = {}
    for dataset_name in dataset_names:
        model_datasets[dataset_name] = []
        for model_file in model_files:
            # print(f"Checking model file: {model_file}")
            if dataset_name in model_file:
                model_datasets[dataset_name].append(model_file)
    return model_datasets


def load_checkpoint_data(dataset_name, model_file):
    # load the checkpoint data
    model, checkpoint = load_checkpoint(model_file)
    ret = {
        "model": model,
        "checkpoint": checkpoint,
        "acc": checkpoint["metrics"]["val_accuracy"],
        "model_options": checkpoint["model_options"],
        "epoch": checkpoint["epoch"],
        "dataset_name": dataset_name,
        "model_file": model_file,
        "hash": checkpoint["model_hash"],
    }
    return ret


def find_best_models(dataset_models):
    best_models = {}
    for dataset_name in dataset_models:
        model_files = dataset_models[dataset_name]
        best_acc = -1.0
        best_model_data = None
        for model_file in model_files:
            model_data = load_checkpoint_data(dataset_name, model_file)
            acc = model_data["acc"]
            print(f"Model: {model_file}, Accuracy: {acc}")
            if acc > best_acc:
                best_acc = acc
                best_model_data = model_data
        if best_model_data is not None:
            best_models[dataset_name] = best_model_data

    print("Best models found for datasets:")
    for dataset_name in best_models:
        print(f"  {dataset_name}: {best_models[dataset_name]['model_file']}")
    return best_models


def get_models_coverage(model_names, datasets, models):
    v = {
        model_name: {dataset_name: 0 for dataset_name in datasets}
        for model_name in model_names
    }

    df = pd.DataFrame.from_dict(v, orient="index")
    for model_name, dataset_name, _ in models:
        df.at[model_name, dataset_name] = df.at[model_name, dataset_name] + 1

    return df


def evaluate_model(model, model_type, dataset_path):
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


if __name__ == "__main__":
    model_type = "BrainDecoder3D"
    model_options = {}
    hash = get_model_hash(model_type, model_options)
    datasets = [
        "eeg_signals_raw_with_mean_std",
        "eeg_55_95_std",
        "eeg_14_70_std",
        "eeg_5_95_std",
    ]

    print(f"Looking for models with hash: {hash}")

    models = get_model_path_by_hash(hash, datasets)

    print(models)
    best_models = find_best_models(models)

    for dataset_name, model_data in best_models.items():
        print(f"Evaluating model for dataset {dataset_name}")
        if model_data is None:
            print(f"No model found for dataset {dataset_name}")
            continue
        print(f"Evaluating model for dataset {dataset_name}")
        model = model_data["model"]
        subject_acc, conf = evaluate_model(
            model,
            model_type,
            os.path.join(PKG_ROOT, "data", "block", f"{dataset_name}.pth"),
        )

        print(f"Dataset: {dataset_name}")
        print(f"Model Type: {model_type}")
        print(f"Model Hash: {model_data['hash']}")
        print(f"Per-subject accuracy: {subject_acc}")
        print(f"Confusion Matrix:\n{conf}")

        # save conf matrix as csv
        conf_df = pd.DataFrame(
            conf,
            index=[f"True_{i}" for i in range(conf.shape[0])],
            columns=[f"Pred_{i}" for i in range(conf.shape[1])],
        )
        conf_csv_path = (
            f"confusion_matrix_{dataset_name}_{model_type}_{model_data['hash']}.csv"
        )
        conf_df.to_csv(conf_csv_path)
        print(f"Confusion matrix saved to {conf_csv_path}")
    print("Done")
