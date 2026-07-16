"""_summary_
Utility functions for EEG visual classification project."""

import glob
import os
import argparse
from pathlib import Path
from datetime import datetime
import platform
import random
import hashlib

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


from .EEGDataset import EEGDataset
from .Splitter import Splitter
from ..models import instantiate_model


RANDOM_STATE = 42

readRecord = lambda data, recordNo: data["dataset"][recordNo]["eeg"].numpy()
readSignal = lambda data, recordNo, channelNo: readRecord(data, recordNo)[channelNo]


def evaluate_clf(clf, X_test, y_test, class_names):
    """
    Print metrics and show confusion matrix (heatmap if seaborn available).
    """
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== Test Accuracy: {acc:.4f} ===\n")
    print("=== Classification Report ===")
    print(
        classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    )


def train_svm_rbf(X_train, y_train, random_state=RANDOM_STATE):
    """
    Build and fit a StandardScaler + RBF SVM pipeline.
    """
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    C=1.0,
                    kernel="rbf",
                    gamma="scale",
                    probability=False,
                    random_state=random_state,
                ),
            ),
        ]
    )
    clf.fit(X_train, y_train)
    return clf


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    class_names=None,
    title: str = "t-SNE of EEG Representations",
    figsize=(9, 7),
    random_state: int = 42,
    max_points: int | None = None,
):
    """
    Visualize 2D t-SNE embedding of feature representations with class-colored scatter.

    Parameters
    ----------
    features : np.ndarray
        Array of shape [N, D] with feature vectors.
    labels : np.ndarray
        Array of shape [N] with integer class labels.
    class_names : list[str] or None
        Optional list of class names indexed by label; falls back to string labels.
    title : str
        Title for the plot.
    figsize : tuple
        Figure size in inches.
    random_state : int
        Random state for reproducibility.
    max_points : int or None
        If set, randomly subsample this many points for faster t-SNE on large datasets.

    Returns
    -------
    None
    """
    if features.ndim != 2:
        raise ValueError(f"`features` must be 2D [N, D], got shape {features.shape}")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("`labels` must be 1D and same length as `features` rows.")

    # Optional subsampling to speed up t-SNE for large N
    if max_points is not None and features.shape[0] > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(features.shape[0], size=max_points, replace=False)
        features = features[idx]
        labels = labels[idx]

    n_samples = features.shape[0]
    # Perplexity must be < n_samples; pick a safe value (typical 5–50)
    perplexity = max(5, min(30, (n_samples - 1) // 3))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
    )
    X_2d = tsne.fit_transform(features)

    # Plot
    plt.figure(figsize=figsize)
    cmap = plt.get_cmap("tab20")
    unique_labels = np.unique(labels)

    # Fallback class names if not provided
    if class_names is None:
        try:
            class_names = [str(c) for c in sorted(unique_labels)]
        except Exception:
            class_names = [str(c) for c in unique_labels]

    for idx, lab in enumerate(unique_labels):
        mask = labels == lab
        name = (
            class_names[lab]
            if (isinstance(lab, (int, np.integer)) and lab < len(class_names))
            else str(lab)
        )
        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=18,
            alpha=0.85,
            color=cmap(idx % 20),
            label=name,
        )

    plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()


def delete_files(pattern):
    """deletes files with given name pattern

    Args:
        pattern (str): file pattern to delete
    """
    # Get a list of all file paths that match the pattern
    files = glob.glob(pattern)
    # Iterate over the list of file paths and remove each file
    for file in files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")


def create_parser(
    split_path_default="data/block/block_splits_by_image_all.pth",
    default_eeg_dataset="data/block/eeg_55_95_std.pth",
):
    parser = argparse.ArgumentParser(description="Template")

    # Dataset options

    # Data - Data needs to be pre-filtered and filtered data is available

    ### BLOCK DESIGN ###
    # Data
    # parser.add_argument('-ed', '--eeg-dataset', default=r"data\block\eeg_55_95_std.pth", help="EEG dataset path") #55-95Hz
    parser.add_argument(
        "-ed",
        "--eeg-dataset",
        default=default_eeg_dataset,
        help="EEG dataset path",
    )  # 5-95Hz
    # parser.add_argument('-ed', '--eeg-dataset', default=r"data\block\eeg_14_70_std.pth", help="EEG dataset path") #14-70Hz
    # Splits
    parser.add_argument(
        "-sp",
        "--splits-path",
        default=split_path_default,
        help="splits path",
    )  # All subjects
    # parser.add_argument('-sp', '--splits-path', default=r"data\block\block_splits_by_image_single.pth", help="splits path") #Single subject
    # BLOCK DESIGN ###

    parser.add_argument(
        "-sn", "--split-num", default=0, type=int, help="split number"
    )  # leave this always to zero.

    # Subject selecting
    parser.add_argument(
        "-sub",
        "--subject",
        default=0,
        type=int,
        help="choose a subject from 1 to 6, default is 0 (all subjects)",
    )

    # Time options: select from 20 to 460 samples from EEG data
    parser.add_argument(
        "-tl", "--time_low", default=20, type=float, help="lowest time value"
    )
    parser.add_argument(
        "-th", "--time_high", default=460, type=float, help="highest time value"
    )

    # Model type/options
    parser.add_argument(
        "-mt",
        "--model_type",
        default="lstm",
        help="specify which generator should be used: lstm|EEGChannelNet",
    )
    # It is possible to test out multiple deep classifiers:
    # - lstm is the model described in the paper "Deep Learning Human Mind for Automated Visual Classification”, in CVPR 2017
    # - model10 is the model described in the paper "Decoding brain representations by multimodal learning of neural activity and visual features", TPAMI 2020
    parser.add_argument(
        "-mp",
        "--model_params",
        default="",
        nargs="*",
        help="list of key=value pairs of model options",
    )
    parser.add_argument(
        "--pretrained_net",
        default="",
        help="path to pre-trained net (to continue training)",
    )

    # Training options
    parser.add_argument("-b", "--batch_size", default=16, type=int, help="batch size")
    parser.add_argument("-o", "--optim", default="Adam", help="optimizer")
    parser.add_argument(
        "-lr", "--learning-rate", default=0.001, type=float, help="learning rate"
    )
    parser.add_argument(
        "-lrdb",
        "--learning-rate-decay-by",
        default=0.125,
        type=float,
        help="learning rate decay factor",
    )
    parser.add_argument(
        "-lrde",
        "--learning-rate-decay-every",
        default=30,
        type=int,
        help="learning rate decay period",
    )

    # learning rate decay limit
    parser.add_argument(
        "-lrl",
        "--learning-rate-decay-limit",
        default=0.0000001,
        type=float,
        help="learning rate decay limit",
    )

    parser.add_argument(
        "-dw", "--data-workers", default=4, type=int, help="data loading workers"
    )
    parser.add_argument("-e", "--epochs", default=200, type=int, help="training epochs")

    # Save options
    parser.add_argument(
        "-sc", "--saveCheck", default=100, type=int, help="learning rate"
    )

    # Backend options
    parser.add_argument(
        "--no-cuda", default=False, help="disable CUDA", action="store_true"
    )

    parser.add_argument(
        "-expn", "--experiment-name", help="experiment name", required=True
    )

    parser.add_argument(
        "-sd",
        "--seed",
        default=1,
        type=int,
        help="random seed for model init, data shuffling and any stochastic op",
    )
    return parser


def set_seed(seed: int):
    """Seed python/numpy/torch (CPU + all CUDA devices) for one reproducible run.

    Call this once, right after argument parsing and before building the model
    or dataloaders, so that weight initialization and batch shuffling are both
    covered by the seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def get_device(no_cuda: bool) -> str:
    """Resolve the torch device, falling back to CPU if CUDA was requested
    but is not actually available (e.g. quick local smoke-tests before moving
    a run to a GPU machine)."""
    return "cuda" if (not no_cuda and torch.cuda.is_available()) else "cpu"


def get_dataloaders(opt, splits=["train", "val", "test"]):
    """Generates the dataset splits

    Args:
        opt (dict): dataset options
        splits (list, optional): list of dataset split name. Defaults to ["train", "val", "test"].

    Returns:
        dict[str, DataLoader]: data loader for each split
    """
    # Load dataset
    dataset = EEGDataset(opt)
    is_semantic = "is_semantic" in opt
    # Create loaders
    loaders = {
        split: DataLoader(
            Splitter(
                dataset,
                split_path=opt["splits_path"],
                split_num=opt["split_num"],
                split_name=split,
                is_semantic=is_semantic,
                combined_test_val=(
                    True if "combined_test_val" not in opt else opt["combined_test_val"]
                ),
                return_full_record=(
                    (split == "test" and opt["return_full_record"])
                    if "return_full_record" in opt
                    else False
                ),
            ),
            batch_size=opt["batch_size"],
            drop_last=True,
            shuffle=True,
        )
        for split in splits
    }
    return loaders


def parse_value(value):
    if value.isdigit():
        return int(value)
    if value[0].isdigit():
        return float(value)
    if value[0] == "(" and value[-1] == ")":
        # Parse tuple
        items = value[1:-1].split(",")
        return tuple(parse_value(item.strip()) for item in items)
    if value[0] == "[" and value[-1] == "]":
        # Parse list
        items = value[1:-1].split(",")
        return [parse_value(item.strip()) for item in items]
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return value


def extract_model_options(model_params):
    return {
        key: parse_value(value) for (key, value) in [x.split("=") for x in model_params]
    }


def get_model_hash(model_name, model_params):
    param_str = "_".join(
        [f"{key}={value}" for key, value in sorted(model_params.items())]
    )
    compressed = f"{model_name}-{param_str}"
    return hashlib.sha256(compressed.encode()).hexdigest()


def to_float(x):
    # Convert tensors/ndarrays/nums to Python float
    try:
        return float(x.item())  # 0-D torch.tensor
    except AttributeError:
        try:
            return float(x)  # Python/numpy scalar
        except Exception:
            return x  # leave as-is (e.g., list of floats)


def get_simple_conf(outputs, targets):
    class_list = list(set(targets))
    conf_mat = pd.DataFrame(0, index=class_list, columns=class_list, dtype=float)
    # conf_mat = pd.DataFrame({cn: [0] * len(class_list) for cn in class_list})
    for o, t in zip(outputs, targets):
        conf_mat.loc[t, o] += 1
    conf_mat = conf_mat.div(conf_mat.sum(axis=1), axis=0)
    return conf_mat


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    SAVING_PATH: str | Path,
    opt,
    MODEL_HASH: str,
    model_options: dict,
    dataset_options: dict,
    losses_per_epoch,
    accuracies_per_epoch,
    TrL,
    TrA,
    VL,
    VA,
    TeL,
    TeA,
    conf_mat=None,
    lr_scheduler=None,
    amp_scaler=None,
):
    save_dir = Path(SAVING_PATH)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Ensure metrics are floats
    TrL = to_float(TrL)
    TrA = to_float(TrA)
    VL = to_float(VL)
    VA = to_float(VA)
    TeL = to_float(TeL)
    TeA = to_float(TeA)

    checkpoint = {
        "schema_version": 1,
        "date": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "epoch": epoch,
        "model_name": opt.model_type,
        "experiment_name": getattr(opt, "experiment_name", None),
        "subject": getattr(opt, "subject", None),
        # Reconstruction inputs (ensure these are sufficient to init the model)
        "model_options": model_options,  # constructor args / hyperparams
        "dataset_options": dataset_options,
        # Core states
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler else None,
        "amp_scaler_state_dict": amp_scaler.state_dict() if amp_scaler else None,
        # Metrics snapshot
        "metrics": {
            "train_loss": TrL,
            "train_accuracy": TrA,
            "val_loss": VL,
            "val_accuracy": VA,
            "test_loss": TeL,
            "test_accuracy": TeA,
        },
        "metrics_per_epoch": {
            "losses_per_epoch": losses_per_epoch,
            "accuracies_per_epoch": accuracies_per_epoch,
        },
        # Provenance
        "model_hash": MODEL_HASH,
        "env": {
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "cuda_is_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.version.cuda else None,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        # Optional: RNG states for perfect resume
        "torch_rng_state": torch.get_rng_state().tolist(),
        "numpy_rng_state": np.random.get_state(),  # requires numpy
        "python_rng_state": random.getstate(),
        "confusion_matrx": conf_mat,
    }

    # Canonical filename, avoid spaces
    fname = f"{opt.model_type}__{MODEL_HASH}.pth"
    target = save_dir / fname
    tmp = save_dir / (fname + ".tmp")

    # Atomic save
    torch.save(checkpoint, tmp)
    tmp.replace(target)

    print(f"Saved checkpoint to: {target}")

    return target


def load_checkpoint(ckpt_path: str | Path, device: str = "cpu"):

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    required = ["model_name", "model_state_dict", "model_options"]
    for k in required:
        if k not in ckpt or ckpt[k] is None:
            raise ValueError(f"Checkpoint missing '{k}': {ckpt_path}")

    model_name = ckpt["model_name"]
    model = instantiate_model(model_name, **ckpt["model_options"])

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        print(f"[warn] Missing keys: {missing}\n[warn] Unexpected keys: {unexpected}")

    model.to(device)
    model.eval()
    return model, ckpt
