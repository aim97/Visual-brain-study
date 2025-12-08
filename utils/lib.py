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

from .EEGDataset import EEGDataset
from .Splitter import Splitter
from ..models import MODEL_REGISTRY

readRecord = lambda data, recordNo: data["dataset"][recordNo]["eeg"].numpy()
readSignal = lambda data, recordNo, channelNo: readRecord(data, recordNo)[channelNo]


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


def create_parser():
    parser = argparse.ArgumentParser(description="Template")

    # Dataset options

    # Data - Data needs to be pre-filtered and filtered data is available

    ### BLOCK DESIGN ###
    # Data
    # parser.add_argument('-ed', '--eeg-dataset', default=r"data\block\eeg_55_95_std.pth", help="EEG dataset path") #55-95Hz
    parser.add_argument(
        "-ed",
        "--eeg-dataset",
        default=r"data\block\eeg_5_95_std.pth",
        help="EEG dataset path",
    )  # 5-95Hz
    # parser.add_argument('-ed', '--eeg-dataset', default=r"data\block\eeg_14_70_std.pth", help="EEG dataset path") #14-70Hz
    # Splits
    parser.add_argument(
        "-sp",
        "--splits-path",
        default=r"data\block\block_splits_by_image_all.pth",
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
        default=0.5,
        type=float,
        help="learning rate decay factor",
    )
    parser.add_argument(
        "-lrde",
        "--learning-rate-decay-every",
        default=10,
        type=int,
        help="learning rate decay period",
    )

    # learning rate decay limit
    parser.add_argument(
        "-lrl",
        "--learning-rate-decay-limit",
        default=0.00001,
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
    return parser


def get_dataloaders(opt):
    # Load dataset
    dataset = EEGDataset(opt)
    # Create loaders
    loaders = {
        split: DataLoader(
            Splitter(
                dataset,
                split_path=opt["splits_path"],
                split_num=opt["split_num"],
                split_name=split,
            ),
            batch_size=opt["batch_size"],
            drop_last=True,
            shuffle=True,
        )
        for split in ["train", "val", "test"]
    }
    return loaders


def extract_model_options(model_params):
    return {
        key: (
            int(value)
            if value.isdigit()
            else (float(value) if value[0].isdigit() else value)
        )
        for (key, value) in [x.split("=") for x in model_params]
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

    # Optional: convert per-epoch metrics to plain lists of floats
    losses_per_epoch = [to_float(v) for v in losses_per_epoch]
    accuracies_per_epoch = [to_float(v) for v in accuracies_per_epoch]

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
    ModelClass = MODEL_REGISTRY[model_name]
    model = ModelClass(**ckpt["model_options"])

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        print(f"[warn] Missing keys: {missing}\n[warn] Unexpected keys: {unexpected}")

    model.to(device)
    model.eval()
    return model, ckpt
