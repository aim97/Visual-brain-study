"""some summary"""

import os
from pathlib import Path


import torch
import pandas as pd

from eeg_visual_classification.utils.lib import (
    create_parser,
    extract_model_options,
    get_dataloaders,
    get_model_hash,
    save_checkpoint,
    load_checkpoint,
)
from ..models import MODEL_REGISTRY

torch.utils.backcompat.broadcast_warning.enabled = True  # type: ignore

import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim

cudnn.benchmark = True


# definitions
PKG_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.getenv("EEG_MODELS_DIR", PKG_ROOT / "semantic models"))
DATA_DIR = Path(os.getenv("EEG_DATA_DIR", PKG_ROOT / "data" / "processed"))
DEFAULT_SPLITS_PATH = r"data\processed\semantic_splits.pth"
DEFAULT_EEG_DATASET = r"data\processed\semantic_eeg_5_95_std.pth"

# Parse arguments
opt = create_parser(
    default_eeg_dataset=DEFAULT_EEG_DATASET,
    split_path_default=DEFAULT_SPLITS_PATH,
).parse_args()
DATASET_TYPE = opt.eeg_dataset.split("\\")[-1].split(".")[0]
model_options = extract_model_options(opt.model_params)
model_options["n_classes"] = 10
MODEL_HASH = get_model_hash(opt.model_type, model_options)
SAVING_PATH = os.path.join(
    MODELS_DIR,
    f"{opt.model_type}_{MODEL_HASH}_{opt.experiment_name}/{DATASET_TYPE}",
)


print(opt)
if not os.path.exists(SAVING_PATH):
    os.makedirs(SAVING_PATH)
else:
    exit("Experiment name already exists. Choose another name.")

# Create model
# module = importlib.import_module("models." + opt.model_type)
# model = module.Model(**model_options)
ModelClass = MODEL_REGISTRY.get(opt.model_type)
if ModelClass is None:
    raise ValueError(f"Model {opt.model_type} not found in MODEL_REGISTRY.")
model = ModelClass(**model_options)
optimizer = getattr(torch.optim, opt.optim)(model.parameters(), lr=opt.learning_rate)

# Setup CUDA
if not opt.no_cuda:
    model.cuda()
    print("Copied to CUDA")

if opt.pretrained_net != "":
    # model = torch.load(opt.pretrained_net)
    model, _ = load_checkpoint(
        opt.pretrained_net,
        device="cuda",
    )
    print(model)

# Load data
dataset_options = {
    "eeg_dataset": os.path.join(PKG_ROOT, opt.eeg_dataset),
    "subject": opt.subject,
    "time_low": opt.time_low,
    "time_high": opt.time_high,
    "splits_path": os.path.join(PKG_ROOT, opt.splits_path),
    "split_num": opt.split_num,
    "batch_size": opt.batch_size,
    "model_type": opt.model_type,
    "is_semantic": True,
}
loaders = get_dataloaders(dataset_options, ["train", "val"])
split_names = list(loaders.keys())

losses_per_epoch = {split: [] for split in split_names}
accuracies_per_epoch = {split: [] for split in split_names}

best_accuracy = 0
best_accuracy_val = 0
best_epoch = 0
# Start training

predicted_labels = []
correct_labels = []

for epoch in range(1, opt.epochs + 1):
    # Initialize loss/accuracy variables
    losses = {split: 0.0 for split in split_names}
    accuracies = {split: 0.0 for split in split_names}
    counts = {split: 0 for split in split_names}
    # Adjust learning rate for SGD
    if epoch % opt.learning_rate_decay_every == 0:
        lr = max(
            opt.learning_rate
            * (opt.learning_rate_decay_by ** (epoch // opt.learning_rate_decay_every)),
            opt.learning_rate_decay_limit,
        )
        print(f"Learning rate dropped to : {lr}")
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    # Process each split
    for split in loaders.keys():
        # Set network mode
        if split == "train":
            model.train()
            torch.set_grad_enabled(True)
        else:
            model.eval()
            torch.set_grad_enabled(False)
        class_list = list(range(10))
        conf_mat = pd.DataFrame(0, index=class_list, columns=class_list, dtype=float)
        # Process all split batches
        for i, (input_eeg, target) in enumerate(loaders[split]):
            # Check CUDA
            if not opt.no_cuda:
                input_eeg = input_eeg.to("cuda")
                target = target.to("cuda")
            # Forward
            output = model(input_eeg)

            # Compute loss
            loss = F.cross_entropy(output, target)
            losses[split] += loss.item()
            # Compute accuracy
            _, pred = output.data.max(1)
            correct = pred.eq(target.data).sum().item()
            accuracy = correct / input_eeg.data.size(0)
            accuracies[split] += accuracy
            counts[split] += 1

            # compute confusion matrix
            pred_list = pred.cpu().tolist()
            target_list = target.cpu().tolist()
            for i, j in zip(target_list, pred_list):
                conf_mat.loc[i, j] += 1

            # Backward and optimize
            if split == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # Print info at the end of the epoch
    if accuracies["val"] / counts["val"] >= best_accuracy_val:
        best_accuracy_val = accuracies["val"] / counts["val"]
        best_epoch = epoch

    TrL = losses["train"] / counts["train"]
    TrA = accuracies["train"] / counts["train"]
    VL = losses["val"] / counts["val"]
    VA = accuracies["val"] / counts["val"]
    conf_mat = conf_mat.div(conf_mat.sum(axis=1), axis=0)

    print(
        f"Model: {opt.model_type} - Subject {opt.subject} - Time interval: [{opt.time_low}-{opt.time_high}] "
        f"- Epoch {epoch}: TrL={TrL:.4f}, TrA={TrA:.4f}, VL={VL:.4f}, VA={VA:.4f}, "
        f"best accuracy reached at epoch {best_epoch:d}"
    )

    if len(accuracies_per_epoch["val"]) == 0 or VA > max(accuracies_per_epoch["val"]):
        save_checkpoint(
            model,
            optimizer,
            epoch,
            SAVING_PATH,
            opt,
            MODEL_HASH,
            model_options,
            dataset_options,
            losses_per_epoch,
            accuracies_per_epoch,
            TrL,
            TrA,
            VL,
            VA,
            VL,
            VA,
            conf_mat,
        )

    losses_per_epoch["train"].append(TrL)
    losses_per_epoch["val"].append(VL)
    accuracies_per_epoch["train"].append(TrA)
    accuracies_per_epoch["val"].append(VA)

    # stop training if 60 epochs have passed without improvement in validation accuracy
    if epoch - best_epoch >= 60:
        print(
            f"Early stopping at epoch {epoch} due to no improvement in validation accuracy for 60 epochs."
        )
        break
