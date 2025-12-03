# Define options
import argparse
import pickle
import torch
import os
import glob

torch.utils.backcompat.broadcast_warning.enabled = True # type: ignore

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
import importlib

# import EEGDataset class from utils
from utils.EEGDataset import EEGDataset
from utils.Splitter import Splitter

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
parser.add_argument("-sc", "--saveCheck", default=100, type=int, help="learning rate")

# Backend options
parser.add_argument(
    "--no-cuda", default=False, help="disable CUDA", action="store_true"
)

parser.add_argument(
    "-expn",
    "--experiment-name", help="experiment name", required=True
)

# Parse arguments
opt = parser.parse_args()
print(opt)

dataset_type = opt.eeg_dataset.split("/")[3].split(".")[0]

saving_path = os.path.join("./stored models", f"{opt.model_type}_{dataset_type}_{opt.experiment_name}")
if not os.path.exists(saving_path):
    os.makedirs(saving_path)
else:
    exit("Experiment name already exists. Choose another name.")

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


# Load dataset
dataset = EEGDataset(opt)
# Create loaders
loaders = {
    split: DataLoader(
        Splitter(
            dataset,
            split_path=opt.splits_path,
            split_num=opt.split_num,
            split_name=split,
        ),
        batch_size=opt.batch_size,
        drop_last=True,
        shuffle=True,
    )
    for split in ["train", "val", "test"]
}

# Load model

model_options = {
    key: (
        int(value)
        if value.isdigit()
        else (float(value) if value[0].isdigit() else value)
    )
    for (key, value) in [x.split("=") for x in opt.model_params]
}
# Create discriminator model/optimizer
module = importlib.import_module("models." + opt.model_type)
model = module.Model(**model_options)
optimizer = getattr(torch.optim, opt.optim)(model.parameters(), lr=opt.learning_rate)

# Setup CUDA
if not opt.no_cuda:
    model.cuda()
    print("Copied to CUDA")

if opt.pretrained_net != "":
    model = torch.load(opt.pretrained_net)
    print(model)

# initialize training,validation, test losses and accuracy list
losses_per_epoch = {"train": [], "val": [], "test": []}
accuracies_per_epoch = {"train": [], "val": [], "test": []}

best_accuracy = 0
best_accuracy_val = 0
best_epoch = 0
# Start training

predicted_labels = []
correct_labels = []

for epoch in range(1, opt.epochs + 1):
    # Initialize loss/accuracy variables
    losses = {"train": 0.0, "val": 0.0, "test": 0.0}
    accuracies = {"train": 0, "val": 0, "test": 0}
    counts = {"train": 0, "val": 0, "test": 0}
    # Adjust learning rate for SGD
    if opt.optim == "SGD":
        lr = max(opt.learning_rate * (
            opt.learning_rate_decay_by ** (epoch // opt.learning_rate_decay_every)
        ), opt.learning_rate_decay_limit)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    # Process each split
    for split in ("train", "val", "test"):
        # Set network mode
        if split == "train":
            model.train()
            torch.set_grad_enabled(True)
        else:
            model.eval()
            torch.set_grad_enabled(False)
        # Process all split batches
        for i, (input, target) in enumerate(loaders[split]):
            # Check CUDA
            if not opt.no_cuda:
                input = input.to("cuda")
                target = target.to("cuda")
            # Forward
            output = model(input)

            # Compute loss
            loss = F.cross_entropy(output, target)
            losses[split] += loss.item()
            # Compute accuracy
            _, pred = output.data.max(1)
            correct = pred.eq(target.data).sum().item()
            accuracy = correct / input.data.size(0)
            accuracies[split] += accuracy
            counts[split] += 1
            # Backward and optimize
            if split == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # Print info at the end of the epoch
    if accuracies["val"] / counts["val"] >= best_accuracy_val:
        best_accuracy_val = accuracies["val"] / counts["val"]
        best_accuracy = accuracies["test"] / counts["test"]
        best_epoch = epoch
    
    TrL =    losses["train"] / counts["train"],
    TrA =    accuracies["train"] / counts["train"],
    VL =    losses["val"] / counts["val"],
    VA =    accuracies["val"] / counts["val"],
    TeL =    losses["test"] / counts["test"],
    TeA =    accuracies["test"] / counts["test"],
    
    print(
        "Model: {11} - Subject {12} - Time interval: [{9}-{10}]  [{9}-{10} Hz] - Epoch {0}: TrL={1:.4f}, TrA={2:.4f}, VL={3:.4f}, VA={4:.4f}, TeL={5:.4f}, TeA={6:.4f}, TeA at max VA = {7:.4f} at epoch {8:d}".format(
            epoch,
            losses["train"] / counts["train"],
            accuracies["train"] / counts["train"],
            losses["val"] / counts["val"],
            accuracies["val"] / counts["val"],
            losses["test"] / counts["test"],
            accuracies["test"] / counts["test"],
            best_accuracy,
            best_epoch,
            opt.time_low,
            opt.time_high,
            opt.model_type,
            opt.subject,
        )
    )

    if len(accuracies_per_epoch["val"]) == 0 or VA > max(accuracies_per_epoch["val"]):
        delete_files("%s/%s__subject%d_epoch_*.pth" % (saving_path,opt.model_type, opt.subject))
        torch.save(
            model, "%s/%s__subject%d_epoch_%d.pth" % (saving_path, opt.model_type, opt.subject, epoch)
        )

    losses_per_epoch["train"].append(TrL)
    losses_per_epoch["val"].append(VL)
    losses_per_epoch["test"].append(TeL)
    accuracies_per_epoch["train"].append(TrA)
    accuracies_per_epoch["val"].append(VA)
    accuracies_per_epoch["test"].append(TeA)


# save the loss and accuracy across all epochs
with open(f"{saving_path}/losses_per_epoch.pkl", "wb") as f:
    pickle.dump(losses_per_epoch, f)

with open(f"{saving_path}/accuracies_per_epoch.pkl", "wb") as f:
    pickle.dump(accuracies_per_epoch, f)
