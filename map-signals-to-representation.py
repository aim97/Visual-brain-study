# Define options
import argparse
import pickle

# Imports
# import sys
# import os
# import random
# import math
# import time
import torch
import os
import glob

torch.utils.backcompat.broadcast_warning.enabled = True

from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
from scipy.fftpack import fft, rfft, fftfreq, irfft, ifft, rfftfreq
from scipy import signal
import numpy as np
import models
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
    default=r"data\block\eeg_signals_raw_with_mean_std.pth",
    help="EEG dataset path",
)  # raw
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
    default="AttnSleep",
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
    default="./build/AttnSleep/AttnSleep__subject0_epoch_199.pth",
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

# Parse arguments
opt = parser.parse_args()
print(opt)

module = importlib.import_module("models." + opt.model_type)
model = module.Model()
optimizer = getattr(torch.optim, opt.optim)(model.parameters(), lr=opt.learning_rate)

# Setup CUDA
if not opt.no_cuda:
    model.cuda()
    print("Copied to CUDA")

if opt.pretrained_net != "":
    model = torch.load("./build/AttnSleep/AttnSleep__subject0_epoch_199.pth")
    print(model)
    model.output_features = True

# load model weights from provided file in the params list
# model.load_state_dict(
#     torch.load(
#         "./build/AttnSleep/AttnSleep__subject0_epoch_199.pth", weights_only=False
#     )
# )

# set model to evaluation mode
model.eval()


# create dataset
dataset = EEGDataset(opt)
sz = dataset.size

data = {"features": [], "images": [], "labels": []}

# read line by line from image_order adding each line in an array
image_order = []
with open(r"image_order.txt", "r") as f:
    for line in f:
        image_order.append(line.strip())

for i in range(sz):
    eeg, label, image_idx = dataset.get_record(i)
    image = image_order[image_idx]
    print(image, label)
    features = model(eeg.cuda())
    data["features"].append(features.cpu())
    data["images"].append(image)
    data["labels"].append(label)
# save data with torch
torch.save(data, "brainFeature-image-mapping.pth")
