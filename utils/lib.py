import glob
import os
import argparse
from .EEGDataset import EEGDataset
from .Splitter import Splitter
from torch.utils.data import DataLoader


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
    return str(abs(hash(compressed)))
