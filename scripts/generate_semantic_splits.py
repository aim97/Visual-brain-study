"""_summary_"""

import argparse
import random
import torch
import numpy as np


is_semantic_matched = lambda s: np.vectorize(lambda x: x["semantic"] == s)

parser = argparse.ArgumentParser(description="Template")

parser.add_argument(
    "-id",
    "--input-dataset",
    help="input EEG dataset path",
)

parser.add_argument(
    "-sp",
    "--output-splits",
    help="output EEG dataset path",
)

opt = parser.parse_args()

dataset_path = opt.input_dataset
split_path = opt.output_splits
dataset = torch.load(dataset_path)

# for each semantic find the indices and split them into train and test sets
semantics = dataset["semantics"]

np_dataset = np.array(dataset["dataset"])

split = {"train": [], "val": []}

for semantic in range(len(semantics)):
    semantic_indices = np.where(is_semantic_matched(semantic)(np_dataset))[0]
    split_idx = int(semantic_indices.shape[0] * 0.8)
    train_split, test_split = semantic_indices[:split_idx], semantic_indices[split_idx:]
    split["train"].extend(train_split.tolist())
    split["val"].extend(test_split.tolist())

# assertions
assert len(split["train"]) == len(set(split["train"])), "Found repetition in train set"
assert len(split["val"]) == len(set(split["val"])), "Found repetition in test set"
assert set(split["val"]).isdisjoint(
    split["train"]
), "Found overlap between train and test sets"
assert (
    len(set(split["train"])) + len(set(split["val"])) == np_dataset.shape[0]
), "Some data samples are dropped"

random.shuffle(split["train"])
random.shuffle(split["val"])

torch.save(split, split_path)
