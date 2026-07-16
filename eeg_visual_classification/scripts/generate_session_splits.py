"""_summary_"""

import argparse
import random
import torch
import numpy as np


is_label_found = lambda l: np.vectorize(lambda x: x["label"] in l)
is_found = lambda l: np.vectorize(lambda x: x in l)

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

test_classes = [
    "n03888257",
    "n03584829",
    "n02607072",
    "n11939491",
    "n03590841",
    "n03272010",
    "n02504458",
    "n03272562",
]

opt = parser.parse_args()

dataset_path = opt.input_dataset
split_path = opt.output_splits
dataset = torch.load(dataset_path)


# for each semantic find the indices and split them into train and test sets
# semantics = dataset["semantics"]
labels = dataset["labels"]
labels = np.array(labels)
test_classes_indices = np.where(is_found(test_classes)(labels))[0]
train_classes_indices = np.where(np.logical_not(is_found(test_classes)(labels)))[0]

np_dataset = np.array(dataset["dataset"])
train_split = np.where(is_label_found(train_classes_indices)(np_dataset))[0].tolist()
test_split = np.where(
    np.logical_not(is_label_found(train_classes_indices)(np_dataset))
)[0].tolist()

split = {"train": train_split, "val": test_split}

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
