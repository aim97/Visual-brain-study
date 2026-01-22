"""This script splits the dataset such that
A selected set of class is separated as a test set while the remaining is the training and val set
"""

import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser(description="Template")

parser.add_argument(
    "-id",
    "--input-dataset",
    required=True,
    help="input EEG dataset path",
)

parser.add_argument(
    "-sp",
    "--output-splits",
    required=True,
    help="output EEG dataset path",
)

opt = parser.parse_args()

test_classes = [0, 1, 2, 3]


dataset_path = opt.input_dataset
splits_path = opt.output_splits

loaded = torch.load(dataset_path)
dataset = np.array(loaded["dataset"])

image_indices = list(range(50))
split_idx = int(len(image_indices) * 0.8)
train_image_indices, val_image_indices = (
    image_indices[:split_idx],
    image_indices[split_idx:],
)

all_labels = np.array([x["label"] for x in dataset])
test_set_indices = np.where(np.isin(all_labels, list(test_classes)))[0].tolist()

label_images = np.array([(x["label"], x["image"]) for x in dataset])

print(label_images)

class_images = {}
for i in set(range(40)).difference(test_classes):
    mask = label_images[:, 0] == i
    class_images[i] = np.unique(label_images[mask, 1]).tolist()


class_splits = {
    i: (
        class_images[i][: int(len(class_images[i]) * 0.8)],
        class_images[i][int(len(class_images[i]) * 0.8) :],
    )
    for i in class_images
}

train_images = []
val_images = []
for i in class_splits:
    train_images.extend(class_splits[i][0])
    val_images.extend(class_splits[i][1])


train_set_indices = np.where(
    np.vectorize(lambda x: x["image"] in train_images)(dataset)
)[0].tolist()

val_set_indices = np.where(np.vectorize(lambda x: x["image"] in val_images)(dataset))[
    0
].tolist()

print(f"Total dataset size: {len(dataset)}")
print(f"Train set size: {len(train_set_indices)}")
print(f"Test set size: {len(test_set_indices)}")
print(f"val set size: {len(val_set_indices)}")

# assertions
# 0. items in the sets are unique
train_set = set(train_set_indices)
val_set = set(val_set_indices)
test_set = set(test_set_indices)

assert len(train_set) == len(train_set_indices), "Train set is not unique"
assert len(val_set) == len(val_set_indices), "Val set is not unique"
assert len(test_set) == len(test_set_indices), "Test set is not unique"

# 1. No intersection between datasets
train_test_intersected = len(train_set.intersection(test_set)) > 0
val_test_intersected = len(val_set.intersection(test_set)) > 0
train_val_intersected = len(train_set.intersection(val_set)) > 0

assert not train_test_intersected, "Train and test set contain common indices"
assert not val_test_intersected, "Val and test set contain common indices"
assert not train_val_intersected, "Train and Val set contain common indices"

# 2. No dropped samples
assert dataset.shape[0] == len(train_set_indices) + len(test_set_indices) + len(
    val_set_indices
), "Found some dropped samples"

output = {
    "splits": {
        0: {
            "train": train_set_indices,
            "val": val_set_indices,
            "test": test_set_indices,
        }
    }
}

torch.save(output, splits_path)
