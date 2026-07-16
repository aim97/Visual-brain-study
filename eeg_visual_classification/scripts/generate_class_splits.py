"""
Divide the dataset such that each subject has several droped sessions for certain classes that are available for other subjects

Subject   DropCount
0         6
1         7
2         7
3         7
4         7
5         6

"""

import random
import argparse
import torch
import numpy as np


drop_count = [6] * 6  # [6] + [7] * 4 + [6]
for i in range(1, len(drop_count)):
    drop_count[i] += drop_count[i - 1]

assert drop_count[-1] == 36, f"only {drop_count[-1]} classes are dropped"

subject_drop_list = {
    (i + 1): list(range(0 if i == 0 else drop_count[i - 1], drop_count[i]))
    for i in range(6)
}

print(subject_drop_list)


is_test = lambda sample: sample["label"] in subject_drop_list[sample["subject"]]


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

dataset = np.array(torch.load(opt.input_dataset)["dataset"])
test_set_indices = np.where(np.vectorize(is_test)(dataset))[0].tolist()

# assert (
#     2100 > len(test_set_indices) > 1800
# ), f"The test set size is less than expected, found {len(test_set_indices)}"

train_set_indices = list(set(range(dataset.shape[0])).difference(test_set_indices))

# assert (
#     len(train_set_indices) > len(test_set_indices) * 3
# ), "The training set should be at least 3 times larger than the test set"

print(f"Train set size: {len(train_set_indices)}")
print(f"Test  set size: {len(test_set_indices)}")

random.shuffle(train_set_indices)
random.shuffle(test_set_indices)

splits = {
    "splits": {
        0: {
            "train": train_set_indices,
            "val": test_set_indices,
            "test": test_set_indices,
        }
    }
}

torch.save(splits, opt.output_splits)
