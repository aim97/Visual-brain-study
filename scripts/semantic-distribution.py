# compate the distribution of the semantic labels in the training set and the test set
import torch
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Template")

parser.add_argument(
    "-id",
    "--input-dataset",
    help="input EEG dataset path",
)

parser.add_argument(
    "-sp1",
    "--semantic_split-path",
    help="splits path",
)

parser.add_argument(
    "-sp2",
    "--session_split-path",
    help="splits path",
)

opt = parser.parse_args()


data = torch.load(opt.input_dataset)
split1 = torch.load(opt.semantic_split_path)
split2 = torch.load(opt.session_split_path)

labels = data["semantics"]

semantic_train_labels = split1["train"]
semantic_test_labels = split1["val"]
session_train_labels = split2["train"]
session_test_labels = split2["val"]

# compute distribution of labels across the dataset
all_labels = np.array([x["semantic"] for x in data["dataset"]])
len_all = len(all_labels)


all_distribution = np.round(np.bincount(all_labels) * 100 / len_all, 2)

semantic_train_distribution = np.round(
    np.bincount(all_labels[semantic_train_labels]) * 100 / len(semantic_train_labels),
    2,
)

semantic_test_distribution = np.round(
    np.bincount(all_labels[semantic_test_labels]) * 100 / len(semantic_test_labels),
    2,
)

session_train_distribution = np.round(
    np.bincount(all_labels[session_train_labels]) * 100 / len(session_train_labels),
    2,
)

session_test_distribution = np.round(
    np.bincount(all_labels[session_test_labels]) * 100 / len(session_test_labels),
    2,
)

print("Semantic distribution: ")
print(all_distribution)
print("mean: ", np.mean(all_distribution), "std: ", np.std(all_distribution))

print("Semantic train distribution: ")
print(semantic_train_distribution)
print(
    "mean: ",
    np.mean(semantic_train_distribution),
    "std: ",
    np.std(semantic_train_distribution),
)

print("Semantic test distribution: ")
print(semantic_test_distribution)
print(
    "mean: ",
    np.mean(semantic_test_distribution),
    "std: ",
    np.std(semantic_test_distribution),
)


print("Session train distribution: ")
print(session_train_distribution)
print(
    "mean: ",
    np.mean(session_train_distribution),
    "std: ",
    np.std(session_train_distribution),
)

print("Session test distribution: ")
print(session_test_distribution)
print(
    "mean: ",
    np.mean(session_test_distribution),
    "std: ",
    np.std(session_test_distribution),
)
