import torch
from torch import nn
import numpy as np

import torch.nn.functional as F
from models.VisualOnlyClassifier import SimpleClassifier
from Training.fit import fit

# read data
data = torch.load(
    "./data/processed/visual-to-brain-features_with_regressed_features.pth"
)

brain_features = torch.tensor(
    np.array([record["brain_features"].detach().numpy() for record in data])
)

regressed_features = torch.tensor(
    np.array([record["regressed_features"] for record in data])
)

labels = torch.tensor(np.array([record["label"] for record in data]), dtype=torch.long)


# split data
train_size = int(0.8 * len(brain_features))
val_size = len(brain_features) - train_size

train_brain_features, val_brain_features = (
    brain_features[:train_size],
    brain_features[train_size:],
)
train_regressed_features, val_regressed_features = (
    regressed_features[:train_size],
    regressed_features[train_size:],
)
train_labels, val_labels = labels[:train_size], labels[train_size:]

# define models
fromBrain = SimpleClassifier([420, 256])
fromRegressed = SimpleClassifier([420, 256])

fit(
    # fromBrain,
    fromRegressed,
    # train_brain_features,
    # val_brain_features,
    train_regressed_features,
    val_regressed_features,
    train_labels,
    val_labels,
    F.cross_entropy,
    epochs=100,
    mode="classification",
)
