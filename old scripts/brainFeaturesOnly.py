# read the visual to brain features file
import torch
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from Training.fit import fit
from models.VisualOnlyClassifier import VisualOnlyClassifier

visual_to_brain_features = torch.load("./data/visual-to-brain-features.pth")

# print(visual_to_brain_features[0])

# split the data into features and labels
brain_features = [
    record["brain_features"].detach() for record in visual_to_brain_features
]
labels = [record["label"] for record in visual_to_brain_features]

print(type(brain_features[0]))

X_train, X_test, y_train, y_test = train_test_split(
    brain_features, labels, test_size=0.2, random_state=42
)

X_train = torch.cat(X_train, dim=0)
X_test = torch.cat(X_test, dim=0)
# print(len(X_train), len(X_train[0]), len(X_train[0][0]))
# to tensors
# X_train = torch.tensor(X_train).squeeze()
# X_test = torch.tensor(X_test, dtype=torch.float32).squeeze()
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

print(len(X_train), X_train[0].shape)
print(len(X_test), X_test[0].shape)
print(len(y_train), y_train[0].shape)
print(len(y_test), y_test[0].shape)

# create model
model = VisualOnlyClassifier(420)

# train model
fit(model, X_train, X_test, y_train, y_test)

# evaluate model'
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
