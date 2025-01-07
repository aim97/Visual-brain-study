# read the visual to brain features file
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from Training.fit import fit
from models.VisualOnlyClassifier import VisualOnlyClassifier

visual_to_brain_features = torch.load("./data/visual-to-brain-features.pth")

# print(visual_to_brain_features[0])

# split the data into features and labels
visual_features = [record["visual_features"] for record in visual_to_brain_features]
labels = [record["label"] for record in visual_to_brain_features]

X_train, X_test, y_train, y_test = train_test_split(
    visual_features, labels, test_size=0.2, random_state=42
)

# to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# create model
model = VisualOnlyClassifier()

# train model
fit(model, X_train, X_test, y_train, y_test, F.cross_entropy)

# evaluate model'
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
