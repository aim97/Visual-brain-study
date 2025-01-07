"""Train visual2brain
"""

import torch
import torch.nn.functional as F
import numpy as np

from models.regressor import RegressorV1
from Training.fit import fit_v2


# Load EEG signals
loaded = torch.load("./data/processed/visual-to-brain-features.pth")

visual_features = torch.tensor([record["visual_features"] for record in loaded])
brain_features = torch.tensor(
    np.array([record["brain_features"].detach().numpy() for record in loaded])
).squeeze()

sz = len(brain_features)
split_idx = int(sz * 0.8)

# split dataset
perm = torch.randperm(sz)
train_idx = perm[:split_idx]
test_idx = perm[split_idx:]

visual_features_train = visual_features[train_idx].cuda()
visual_features_test = visual_features[test_idx].cuda()
brain_features_train = brain_features[train_idx].cuda()
brain_features_test = brain_features[test_idx].cuda()

model = RegressorV1(768, 420).cuda()

fit_v2(
    model,
    visual_features_train,
    visual_features_test,
    brain_features_train,
    brain_features_test,
    F.mse_loss,
    epochs=300,
    mode="regression",
)


# Save regressed features
regressed_features = model(visual_features.cuda())

# save the regressed features along with other features


data = [
    {**record, "regressed_features": regressed_features[i].cpu().detach().numpy()}
    for i, record in enumerate(loaded)
]


torch.save(
    data, "./data/processed/visual-to-brain-features_with_regressed_features.pth"
)
# Save the trained model
torch.save(
    model.cpu().state_dict(), "./build/Visual2brainRegressor/visual2brain_model.pth"
)

print("Training completed and model saved.")
