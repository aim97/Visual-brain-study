import torch

# from torch import nn
import numpy as np
from matplotlib import pyplot as plt

# import torch.nn.functional as F
# from models.VisualOnlyClassifier import SimpleClassifier
# from Training.fit import fit

# import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean

# Generate two high-dimensional datasets
# read data
data = torch.load(
    "./data/processed/visual-to-brain-features_with_regressed_features.pth"
)

brain_features = np.array(
    [record["brain_features"].detach().numpy() for record in data]
).squeeze()


regressed_features = np.array([record["regressed_features"] for record in data])

data1 = brain_features
data2 = regressed_features

print(data1.shape)
print(data2.shape)
# Apply PCA to reduce the dimensionality to 2D
pca1 = PCA(n_components=2)
data1_pca = pca1.fit_transform(data1)
sz = 100
x1 = data1_pca[:sz, 0]
y1 = data1_pca[:sz, 1]

pca2 = PCA(n_components=2)
data2_pca = pca2.fit_transform(data2)

x2 = data2_pca[:sz, 0]
y2 = data2_pca[:sz, 1]

print(data1_pca.shape)
print(data2_pca.shape)

print(x1.shape)
print(y1.shape)
print(x2.shape)
print(y2.shape)

multiplier = 1

plt.scatter(x1[:1], y1[:1], c="red")
plt.scatter(x1[1:], y1[1:], c="cyan")
plt.scatter(multiplier * x2[1:], multiplier * y2[1:], c="blue")
plt.scatter(multiplier * x2[:1], multiplier * y2[:1], c="brown")
plt.show()

# Calculate the Euclidean distance between the means of the two distributions
mean1 = np.mean(data1_pca, axis=0)
mean2 = np.mean(data2_pca, axis=0)
distance = euclidean(mean1, mean2)

print(f"Euclidean distance between the means of the two distributions: {distance}")
