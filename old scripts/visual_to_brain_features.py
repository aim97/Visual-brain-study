"""Visual to brain features
Builds a map from visual features to brain features
"""

import torch

# read the ./image_features_mapping file
image_features_mapping = torch.load(
    "./data/processed/image_features_mapping_compressed.pth"
)

# read image brain features
brain_feature_image_mapping = torch.load(
    "./data/processed/brainFeature-image-mapping.pth"
)

brain_features = brain_feature_image_mapping["features"]
images = brain_feature_image_mapping["images"]
labels = brain_feature_image_mapping["labels"]

dataset = []
found = 0
not_found = 0

print("Total number of images: " + str(len(images)))

for i, image_name in enumerate(images):
    if image_name in image_features_mapping:
        visual_features = image_features_mapping[image_name]
        brain_features = brain_feature_image_mapping["features"][i]
        label = labels[i]
        dataset.append(
            {
                "image_name": image_name,
                "visual_features": visual_features,
                "brain_features": brain_features,
                "label": label,
            }
        )
        found += 1
    else:
        print(f"No mapping found for {image_name}")
        not_found += 1

print(f"Found {found} mappings, not found mappings {not_found}")

# save the dataset to a .pth file
torch.save(dataset, "./data/processed/visual-to-brain-features.pth")

print("Dataset saved to visual-to-brain-features.pth")
