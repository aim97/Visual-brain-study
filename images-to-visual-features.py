from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the feature extractor and model
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)


# Function to load and preprocess images
def load_image(images_path):
    image = Image.open(images_path).convert("RGB")
    return feature_extractor(images=image, return_tensors="pt")


# Directory containing your images
IMAGE_DIR = "D:/aim97/lab/dataset/stimuli/image"

# Dictionary to store the mapping
image_features_mapping = {}
# Extract features for each image
for image_name in os.listdir(IMAGE_DIR):
    if not image_name.endswith(".JPEG"):
        continue
    image_path = os.path.join(IMAGE_DIR, image_name)
    inputs = load_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # features = outputs.last_hidden_state.squeeze().cpu().tolist()
        features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().tolist()
        image_features_mapping[image_name] = features
        # print(f"Image: {image_name}, Features: {len(features)}")

# Save the mapping to a .pth file
torch.save(image_features_mapping, "image_features_mapping_compressed.pth")

print("Mapping saved to image_features_mapping_compressed.pth")
