import torch
import os
from glob import glob
from pathlib import Path
import argparse

PKG_ROOT = Path(__file__).resolve().parents[1]


def load_class_image_list(path):
    class_name = os.path.basename(path).split(".")[0]
    with open(path, "r") as f:
        lines = f.readlines()
    lines = [line.strip().split(".")[0] for line in lines]
    return (class_name, lines)


def get_semantic(semantic_images, image):
    for semantic, images in semantic_images.items():
        if image in images:
            return semantic
    return None


def get_images_classes(data):
    dict = {}
    for class_name, image_name in data:
        if image_name not in dict:
            dict[image_name] = [class_name]
        else:
            dict[image_name].append(class_name)
    return dict


def replace_image_with_index(image_names, images):
    return [images.index(image_name) for image_name in image_names]


parser = argparse.ArgumentParser(description="Template")

parser.add_argument(
    "-id",
    "--input-dataset",
    help="input EEG dataset path",
)

parser.add_argument(
    "-od",
    "--output-dataset",
    help="output EEG dataset path",
)

opt = parser.parse_args()

dataset_path = opt.input_dataset
output_dataset_path = opt.output_dataset
SEMANTIC_IMAGES_PATH = str(PKG_ROOT / "resources" / "SemanticSplits" / "*")

dataset = torch.load(dataset_path)

semantic_splits = glob(SEMANTIC_IMAGES_PATH)
print(f"Found {len(semantic_splits)} semantic splits.")
semantic_images = [load_class_image_list(path) for path in semantic_splits]
all_images = []
for _, images in semantic_images:
    all_images.extend(images)

semantic_images = {semantic: images for (semantic, images) in semantic_images}

output_dataset = {
    "labels": dataset["labels"],
    "semantics": list(semantic_images.keys()),
    "images": all_images,  # dataset["images"],
    "dataset": [],
}

print_cnt = 3

for sample in dataset["dataset"]:
    image_idx = sample["image"]
    image = dataset["images"][image_idx]
    semantic = get_semantic(semantic_images, image)
    if semantic is None:
        continue
    semantic = output_dataset["semantics"].index(semantic)
    sample["semantic"] = semantic
    sample["image"] = all_images.index(image)
    output_dataset["dataset"].append(sample)
    if print_cnt > 0:
        print(sample)
        print_cnt -= 1

torch.save(output_dataset, output_dataset_path)

print("Labels#: ", len(output_dataset["labels"]))
print("Semantics#: ", len(output_dataset["semantics"]))
print("images#: ", len(output_dataset["images"]))
print("Dataset size: ", len(output_dataset["dataset"]))
