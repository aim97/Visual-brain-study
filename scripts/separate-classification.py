"""EEG Separate Classification & SVM Evaluation Script"""
import os
import random
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

from eeg_visual_classification.utils.lib import (
    create_parser, extract_model_options, get_dataloaders, get_model_hash,
    load_checkpoint, train_svm_rbf, evaluate_clf, plot_tsne
)
from ..models import MODEL_REGISTRY
from .utils.lib import train_model_pipeline  # Import the shared pipeline

torch.utils.backcompat.broadcast_warning.enabled = True
cudnn.benchmark = True

PKG_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.getenv("EEG_MODELS_DIR", PKG_ROOT / "stored models"))

# Parse arguments
opt = create_parser(
    split_path_default="./data/block/unseenclasses_path.pth"
).parse_args()

DATASET_TYPE = Path(opt.eeg_dataset).stem  # Cleaner path extraction
model_options = extract_model_options(opt.model_params)
model_options["n_classes"] = 36  # Specific to separate classification

MODEL_HASH = get_model_hash(opt.model_type, model_options)
SAVING_PATH = os.path.join(
    MODELS_DIR, f"{opt.model_type}_{MODEL_HASH}_{opt.experiment_name}/{DATASET_TYPE}"
)
os.makedirs(SAVING_PATH, exist_ok=True)

# Model and Optimizer setup
ModelClass = MODEL_REGISTRY.get(opt.model_type)
if ModelClass is None:
    raise ValueError(f"Model {opt.model_type} not found in MODEL_REGISTRY.")

model = ModelClass(**model_options)
if not opt.no_cuda:
    model.cuda()

optimizer = getattr(torch.optim, opt.optim)(model.parameters(), lr=opt.learning_rate)

if opt.pretrained_net:
    model, _ = load_checkpoint(opt.pretrained_net, device="cuda" if not opt.no_cuda else "cpu")

# Load data for Train, Val, and Test
dataset_options = {
    "eeg_dataset": os.path.join(PKG_ROOT, opt.eeg_dataset),
    "subject": opt.subject,
    "time_low": opt.time_low,
    "time_high": opt.time_high,
    "splits_path": os.path.join(PKG_ROOT, opt.splits_path),
    "split_num": opt.split_num,
    "batch_size": opt.batch_size,
    "model_type": opt.model_type,
    "combined_test_val": False,
    "return_full_record": True,
}

all_loaders = get_dataloaders(dataset_options, splits=["train", "val", "test"])

# Pass only train and val to the training pipeline
train_val_loaders = {"train": all_loaders["train"], "val": all_loaders["val"]}

# Execute generic training pipeline
trained_model, best_epoch = train_model_pipeline(
    model=model,
    optimizer=optimizer,
    loaders=train_val_loaders,
    opt=opt,
    n_classes=model_options["n_classes"],
    saving_path=SAVING_PATH,
    model_hash=MODEL_HASH,
    model_options=model_options,
    dataset_options=dataset_options
)

print(f"Training phase complete (Best Epoch: {best_epoch}). Commencing SVM Feature Evaluation...")

# =====================================================================
# Post-Training Evaluation: SVM & t-SNE on Test Set (Unseen Classes)
# =====================================================================

trained_model.eval()
torch.set_grad_enabled(False)
test_set = []

# Assuming target adjustments (-4) are handled either here or in your Dataset class.
# Matching your original code's specific unpacking structure:
for i, (input_eeg, target, _, subjects) in enumerate(all_loaders["test"]):
    if not opt.no_cuda:
        input_eeg = input_eeg.to("cuda")
        target = target.to("cuda") - 4  # Offset for the 4 held-out classes

    # Extract representation features rather than raw predictions
    output = trained_model.forward_features(input_eeg)

    test_set.extend(
        list(zip(output.cpu().tolist(), target.data.cpu().tolist(), subjects))
    )

# Train an SVM classifier with RBF kernel on the new dataset representations
random.shuffle(test_set)
split_idx = int(len(test_set) * 0.8)

test_set_features = np.array([x[0] for x in test_set])
test_set_labels = np.array([x[1] for x in test_set])
test_set_subjects = np.array([x[2] for x in test_set])

train_features, train_labels = test_set_features[:split_idx], test_set_labels[:split_idx]
test_features, test_labels = test_set_features[split_idx:], test_set_labels[split_idx:]

clf = train_svm_rbf(np.array(train_features), np.array(train_labels))
evaluate_clf(clf, test_features, test_labels, ["0", "1", "2", "3"])

# Display t-SNE image for the generated representations (class/subject colored)
plot_labels = np.array([
    f"Class-{lbl}/Subj-{subj}" for lbl, subj in zip(test_set_labels, test_set_subjects)
])
labels_set = list(set(plot_labels))

plot_tsne(test_set_features, plot_labels, labels_set, "t-SNE")
print("Evaluation and t-SNE plotting complete.")
