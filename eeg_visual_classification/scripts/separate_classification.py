"""Unseen-class EEG classification & SVM evaluation. Trains on 36 classes
(train/val), holds out 4 classes entirely, then probes whether the trained
encoder's representation of those unseen classes is linearly separable via
an SVM + t-SNE on the extracted features (forward_features)."""
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from eeg_visual_classification.utils.lib import (
    create_parser,
    extract_model_options,
    get_dataloaders,
    get_model_hash,
    load_checkpoint,
    set_seed,
    get_device,
    train_svm_rbf,
    evaluate_clf,
    plot_tsne,
)
from eeg_visual_classification.utils.training import train_model_pipeline
from eeg_visual_classification.models import instantiate_model

torch.utils.backcompat.broadcast_warning.enabled = True
cudnn.benchmark = True

# The dataset's first 4 classes (label indices 0-3) are the held-out, unseen
# ones; the model is trained on the remaining 36 (indices 4-39). We shift
# targets down by this offset so the 36-way classifier head sees labels 0-35.
UNSEEN_CLASS_OFFSET = 4
N_TRAIN_CLASSES = 36

PKG_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.getenv("EEG_MODELS_DIR", PKG_ROOT / "stored models"))

# Parse arguments
opt = create_parser(
    split_path_default="resources/unseenclasses_path.pth"
).parse_args()
set_seed(opt.seed)

DATASET_TYPE = Path(opt.eeg_dataset).stem
model_options = extract_model_options(opt.model_params)
model_options["n_classes"] = N_TRAIN_CLASSES

MODEL_HASH = get_model_hash(opt.model_type, model_options)
SAVING_PATH = (
    MODELS_DIR
    / f"{opt.model_type}_{MODEL_HASH}_{opt.experiment_name}_seed{opt.seed}"
    / DATASET_TYPE
)
SAVING_PATH.mkdir(parents=True, exist_ok=True)

# Model and Optimizer setup
model = instantiate_model(opt.model_type, **model_options)
device = get_device(opt.no_cuda)
model.to(device)

optimizer = getattr(torch.optim, opt.optim)(model.parameters(), lr=opt.learning_rate)

if opt.pretrained_net:
    model, _ = load_checkpoint(opt.pretrained_net, device=device)

# Load data for Train, Val, and Test
dataset_options = {
    "eeg_dataset": str(PKG_ROOT / opt.eeg_dataset),
    "subject": opt.subject,
    "time_low": opt.time_low,
    "time_high": opt.time_high,
    "splits_path": str(PKG_ROOT / opt.splits_path),
    "split_num": opt.split_num,
    "batch_size": opt.batch_size,
    "model_type": opt.model_type,
    "combined_test_val": False,
    "return_full_record": True,
}

all_loaders = get_dataloaders(dataset_options, splits=["train", "val", "test"])
train_val_loaders = {"train": all_loaders["train"], "val": all_loaders["val"]}

# Train on the 36 known classes only (targets shifted down by the offset).
trained_model, best_epoch = train_model_pipeline(
    model=model,
    optimizer=optimizer,
    loaders=train_val_loaders,
    opt=opt,
    n_classes=model_options["n_classes"],
    saving_path=SAVING_PATH,
    model_hash=MODEL_HASH,
    model_options=model_options,
    dataset_options=dataset_options,
    label_offset=UNSEEN_CLASS_OFFSET,
)

print(
    f"Training phase complete (Best Epoch: {best_epoch}). "
    "Commencing SVM feature evaluation on the unseen classes..."
)

# =====================================================================
# Post-Training Evaluation: SVM & t-SNE on the held-out unseen classes
# =====================================================================
trained_model.eval()
torch.set_grad_enabled(False)
test_set = []

for input_eeg, target, _, subjects in all_loaders["test"]:
    input_eeg = input_eeg.to(device)
    target = target.to(device) - UNSEEN_CLASS_OFFSET

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

train_features, train_labels = (
    test_set_features[:split_idx],
    test_set_labels[:split_idx],
)
test_features, test_labels = (
    test_set_features[split_idx:],
    test_set_labels[split_idx:],
)

clf = train_svm_rbf(np.array(train_features), np.array(train_labels))
evaluate_clf(clf, test_features, test_labels, ["0", "1", "2", "3"])

# Display t-SNE image for the generated representations (class/subject colored)
plot_labels = np.array(
    [f"Class-{lbl}/Subj-{subj}" for lbl, subj in zip(test_set_labels, test_set_subjects)]
)
labels_set = list(set(plot_labels))

plot_tsne(test_set_features, plot_labels, labels_set, "t-SNE")
print("Evaluation and t-SNE plotting complete.")
