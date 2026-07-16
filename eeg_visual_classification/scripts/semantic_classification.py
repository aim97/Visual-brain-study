"""Semantic-relabeling EEG classification: 10 broad semantic categories that
cut across the original 40 classes, used both for the direct relabeling
protocol (--splits-path .../semantic_splits.pth) and the session-based /
isolated-classes protocol (--splits-path .../session_splits.pth)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
)
from eeg_visual_classification.utils.training import train_model_pipeline
from eeg_visual_classification.models import instantiate_model

torch.utils.backcompat.broadcast_warning.enabled = True
cudnn.benchmark = True

PKG_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.getenv("EEG_MODELS_DIR", PKG_ROOT / "semantic_models"))

# Parse and configure
opt = create_parser(
    default_eeg_dataset="data/processed/semantic_eeg_5_95_std.pth",
    split_path_default="resources/SemanticSplits/semantic_splits.pth",
).parse_args()
set_seed(opt.seed)

DATASET_TYPE = Path(opt.eeg_dataset).stem
model_options = extract_model_options(opt.model_params)
model_options["n_classes"] = 10  # Specific to semantic classification

MODEL_HASH = get_model_hash(opt.model_type, model_options)

# Seed is part of the saving path so repeated runs never collide.
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

# Dataloader setup
dataset_options = {
    "eeg_dataset": str(PKG_ROOT / opt.eeg_dataset),
    "subject": opt.subject,
    "time_low": opt.time_low,
    "time_high": opt.time_high,
    "splits_path": str(PKG_ROOT / opt.splits_path),
    "split_num": opt.split_num,
    "batch_size": opt.batch_size,
    "model_type": opt.model_type,
    "is_semantic": True,
}
loaders = get_dataloaders(dataset_options, ["train", "val"])

# Execute generic training pipeline
trained_model, best_epoch = train_model_pipeline(
    model=model,
    optimizer=optimizer,
    loaders=loaders,
    opt=opt,
    n_classes=model_options["n_classes"],
    saving_path=SAVING_PATH,
    model_hash=MODEL_HASH,
    model_options=model_options,
    dataset_options=dataset_options,
)

print(f"Semantic training completed. Best model saved at epoch {best_epoch}.")
