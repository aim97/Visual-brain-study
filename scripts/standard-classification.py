"""Simplified EEG Signal Classification Execution Script"""
import os
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

from eeg_visual_classification.utils.lib import (
    create_parser, extract_model_options, get_dataloaders, get_model_hash, load_checkpoint
)
from ..models import MODEL_REGISTRY
from eeg_visual_classification.utils.lib import train_model_pipeline # Import the new pipeline

torch.utils.backcompat.broadcast_warning.enabled = True
cudnn.benchmark = True

PKG_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.getenv("EEG_MODELS_DIR", PKG_ROOT / "stored models"))

# Parse and configure
opt = create_parser().parse_args()
DATASET_TYPE = opt.eeg_dataset.split("/")[-1].split(".")[0]
model_options = extract_model_options(opt.model_params)
model_options["n_classes"] = 40 # Configuration specific to this file

MODEL_HASH = get_model_hash(opt.model_type, model_options)
SAVING_PATH = os.path.join(MODELS_DIR, f"{opt.model_type}_{MODEL_HASH}_{opt.experiment_name}/{DATASET_TYPE}")

os.makedirs(SAVING_PATH, exist_ok=True)

# Model and Optimizer setup
ModelClass = MODEL_REGISTRY.get(opt.model_type)
if ModelClass is None:
    raise ValueError(f"Model {opt.model_type} not found.")

model = ModelClass(**model_options)
if not opt.no_cuda:
    model.cuda()

optimizer = getattr(torch.optim, opt.optim)(model.parameters(), lr=opt.learning_rate)

if opt.pretrained_net:
    model, _ = load_checkpoint(opt.pretrained_net, device="cuda" if not opt.no_cuda else "cpu")

# Dataloader setup
dataset_options = {
    "eeg_dataset": os.path.join(PKG_ROOT, opt.eeg_dataset),
    "subject": opt.subject,
    "time_low": opt.time_low,
    "time_high": opt.time_high,
    "splits_path": os.path.join(PKG_ROOT, opt.splits_path),
    "split_num": opt.split_num,
    "batch_size": opt.batch_size,
    "model_type": opt.model_type,
}
loaders = get_dataloaders(dataset_options)

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
    dataset_options=dataset_options
)

print(f"Training completed. Best model saved at epoch {best_epoch}.")
