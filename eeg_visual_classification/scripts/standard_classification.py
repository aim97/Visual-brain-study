"""Standard-split EEG visual classification: 70/15/15 train/val/test over all
40 classes and all subjects. Reproduces Table 2/3-style headline results."""
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
MODELS_DIR = Path(os.getenv("EEG_MODELS_DIR", PKG_ROOT / "stored models"))

# Parse and configure
opt = create_parser().parse_args()
set_seed(opt.seed)

DATASET_TYPE = Path(opt.eeg_dataset).stem
model_options = extract_model_options(opt.model_params)
model_options["n_classes"] = 40  # Configuration specific to this file

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
    dataset_options=dataset_options,
)

print(f"Training completed. Best model saved at epoch {best_epoch}.")
