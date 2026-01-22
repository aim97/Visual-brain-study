"""some summary"""

import os
from pathlib import Path
import random

import torch
import pandas as pd
import numpy as np
from eeg_visual_classification.utils.lib import (
    create_parser,
    extract_model_options,
    get_dataloaders,
    get_model_hash,
    save_checkpoint,
    load_checkpoint,
    train_svm_rbf,
    evaluate_clf,
    plot_tsne,
)
from ..models import MODEL_REGISTRY

torch.utils.backcompat.broadcast_warning.enabled = True  # type: ignore

import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim

cudnn.benchmark = True


# definitions
PKG_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.getenv("EEG_MODELS_DIR", PKG_ROOT / "stored models"))
DATA_DIR = Path(os.getenv("EEG_DATA_DIR", PKG_ROOT / "data" / "block"))

# Parse arguments
opt = create_parser(
    split_path_default="./data/block/unseenclasses_path.pth"
).parse_args()
DATASET_TYPE = opt.eeg_dataset.split("/")[3].split(".")[0]
model_options = extract_model_options(opt.model_params)
MODEL_HASH = get_model_hash(opt.model_type, model_options)
SAVING_PATH = os.path.join(
    MODELS_DIR,
    f"{opt.model_type}_{MODEL_HASH}_{opt.experiment_name}/{DATASET_TYPE}",
)
print(opt)
if not os.path.exists(SAVING_PATH):
    os.makedirs(SAVING_PATH)
else:
    exit("Experiment name already exists. Choose another name.")

# Create model
# module = importlib.import_module("models." + opt.model_type)
# model = module.Model(**model_options)
ModelClass = MODEL_REGISTRY.get(opt.model_type)
if ModelClass is None:
    raise ValueError(f"Model {opt.model_type} not found in MODEL_REGISTRY.")
model_options["n_classes"] = 36
model = ModelClass(**model_options)
optimizer = getattr(torch.optim, opt.optim)(model.parameters(), lr=opt.learning_rate)

# Setup CUDA
if not opt.no_cuda:
    model.cuda()
    print("Copied to CUDA")

if opt.pretrained_net != "":
    # model = torch.load(opt.pretrained_net)
    model, _ = load_checkpoint(
        opt.pretrained_net,
        device="cuda",
    )
    print(model)

# Load data
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
loaders = get_dataloaders(dataset_options, splits=["train", "val", "test"])
test_loaders = {"test": loaders["test"]}
loaders = {"train": loaders["train"], "val": loaders["val"]}

# initialize training,validation, test losses and accuracy list
losses_per_epoch = {split: [] for split in loaders}
# {"train": [], "val": [], "test": []}
accuracies_per_epoch = {split: [] for split in loaders}
# {"train": [], "val": [], "test": []}

best_accuracy = 0
best_accuracy_val = 0
best_epoch = 0
# Start training

predicted_labels = []
correct_labels = []

for epoch in range(1, opt.epochs + 1):
    # Initialize loss/accuracy variables
    losses = {split: 0.0 for split in loaders}
    # {"train": 0.0, "val": 0.0, "test": 0.0}
    accuracies = {split: 0.0 for split in loaders}
    # {"train": 0, "val": 0, "test": 0}
    counts = {split: 0 for split in loaders}
    # {"train": 0, "val": 0, "test": 0}
    # Adjust learning rate for SGD
    if epoch % opt.learning_rate_decay_every == 0:
        lr = max(
            opt.learning_rate
            * (opt.learning_rate_decay_by ** (epoch // opt.learning_rate_decay_every)),
            opt.learning_rate_decay_limit,
        )
        print(f"Learning rate dropped to : {lr}")
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    # Process each split
    class_list = list(range(40))
    conf_mat = pd.DataFrame(0, index=class_list, columns=class_list, dtype=float)
    for split, loader in loaders.items():
        # Set network mode
        if split == "train":
            model.train()
            torch.set_grad_enabled(True)
        else:
            model.eval()
            torch.set_grad_enabled(False)
        # Process all split batches
        for i, (input_eeg, target) in enumerate(loader):
            # Check CUDA
            if not opt.no_cuda:
                input_eeg = input_eeg.to("cuda")
                target = target.to("cuda") - 4
            # Forward
            output = model(input_eeg)

            # Compute loss
            loss = F.cross_entropy(output, target)
            losses[split] += loss.item()
            # Compute accuracy
            _, pred = output.data.max(1)
            correct = pred.eq(target.data).sum().item()
            accuracy = correct / input_eeg.data.size(0)
            accuracies[split] += accuracy
            counts[split] += 1

            # compute confusion matrix
            pred_list = pred.cpu().tolist()
            target_list = target.cpu().tolist()
            for i, j in zip(target_list, pred_list):
                conf_mat.loc[i, j] += 1

            # print(pred)
            # print(target.data)
            # Backward and optimize
            if split == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # Print info at the end of the epoch
    if accuracies["val"] / counts["val"] >= best_accuracy_val:
        best_accuracy_val = accuracies["val"] / counts["val"]
        best_epoch = epoch

    TrL = losses["train"] / counts["train"]
    TrA = accuracies["train"] / counts["train"]
    VL = losses["val"] / counts["val"]
    VA = accuracies["val"] / counts["val"]
    conf_mat = conf_mat.div(conf_mat.sum(axis=1), axis=0)

    print(
        f"Model: {opt.model_type} - Subject {opt.subject} - Time interval: [{opt.time_low}-{opt.time_high}] "
        f"- Epoch {epoch}: TrL={TrL:.4f}, TrA={TrA:.4f}, VL={VL:.4f}, VA={VA:.4f}, "
        f"max VA = {best_accuracy_val:.4f} at epoch {best_epoch:d}"
    )

    if len(accuracies_per_epoch["val"]) == 0 or VA > max(accuracies_per_epoch["val"]):
        save_checkpoint(
            model,
            optimizer,
            epoch,
            SAVING_PATH,
            opt,
            MODEL_HASH,
            model_options,
            dataset_options,
            losses_per_epoch,
            accuracies_per_epoch,
            TrL,
            TrA,
            VL,
            VA,
            VL,
            VA,
            conf_mat,
        )

    losses_per_epoch["train"].append(TrL)
    losses_per_epoch["val"].append(VL)
    accuracies_per_epoch["train"].append(TrA)
    accuracies_per_epoch["val"].append(VA)

    print("test")
    if best_epoch != epoch or VA < 0.66:
        continue
    # evaluate eeg representations for test set and map new representations to class labels
    test_set_options = dataset_options.copy()
    test_set_options["batch_size"] = 128
    # test_set_loader = get_dataloaders(test_set_options, splits=["test"])["test"]
    model.eval()
    torch.set_grad_enabled(False)
    test_set = []
    for i, (input_eeg, target, _, subjects) in enumerate(test_loaders["test"]):
        if not opt.no_cuda:
            input_eeg = input_eeg.to("cuda")
            target = target.to("cuda")
        # Forward
        output: torch.Tensor = model.forward_features(input_eeg)

        test_set.extend(
            list(zip(output.cpu().tolist(), target.data.cpu().tolist(), subjects))
        )

    # train an SVM classifier with rbf kernel on the new dataset
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

    plot_labels = np.array(
        [
            f"Class-{lbl}/Subj-{subj}"
            for lbl, subj in zip(test_set_labels, test_set_subjects)
        ]
    )
    labels_set = list(set(plot_labels))
    # display t-SNE image for the generated representations (class colored)
    # plot_tsne(test_set_features, test_set_labels, ["0", "1", "2", "3"], "t-SNE")
    plot_tsne(test_set_features, plot_labels, labels_set, "t-SNE")
