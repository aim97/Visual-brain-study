import torch
import pandas as pd
import torch.nn.functional as F

from .evaluation_metrics import update_per_subject_stats, calculate_subject_accuracy
from .lib import save_checkpoint, get_device


def _unpack_batch(batch):
    """Splitter/ValidationOnlySplitter yield (eeg, label), (eeg, label, subject),
    or (eeg, label, image, subject) depending on dataset options. Normalize all
    three shapes to (input_eeg, target, subject)."""
    if len(batch) == 4:
        input_eeg, target, _, subject = batch
    elif len(batch) == 3:
        input_eeg, target, subject = batch
    else:
        input_eeg, target = batch
        subject = torch.full_like(target, -1)
    return input_eeg, target, subject


def train_model_pipeline(
    model,
    optimizer,
    loaders,
    opt,
    n_classes,
    saving_path,
    model_hash,
    model_options,
    dataset_options,
    label_offset=0,
    early_stop_patience=60,
):
    """A generalized training and evaluation loop shared by every experiment script.

    Adapts dynamically to the splits provided in `loaders` (e.g. train/val or
    train/val/test). `label_offset` subtracts a constant from every target
    before the loss/accuracy/confusion-matrix computation - used by the
    unseen-class protocol, where the held-out classes occupy the first four
    label indices and the model is only trained to output the remaining
    `n_classes` classes.
    """
    device = get_device(opt.no_cuda)
    model.to(device)

    if hasattr(opt, "seed"):
        dataset_options["seed"] = opt.seed

    split_names = list(loaders.keys())

    losses_per_epoch = {split: [] for split in split_names}
    accuracies_per_epoch = {split: [] for split in split_names}

    best_accuracy_val = 0
    best_epoch = 0

    for epoch in range(1, opt.epochs + 1):
        losses = {split: 0.0 for split in split_names}
        accuracies = {split: 0.0 for split in split_names}
        counts = {split: 0 for split in split_names}

        # Adjust learning rate
        if epoch % opt.learning_rate_decay_every == 0:
            lr = max(
                opt.learning_rate
                * (opt.learning_rate_decay_by ** (epoch // opt.learning_rate_decay_every)),
                opt.learning_rate_decay_limit,
            )
            print(f"Learning rate dropped to: {lr}")
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        # Tracking matrices, computed over the non-train splits only
        class_list = list(range(n_classes))
        conf_mat = pd.DataFrame(0, index=class_list, columns=class_list, dtype=int)
        per_subject_counts = {}

        for split in split_names:
            if split == "train":
                model.train()
                torch.set_grad_enabled(True)
            else:
                model.eval()
                torch.set_grad_enabled(False)

            for batch in loaders[split]:
                input_eeg, target, subject = _unpack_batch(batch)

                input_eeg = input_eeg.to(device)
                target = target.to(device) - label_offset

                output = model(input_eeg)
                loss = F.cross_entropy(output, target)
                losses[split] += loss.item()

                _, pred = output.data.max(1)
                correct = pred.eq(target.data).sum().item()
                accuracies[split] += correct / input_eeg.data.size(0)
                counts[split] += 1

                if split == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    pred_list = pred.cpu().tolist()
                    target_list = target.cpu().tolist()
                    subject_list = subject.cpu().tolist()

                    for t_idx, p_idx in zip(target_list, pred_list):
                        conf_mat.loc[t_idx, p_idx] += 1

                    per_subject_counts = update_per_subject_stats(
                        per_subject_counts, subject_list, target_list, pred_list
                    )

        avg_loss = {split: losses[split] / counts[split] for split in split_names}
        avg_acc = {split: accuracies[split] / counts[split] for split in split_names}

        VA = avg_acc.get("val", 0)
        if VA >= best_accuracy_val:
            best_accuracy_val = VA
            best_epoch = epoch
            dataset_options["subject_validation_acc"] = calculate_subject_accuracy(
                per_subject_counts
            )

            save_checkpoint(
                model,
                optimizer,
                epoch,
                saving_path,
                opt,
                model_hash,
                model_options,
                dataset_options,
                losses_per_epoch,
                accuracies_per_epoch,
                avg_loss.get("train", 0),
                avg_acc.get("train", 0),
                avg_loss.get("val", 0),
                avg_acc.get("val", 0),
                avg_loss.get("test", 0),
                avg_acc.get("test", 0),
                conf_mat,
            )

        for split in split_names:
            losses_per_epoch[split].append(avg_loss[split])
            accuracies_per_epoch[split].append(avg_acc[split])

        print(
            f"Model: {opt.model_type} - Epoch {epoch}: "
            f"TrL={avg_loss.get('train', 0):.4f}, TrA={avg_acc.get('train', 0):.4f}, "
            f"VL={avg_loss.get('val', 0):.4f}, VA={avg_acc.get('val', 0):.4f} "
            f"(Best VA={best_accuracy_val:.4f} at epoch {best_epoch})"
        )

        if epoch - best_epoch >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    return model, best_epoch
