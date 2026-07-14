import os
import torch
import pandas as pd
import torch.nn.functional as F
from ..utils.evaluation_metrics import update_per_subject_stats, calculate_subject_accuracy # From our previous refactor

def train_model_pipeline(
    model,
    optimizer,
    loaders,
    opt,
    n_classes,
    saving_path,
    model_hash,
    model_options,
    dataset_options
):
    """
    A generalized training and evaluation loop.
    Adapts dynamically to the splits provided in 'loaders' (e.g., train/val or train/val/test).
    """
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
                opt.learning_rate * (opt.learning_rate_decay_by ** (epoch // opt.learning_rate_decay_every)),
                opt.learning_rate_decay_limit,
            )
            print(f"Learning rate dropped to: {lr}")
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        # Initialize tracking matrices
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
                # Handle varying batch returns dynamically
                if len(batch) == 3:
                    input_eeg, target, subject = batch
                elif len(batch) == 4:
                    # Specific to eeg_separate_classification.py
                    input_eeg, target, _, subject = batch
                else:
                    input_eeg, target = batch
                    subject = torch.full_like(target, -1)

                if not opt.no_cuda:
                    input_eeg = input_eeg.to("cuda")
                    # Adjust target offset dynamically if needed (e.g., target - 4 for separate classification)
                    # Note: It's better to handle offset logic in the Dataset class to keep the loop clean
                    target = target.to("cuda")

                output = model(input_eeg)
                loss = F.cross_entropy(output, target)
                losses[split] += loss.item()

                _, pred = output.data.max(1)
                correct = pred.eq(target.data).sum().item()
                accuracies[split] += (correct / input_eeg.data.size(0))
                counts[split] += 1

                # Backward and optimize
                if split == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    # Track metrics strictly on validation/test phases
                    pred_list = pred.cpu().tolist()
                    target_list = target.cpu().tolist()
                    subject_list = subject.cpu().tolist()

                    for t_idx, p_idx in zip(target_list, pred_list):
                        conf_mat.loc[t_idx, p_idx] += 1

                    per_subject_counts = update_per_subject_stats(
                        per_subject_counts, subject_list, target_list, pred_list
                    )

        # Average metrics
        avg_loss = {split: losses[split] / counts[split] for split in split_names}
        avg_acc = {split: accuracies[split] / counts[split] for split in split_names}

        # Checkpointing logic
        VA = avg_acc.get("val", 0)
        if VA >= best_accuracy_val:
            best_accuracy_val = VA
            best_epoch = epoch
            dataset_options["subject_validation_acc"] = calculate_subject_accuracy(per_subject_counts)

            # Requires save_checkpoint to be updated to accept dynamic splits dict
            save_checkpoint(
                model, optimizer, epoch, saving_path, opt, model_hash,
                model_options, dataset_options, losses_per_epoch, accuracies_per_epoch,
                avg_loss.get("train", 0), avg_acc.get("train", 0),
                avg_loss.get("val", 0), avg_acc.get("val", 0),
                avg_loss.get("test", 0), avg_acc.get("test", 0),
                conf_mat
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

        if epoch - best_epoch >= 60:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    return model, best_epoch
