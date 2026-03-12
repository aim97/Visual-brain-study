import glob
import os
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from ..utils import EEGDataset, load_checkpoint, ValidationOnlySplitter, get_model_hash

# orig_torch_load = torch.load


# def torch_wrapper(*args, **kwargs):
#     logging.warning(
#         "[comfyui-unsafe-torch] I have unsafely patched `torch.load`.  The `weights_only` option of `torch.load` is forcibly disabled."
#     )
#     kwargs["weights_only"] = False

#     return orig_torch_load(*args, **kwargs)


# torch.load = torch_wrapper

# NODE_CLASS_MAPPINGS = {}
# __all__ = ["NODE_CLASS_MAPPINGS"]

PKG_ROOT = Path(__file__).resolve().parents[1]  # .../eeg_visual_classification
MODELS_DIR = PKG_ROOT / "stored models"
DATA_DIR = PKG_ROOT / "data" / "block"


def get_pretrained_model_paths(model_names, dataset_names):
    models = []
    for model_name in model_names:
        for dataset_name in dataset_names:
            model_path_pattern = (
                f"{MODELS_DIR}/{model_name}*/{dataset_name}/{model_name}_*.pth"
            )

            # find all files matching the pattern
            model_files = glob.glob(model_path_pattern)

            models.extend([(model_name, dataset_name, f) for f in model_files])
    return models


def get_conf_stats(conf_mat: pd.DataFrame, labels: list, k, eta):
    """This method compute the stats of a confusion matrix to reflect which classes
    are not learnt well by our model

    Args:
        conf (pd.DataFrame): dataframe for the confusion matrix
        labels (list): a list of actual labels names
    Returns:
        - A list of classes whose accuracy is higher than <high_limit>
        - A list of classes whose accuracy is lower than <low_limit>, and for each we want
            - The sorted miss classes
    """
    # fix col and row names
    # conf_mat.set_index(conf_mat.columns[0], inplace=True)
    conf_mat.index = conf_mat.index.map(lambda x: labels[int(x.split("_")[1])])
    conf_mat.columns = conf_mat.columns.map(lambda x: labels[int(x.split("_")[1])])

    # compute acc
    acc = pd.Series(
        {label: conf_mat.loc[label, label] for label in conf_mat.index}
    ).sort_values(ascending=False)
    high_acc_classes = acc.head(k)
    low_acc_classes = acc.tail(k)

    low_acc_classes_details = conf_mat.loc[low_acc_classes.index]
    low_acc_classes_details = {
        tc: low_acc_classes_details.loc[tc][low_acc_classes_details.loc[tc] > eta]
        for tc in low_acc_classes_details.index
    }

    return high_acc_classes, low_acc_classes, low_acc_classes_details


def get_model_path_by_hash(model_hash, dataset_names):
    model_path_pattern = f"{MODELS_DIR}/*{model_hash}*/*/*.pth"
    model_path_pattern = model_path_pattern.replace("\\", "\\/")

    # print(f"Searching for models with pattern: {model_path_pattern}")
    model_files = glob.glob(model_path_pattern)
    model_datasets = {}
    for dataset_name in dataset_names:
        model_datasets[dataset_name] = []
        for model_file in model_files:
            # print(f"Checking model file: {model_file}")
            if dataset_name in model_file:
                model_datasets[dataset_name].append(model_file)
    return model_datasets


def load_checkpoint_data(dataset_name, model_file):
    # load the checkpoint data
    model, checkpoint = load_checkpoint(model_file)
    ret = {
        "model": model,
        "checkpoint": checkpoint,
        "acc": checkpoint["metrics"]["val_accuracy"],
        "model_options": checkpoint["model_options"],
        "epoch": checkpoint["epoch"],
        "dataset_name": dataset_name,
        "model_file": model_file,
        "hash": checkpoint["model_hash"],
    }
    return ret


def find_best_models(dataset_models):
    best_models = {}
    for dataset_name in dataset_models:
        model_files = dataset_models[dataset_name]
        best_acc = -1.0
        best_model_data = None
        for model_file in model_files:
            model_data = load_checkpoint_data(dataset_name, model_file)
            acc = model_data["acc"]
            if acc > best_acc:
                best_acc = acc
                best_model_data = model_data
        if best_model_data is not None:
            best_models[dataset_name] = best_model_data

    return best_models


def get_models_coverage(model_names, datasets, models):
    v = {
        model_name: {dataset_name: 0 for dataset_name in datasets}
        for model_name in model_names
    }

    df = pd.DataFrame.from_dict(v, orient="index")
    for model_name, dataset_name, _ in models:
        df.at[model_name, dataset_name] = df.at[model_name, dataset_name] + 1

    return df


def evaluate_model(model, model_type, dataset_path):
    dataset = EEGDataset(
        {
            "eeg_dataset": dataset_path,
            "subject": 0,
            "time_low": 20,
            "time_high": 460,
            "model_type": model_type,
        }
    )
    dataset_loader = DataLoader(
        ValidationOnlySplitter(
            dataset,
            split_path=f"{DATA_DIR}/block_splits_by_image_all.pth",
            split_num=0,
        ),
        batch_size=32,
        drop_last=True,
        shuffle=True,
    )

    # load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    per_subject_counts = {}  # subject_id -> [correct, total]
    confusion = None
    total_correct = 0
    total_samples = 0

    # evaluate the model on the validation set
    with torch.no_grad():
        for batch in dataset_loader:
            # support batches of (eeg,label,subject) or (eeg,label)
            if len(batch) == 3:
                eeg, label, subject = batch
            else:
                eeg, label = batch
                # if subject not provided set to -1 so it will be grouped separately
                subject = torch.full_like(label, -1)

            eeg = eeg.to(device)
            label = label.to(device)
            subject = subject.to(device)

            outputs = model(eeg)

            # determine predictions for binary or multi-class outputs
            if outputs.ndim == 1 or (outputs.ndim == 2 and outputs.size(1) == 1):
                probs = torch.sigmoid(outputs.view(-1))
                preds = (probs > 0.5).long()
                num_classes = 2
            else:
                preds = outputs.argmax(dim=1)
                num_classes = outputs.size(1)

            # init confusion matrix as [true_label, predicted_label]
            if confusion is None:
                confusion = torch.zeros(
                    (num_classes, num_classes), dtype=torch.int64, device="cpu"
                )

            # iterate element-wise to update per-subject stats and confusion
            for t, p, s in zip(
                label.view(-1).cpu().tolist(),
                preds.view(-1).cpu().tolist(),
                subject.view(-1).cpu().tolist(),
            ):
                s = int(s)
                t = int(t)
                p = int(p)
                counts = per_subject_counts.setdefault(s, [0, 0])
                counts[1] += 1
                if p == t:
                    counts[0] += 1
                    total_correct += 1
                total_samples += 1
                # clamp to valid class indices if needed
                if 0 <= t < confusion.shape[0] and 0 <= p < confusion.shape[1]:
                    confusion[t, p] += 1

    # compute per-subject accuracy
    per_subject_accuracy = {
        sub: (c[0] / c[1] if c[1] > 0 else 0.0) for sub, c in per_subject_counts.items()
    }

    overall_accuracy = (total_correct / total_samples) if total_samples > 0 else 0.0
    print(f"Overall validation accuracy: {overall_accuracy:.4f}")

    # return per-subject accuracy dict and confusion matrix (numpy array)
    return per_subject_accuracy, confusion.cpu().numpy(), dataset.labels  # type: ignore


if __name__ == "__main__":
    model_type = "SpectralBrainDecoder3D"
    model_options = {
        "base": 48,
        "n_blocks": 3,
    }
    hash = get_model_hash(model_type, model_options)
    datasets = [
        "eeg_signals_raw_with_mean_std",
        "eeg_55_95_std",
        "eeg_14_70_std",
        "eeg_5_95_std",
    ]
    models = get_model_path_by_hash(hash, datasets)
    best_models = find_best_models(models)

    dataset_stats = {dataset_name: {} for dataset_name in best_models}
    for dataset_name, model_data in best_models.items():
        if model_data is None:
            print(f"No model found for dataset {dataset_name}")
            continue
        model = model_data["model"]
        subject_acc, conf, labels = evaluate_model(
            model,
            model_type,
            os.path.join(PKG_ROOT, "data", "block", f"{dataset_name}.pth"),
        )

        print(f"Dataset: {dataset_name}")
        print(f"Model Type: {model_type}")
        print(f"Model Hash: {model_data['hash']}")
        print(f"Per-subject accuracy: {subject_acc}")
        print(f"Confusion Matrix:\n{conf}")

        # save conf matrix as csv
        conf_df = pd.DataFrame(
            conf,
            index=[f"True_{i}" for i in range(conf.shape[0])],
            columns=[f"Pred_{i}" for i in range(conf.shape[1])],
        )
        conf_df = conf_df.div(conf_df.sum(axis=1), axis=0)
        conf_stats = get_conf_stats(conf_df, labels, 5, 0.1)

        conf_csv_path = (
            f"confusion_matrix_{dataset_name}_{model_type}_{model_data['hash']}.csv"
        )
        conf_df.to_csv(conf_csv_path)
        print(f"Confusion matrix saved to {conf_csv_path}")

        dataset_stats[dataset_name]["subject_acc"] = subject_acc
        dataset_stats[dataset_name]["confusion"] = conf_df
        dataset_stats[dataset_name]["best_classified"] = conf_stats[0]
        dataset_stats[dataset_name]["worst_classified"] = conf_stats[1]
        dataset_stats[dataset_name]["miss_classifications"] = conf_stats[2]

        # Assume your confusion matrix is a DataFrame named `cm`
        # rows = true labels (index),
        # columns = predicted labels (columns)

    subject_acc = {
        dataset_name: dataset_stats[dataset_name]["subject_acc"]
        for dataset_name in dataset_stats
    }

    subject_acc = pd.DataFrame(subject_acc)
    subject_acc.to_csv(f"{model_type}_subject_acc.csv")

    confusions = {name: dataset_stats[name]["confusion"] for name in dataset_stats}
    dataset_names = list(confusions.keys())

    # ---- Ensure consistent label order across matrices ----
    # Use the union of all labels found in index/columns and reindex each DF.
    # all_true_labels = pd.Index(
    #     sorted(set().union(*[set(df.index) for df in confusions.values()]))
    # )
    # all_pred_labels = pd.Index(
    #     sorted(set().union(*[set(df.columns) for df in confusions.values()]))
    # )

    confusions_aligned = confusions
    # for name, cm in confusions.items():
    #     cm_aligned = cm.reindex(
    #         index=all_true_labels, columns=all_pred_labels, fill_value=0
    #     )
    #     confusions_aligned[name] = cm_aligned

    # ---- Shared color scale based on max count across all matrices ----
    max_val = max(cm.values.max() for cm in confusions_aligned.values())
    # If you want per-row normalization for each matrix instead, compute separately (see below).

    # ---- Create subplots side-by-side ----
    n = len(dataset_names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), constrained_layout=True)

    # In case there is only one dataset, axes may not be iterable
    if n == 1:
        axes = [axes]

    # Plot each confusion matrix
    for ax, name in zip(axes, dataset_names):
        cm = confusions_aligned[name]
        # stats = compute_confusion_stats(cm)
        # best = stats["best"]
        # worst = stats["worst"]
        # print("Stats for dataset: ", name)
        # print(stats)

        sns.heatmap(
            cm,
            ax=ax,
            cmap="Blues",
            vmin=0,
            vmax=max_val,  # consistent color scale across datasets
            cbar=False,  # we'll add a shared colorbar later
            annot=False,  # set True if the matrix is small enough to be readable
            fmt=".0f",
        )
        ax.set_title(f"Confusion Matrix – {name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        # Improve tick readability
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        # ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # ---- Add a single shared colorbar ----
    # Create a ScalarMappable to attach a colorbar
    norm = plt.Normalize(vmin=0, vmax=max_val)
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
    sm.set_array([])  # required for older Matplotlib versions
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Count")

    plt.show()

    # handle conf stats
    for dataset in dataset_stats:
        print("----------------------------------------------------")
        print(f"best class accuracies obtained on dataset {dataset}")
        print(dataset_stats[dataset]["best_classified"])
        print(f"worst class accuracies obtained on dataset {dataset}")
        for bad_class in dataset_stats[dataset]["miss_classifications"]:
            print(f"class {bad_class} is often classified as:")
            print(dataset_stats[dataset]["miss_classifications"][bad_class])
        print("====================================================")

    # best classified across datasets
    best_classified = set.intersection(
        *[
            set(dataset_stats[dataset]["best_classified"].index.tolist())
            for dataset in dataset_stats
        ]
    )
    print("The best classified classes across datasets are:")
    print(best_classified)

    # worst classified across datasets
    worst_classified = set.intersection(
        *[
            set(dataset_stats[dataset]["worst_classified"].index.tolist())
            for dataset in dataset_stats
        ]
    )
    print("The worst classified classes across datasets are:")
    print(worst_classified)
