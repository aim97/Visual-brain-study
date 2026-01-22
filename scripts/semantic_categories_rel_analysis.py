import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main(output_dataset_path, out_csv_path=None, make_plot=False, plot_path=None):
    # Load the output dataset
    ds = torch.load(output_dataset_path)

    # Expect keys: "labels", "semantics", "dataset"
    labels = ds["labels"]  # list of label names
    semantics = ds["semantics"]  # list of semantic names
    samples = ds["dataset"]  # list of dicts, each with 'label' and 'semantic'

    # Build a DataFrame of samples with readable names
    rows = []
    for s in samples:
        # s['label'] and s['semantic'] are indices
        lbl_idx = s["label"]
        sem_idx = s["semantic"]
        # Safety: bounds check
        if lbl_idx < 0 or lbl_idx >= len(labels):
            continue
        if sem_idx < 0 or sem_idx >= len(semantics):
            continue
        rows.append(
            {
                "label_idx": lbl_idx,
                "semantic_idx": sem_idx,
                "label": labels[lbl_idx],
                "semantic": semantics[sem_idx],
            }
        )

    df = pd.DataFrame(rows, columns=["label_idx", "semantic_idx", "label", "semantic"])

    # Create full index to ensure zero-count rows/cols are included
    label_order = pd.Index(labels, name="label")
    semantic_order = pd.Index(semantics, name="semantic")

    # Cross-tab (counts)
    ct = pd.crosstab(df["label"], df["semantic"]).reindex(
        index=label_order, columns=semantic_order, fill_value=0
    )

    # Print a compact summary
    print("\n=== Intersections: counts per (label × semantic) ===")
    print(ct)

    # Also print totals
    print("\nRow totals (per label):")
    print(ct.sum(axis=1))

    print("\nColumn totals (per semantic):")
    print(ct.sum(axis=0))

    # Save to CSV if requested
    if out_csv_path:
        ct.to_csv(out_csv_path, index=True)
        print(f"\nSaved intersections table to: {out_csv_path}")

    # Optional: normalized versions for convenience
    ct_by_label = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0)  # row-normalized
    ct_by_semantic = ct.div(
        ct.sum(axis=0).replace(0, np.nan), axis=1
    )  # column-normalized

    print("\n=== Row-normalized (proportion within each label) ===")
    print(ct_by_label.fillna(0).round(3))

    print("\n=== Column-normalized (proportion within each semantic) ===")
    print(ct_by_semantic.fillna(0).round(3))

    # Optional plot
    if make_plot:
        plt.figure(figsize=(max(8, len(semantics) * 0.5), max(6, len(labels) * 0.4)))
        im = plt.imshow(ct.values, aspect="auto", cmap="viridis")
        plt.colorbar(im, label="Count")
        plt.xticks(range(len(semantics)), semantics, rotation=90)
        plt.yticks(range(len(labels)), labels)
        plt.title("Intersections between Labels and Semantics (Counts)")
        plt.xlabel("Semantic")
        plt.ylabel("Label")
        plt.tight_layout()
        if plot_path:
            plt.savefig(plot_path, dpi=200)
            print(f"Saved heatmap to: {plot_path}")
        else:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute intersections between labels and semantics from output_dataset"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        help="Path to output_dataset .pt file (torch.save)",
    )
    parser.add_argument(
        "--out",
        "-o",
        required=False,
        help="Output CSV path (e.g., label_semantic_intersections.csv)",
    )
    parser.add_argument("--plot", action="store_true", help="Render a heatmap plot")
    parser.add_argument(
        "--plot-path",
        required=False,
        help="Path to save plot (PNG). If not provided, shows interactively.",
    )
    args = parser.parse_args()

    main(
        output_dataset_path=args.dataset,
        out_csv_path=args.out,
        make_plot=args.plot,
        plot_path=args.plot_path,
    )
