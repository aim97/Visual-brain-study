import pandas as pd
import numpy as np


def _compute_bounds(totals, target=0.2, low=0.18, high=0.22):
    """
    Compute integer lower/upper bounds for each semantic.
    """
    lb = np.floor(totals * low).astype(int)
    ub = np.ceil(totals * high).astype(int)
    tgt = np.round(totals * target).astype(int)
    # Ensure lb <= tgt <= ub (may be equal due to small totals)
    lb = np.minimum(lb, tgt)
    ub = np.maximum(ub, tgt)
    return lb, ub, tgt


def _score_violation(sums, lb, ub, overshoot_weight=2.0):
    """
    Returns a scalar score: 0 is perfect.
    - Undershoot contributes linearly.
    - Overshoot penalized more (overshoot_weight).
    """
    under = np.maximum(0, lb - sums)
    over = np.maximum(0, sums - ub)
    return under.sum() + overshoot_weight * over.sum()


def _apply_row(sums, row, sign=+1):
    """
    Update sums by adding (sign=+1) or removing (sign=-1) a row.
    """
    return sums + sign * row


def select_test_labels_tight(
    df,
    semantic_cols,
    target=0.2,
    low=0.18,
    high=0.22,
    overshoot_weight=2.0,
    randomize=True,
    seed=42,
    max_iters=2000,
):
    """
    Improved selection:
    1) Greedy add labels to approach targets.
    2) Iterative repair: remove labels to reduce overshoot, add labels for undershoot.
    3) Optional random restarts for robustness.

    Returns: list of selected labels, achieved sums, bounds.
    """
    rng = np.random.default_rng(seed)

    # Totals and bounds
    totals = df[semantic_cols].sum(axis=0).astype(int)
    lb, ub, tgt = _compute_bounds(totals.values, target, low, high)
    lb = pd.Series(lb, index=semantic_cols)
    ub = pd.Series(ub, index=semantic_cols)
    tgt = pd.Series(tgt, index=semantic_cols)

    rows = df.set_index("label")[semantic_cols].astype(int)
    labels = list(rows.index)

    if randomize:
        rng.shuffle(labels)

    # ---- Phase 1: Greedy build toward targets (prioritize undershot semantics) ----
    selected = set()
    sums = pd.Series(0, index=semantic_cols, dtype=int)

    remaining = set(labels)
    while True:
        # If all within [lb, ub], stop
        if ((sums >= lb) & (sums <= ub)).all():
            break

        # Compute gain per candidate
        best_label = None
        best_gain = -1
        best_delta_violation = 0.0

        # Current violation score
        cur_v = _score_violation(sums.values, lb.values, ub.values, overshoot_weight)

        for lbl in list(remaining):
            row = rows.loc[lbl]
            # Tentative add
            new_sums = _apply_row(sums, row, sign=+1)
            new_v = _score_violation(
                new_sums.values, lb.values, ub.values, overshoot_weight
            )
            delta = cur_v - new_v  # positive is improvement
            if delta > best_gain or (
                delta == best_gain and randomize and rng.random() < 0.2
            ):
                best_gain = delta
                best_label = lbl
                best_delta_violation = delta

        if best_label is None or best_gain <= 0:
            # No progress possible with additions
            break

        # Accept the best label
        selected.add(best_label)
        remaining.remove(best_label)
        sums = _apply_row(sums, rows.loc[best_label], sign=+1)

    # ---- Phase 2: Local improvement (repair overshoot / undershoot) ----
    # Attempt swaps/removals/additions to reduce violation.
    iters = 0
    improved = True
    while improved and iters < max_iters:
        iters += 1
        improved = False
        cur_v = _score_violation(sums.values, lb.values, ub.values, overshoot_weight)

        # 2a) If overshoot exists, try removing the label that reduces violation most
        over_mask = sums > ub
        if over_mask.any() and selected:
            best_lbl = None
            best_gain = 0.0
            for lbl in list(selected):
                new_sums = _apply_row(sums, rows.loc[lbl], sign=-1)
                new_v = _score_violation(
                    new_sums.values, lb.values, ub.values, overshoot_weight
                )
                delta = cur_v - new_v
                if delta > best_gain or (
                    delta == best_gain and randomize and rng.random() < 0.2
                ):
                    best_gain = delta
                    best_lbl = lbl
            if best_lbl is not None and best_gain > 0:
                selected.remove(best_lbl)
                remaining.add(best_lbl)
                sums = _apply_row(sums, rows.loc[best_lbl], sign=-1)
                improved = True
                continue  # re-evaluate

        # 2b) If undershoot exists, try adding the label that reduces violation most
        under_mask = sums < lb
        if under_mask.any() and remaining:
            best_lbl = None
            best_gain = 0.0
            for lbl in list(remaining):
                new_sums = _apply_row(sums, rows.loc[lbl], sign=+1)
                new_v = _score_violation(
                    new_sums.values, lb.values, ub.values, overshoot_weight
                )
                delta = cur_v - new_v
                if delta > best_gain or (
                    delta == best_gain and randomize and rng.random() < 0.2
                ):
                    best_gain = delta
                    best_lbl = lbl
            if best_lbl is not None and best_gain > 0:
                selected.add(best_lbl)
                remaining.remove(best_lbl)
                sums = _apply_row(sums, rows.loc[best_lbl], sign=+1)
                improved = True
                continue  # re-evaluate

        # 2c) Try swaps: remove one selected and add one remaining if it helps
        if selected and remaining:
            best_pair = None
            best_gain = 0.0
            for rem_lbl in (
                rng.choice(list(selected), size=min(10, len(selected)), replace=False)
                if randomize
                else selected
            ):
                tmp_sums = _apply_row(sums, rows.loc[rem_lbl], sign=-1)
                for add_lbl in (
                    rng.choice(
                        list(remaining), size=min(30, len(remaining)), replace=False
                    )
                    if randomize
                    else remaining
                ):
                    new_sums = _apply_row(tmp_sums, rows.loc[add_lbl], sign=+1)
                    new_v = _score_violation(
                        new_sums.values, lb.values, ub.values, overshoot_weight
                    )
                    delta = cur_v - new_v
                    if delta > best_gain:
                        best_gain = delta
                        best_pair = (rem_lbl, add_lbl, new_sums)
            if best_pair and best_gain > 0:
                rem_lbl, add_lbl, new_sums = best_pair
                selected.remove(rem_lbl)
                remaining.add(rem_lbl)
                selected.add(add_lbl)
                remaining.remove(add_lbl)
                sums = new_sums  # already applied
                improved = True

    return list(selected), sums, pd.DataFrame({"lb": lb, "tgt": tgt, "ub": ub})


# Read your CSV (space- or comma-separated)
df = pd.read_csv("distribution.csv", sep=r"\s+|,", engine="python")
if df.columns[0] != "label":
    df.rename(columns={df.columns[0]: "label"}, inplace=True)

semantic_cols = [c for c in df.columns if c != "label"]

test_labels, test_sums, bounds = select_test_labels_tight(
    df,
    semantic_cols,
    target=0.20,
    low=0.15,
    high=0.48,
    overshoot_weight=5.0,  # penalize overshoot more
    randomize=True,
    seed=42,
    max_iters=100000,
)

test_df = df[df["label"].isin(test_labels)].copy()
train_df = df[~df["label"].isin(test_labels)].copy()

totals = df[semantic_cols].sum()
achieved = (test_df[semantic_cols].sum() / totals.replace(0, np.nan)).round(3)

print("\n== Bounds ==")
print(bounds)

print("\n== Achieved test ratios per semantic ==")
print(achieved.fillna(0))

# Save splits
test_df.to_csv("test-set.csv", index=False)
train_df.to_csv("train-set.csv", index=False)
