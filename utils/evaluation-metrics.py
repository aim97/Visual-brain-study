import numpy as np
import pandas as pd
import math

# TODO: make sure it's a weighted f1-score
def compute_macro_f1_and_ci(conf_mat_df):
    """
    Computes Top-1 Accuracy, Macro-F1, and 95% Confidence Interval
    purely from an unnormalized confusion matrix dataframe.
    """
    cm = conf_mat_df.values
    total = np.sum(cm)
    correct = np.trace(cm)
    acc = correct / total if total > 0 else 0.0

    # Avoid division by zero
    eps = 1e-9
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1_scores = 2 * (precision * recall) / (precision + recall + eps)

    macro_f1 = np.mean(f1_scores)

    # 95% Wald Confidence Interval
    ci = 1.96 * math.sqrt((acc * (1 - acc)) / total) if total > 0 else 0.0

    return acc, macro_f1, ci

def update_per_subject_stats(per_subject_counts, subjects, targets, preds):
    """
    Updates the per-subject correctness counts during the validation loop.
    """
    for s, t, p in zip(subjects, targets, preds):
        s_id = int(s)
        counts = per_subject_counts.setdefault(s_id, [0, 0]) # [correct, total]
        counts[1] += 1
        if int(t) == int(p):
            counts[0] += 1
    return per_subject_counts

def calculate_subject_accuracy(per_subject_counts):
    """
    Converts raw counts to percentages.
    """
    return {sub: (c[0] / c[1] if c[1] > 0 else 0.0) for sub, c in per_subject_counts.items()}
