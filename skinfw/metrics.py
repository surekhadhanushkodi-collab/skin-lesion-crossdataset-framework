"""Metrics, threshold selection and grouped bootstrap (no torch needed)."""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             roc_auc_score, roc_curve)

METRICS = ("roc_auc", "pr_auc", "sensitivity", "specificity", "ppv")


def youden_threshold(y, p):
    """Threshold maximising sensitivity + specificity - 1. Choose it on validation data only."""
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def point_metrics(p, y, thr):
    tn, fp, fn, tp = confusion_matrix(y, np.asarray(p) >= thr, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "ppv": float(tp / max(tp + fp, 1)),
    }


def grouped_bootstrap(p, y, groups, thr, n_boot=1000, seed=42, ref=None):
    """95% CIs by resampling whole groups (lesions or patients), not single images.

    ref = (p_ref, thr_ref) optionally gives a paired CI for the ROC-AUC difference
    (this model minus the reference model) on the same resamples.
    """
    p, y = np.asarray(p), np.asarray(y)
    index = pd.DataFrame({"g": np.asarray(groups)}).groupby("g").indices
    keys = list(index)
    rng = np.random.default_rng(seed)
    boot = {m: [] for m in METRICS}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        idx = np.concatenate([index[keys[i]] for i in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        r = point_metrics(p[idx], y[idx], thr)
        for m in METRICS:
            boot[m].append(r[m])
        if ref is not None:
            diffs.append(r["roc_auc"] - roc_auc_score(y[idx], ref[0][idx]))
    out = {"point": point_metrics(p, y, thr),
           "ci95": {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                    for m, v in boot.items()},
           "n_boot": n_boot}
    if ref is not None:
        out["roc_auc_diff_vs_ref"] = {
            "point": out["point"]["roc_auc"] - float(roc_auc_score(y, ref[0])),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    return out
