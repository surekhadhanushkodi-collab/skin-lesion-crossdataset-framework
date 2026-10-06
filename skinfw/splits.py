"""Leakage-safe splits.

HAM10000 is split by lesion (several images can show the same lesion).
ISIC 2020 is split by patient (one patient can have many images).
Both use StratifiedGroupKFold with a fixed seed, so the splits are reproducible.
"""
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

HAM_FOLDERS = ["HAM10000_images_part_1", "HAM10000_images_part_2"]


def load_ham_metadata(ham_root):
    """Read HAM10000 metadata, add the melanoma label (1 = mel) and image paths."""
    ham_root = Path(ham_root)
    meta = pd.read_csv(ham_root / "HAM10000_metadata.csv")
    meta["label"] = (meta["dx"] == "mel").astype(int)
    paths = {}
    for d in HAM_FOLDERS:
        for f in os.listdir(ham_root / d):
            paths[f[:-4]] = f"{d}/{f}"
    meta["path"] = meta["image_id"].map(paths)
    if meta["path"].isna().any():
        raise ValueError("Some metadata images were not found on disk.")
    return meta


def _grouped_folds(df, label_col, group_col, n_splits, seed):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = pd.Series(-1, index=df.index)
    for k, (_, idx) in enumerate(sgkf.split(df, df[label_col], groups=df[group_col])):
        fold.iloc[idx] = k
    return fold


def make_ham_splits(meta, seed=42):
    """About 70/15/15 train/val/test, split by lesion_id (3 of 20 folds test, 3 val)."""
    meta = meta.copy()
    fold = _grouped_folds(meta, "label", "lesion_id", 20, seed)
    meta["split"] = "train"
    meta.loc[fold.isin([0, 1, 2]), "split"] = "test"
    meta.loc[fold.isin([3, 4, 5]), "split"] = "val"
    if not (meta.groupby("lesion_id")["split"].nunique() == 1).all():
        raise AssertionError("Lesion leakage found")
    return meta


def make_isic_split(isic_meta, seed=42):
    """About 30% 'design' and 70% 'final', split by patient_id (3 of 10 folds design)."""
    m = isic_meta[["isic_id", "patient_id", "target"]].copy()
    fold = _grouped_folds(m, "target", "patient_id", 10, seed)
    m["part"] = "final"
    m.loc[fold <= 2, "part"] = "design"
    if not (m.groupby("patient_id")["part"].nunique() == 1).all():
        raise AssertionError("Patient leakage found")
    return m
