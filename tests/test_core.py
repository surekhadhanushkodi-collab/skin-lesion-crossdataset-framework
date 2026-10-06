"""Torch-free tests: splits, metrics, colour normalisation and the command-line scripts.
Run with:  pytest -q
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skinfw.colour import ShadesOfGray  # noqa: E402
from skinfw.metrics import grouped_bootstrap, point_metrics, youden_threshold  # noqa: E402
from skinfw.splits import load_ham_metadata, make_ham_splits, make_isic_split  # noqa: E402


def fake_ham(root, n_lesions=1400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for L in range(n_lesions):
        dx = "mel" if rng.random() < 0.11 else rng.choice(["nv", "bkl", "bcc"])
        for k in range(rng.integers(1, 4)):
            rows.append({"lesion_id": f"HAM_{L}", "image_id": f"ISIC_{L}_{k}", "dx": dx,
                         "dx_type": "histo", "age": 50, "sex": "male", "localization": "back"})
    meta = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    meta.to_csv(root / "HAM10000_metadata.csv", index=False)
    for d in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        (root / d).mkdir(exist_ok=True)
    for i, name in enumerate(meta["image_id"]):
        d = "HAM10000_images_part_1" if i % 2 == 0 else "HAM10000_images_part_2"
        (root / d / f"{name}.jpg").touch()
    return meta


def test_ham_split_has_no_lesion_leakage_and_is_reproducible(tmp_path):
    fake_ham(tmp_path)
    meta = load_ham_metadata(tmp_path)
    a = make_ham_splits(meta, seed=42)
    b = make_ham_splits(meta, seed=42)
    assert (a["split"].values == b["split"].values).all()
    assert (a.groupby("lesion_id")["split"].nunique() == 1).all()
    frac = a["split"].value_counts(normalize=True)
    assert 0.6 < frac["train"] < 0.8 and 0.1 < frac["val"] < 0.2 and 0.1 < frac["test"] < 0.2
    for s in ("train", "val", "test"):
        assert 0.05 < a.loc[a["split"] == s, "label"].mean() < 0.2


def test_isic_split_is_patient_level():
    rng = np.random.default_rng(1)
    m = pd.DataFrame({"isic_id": [f"I{i}" for i in range(3000)],
                      "patient_id": [f"P{rng.integers(0, 400)}" for _ in range(3000)],
                      "target": (rng.random(3000) < 0.02).astype(int)})
    s = make_isic_split(m, seed=42)
    assert (s.groupby("patient_id")["part"].nunique() == 1).all()
    assert 0.2 < (s["part"] == "design").mean() < 0.4


def test_metrics_and_threshold():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.1).astype(int)
    p = np.clip(0.5 * y + rng.normal(0.25, 0.2, 2000), 0, 1)
    thr = youden_threshold(y, p)
    m = point_metrics(p, y, thr)
    assert m["roc_auc"] > 0.8 and 0 <= m["sensitivity"] <= 1 and 0 <= m["specificity"] <= 1


def test_grouped_bootstrap_ci_contains_point_and_diff():
    rng = np.random.default_rng(0)
    n = 1500
    groups = rng.integers(0, 600, n)
    y = (rng.random(n) < 0.1).astype(int)
    good = np.clip(0.5 * y + rng.normal(0.25, 0.2, n), 0, 1)
    weak = np.clip(0.2 * y + rng.normal(0.25, 0.2, n), 0, 1)
    out = grouped_bootstrap(good, y, groups, 0.5, n_boot=200, ref=(weak, 0.5))
    lo, hi = out["ci95"]["roc_auc"]
    assert lo <= out["point"]["roc_auc"] <= hi
    assert out["roc_auc_diff_vs_ref"]["ci95"][0] > 0


def test_shades_of_gray_neutralises_colour_cast():
    base = np.random.default_rng(0).integers(60, 200, (64, 64, 1)).astype(np.float64)
    cast = np.concatenate([base * 1.0, base * 0.8, base * 0.5], axis=2)
    img = Image.fromarray(np.clip(cast, 0, 255).astype(np.uint8))
    out = np.asarray(ShadesOfGray()(img), dtype=float).reshape(-1, 3).mean(0)
    assert out.max() / out.min() < 1.1


def test_scripts_make_splits_check_evaluate_and_summarize(tmp_path):
    ham = tmp_path / "ham"
    fake_ham(ham)
    splits = tmp_path / "splits"
    env = dict(**__import__("os").environ)
    run = lambda *a: subprocess.run([sys.executable, str(ROOT / "scripts" / a[0]), *a[1:]],
                                    capture_output=True, text=True, cwd=tmp_path, env=env)
    r = run("make_splits.py", "--ham-root", str(ham), "--isic-root", str(tmp_path / "none"),
            "--out", str(splits))
    assert r.returncode == 0, r.stderr
    r = run("make_splits.py", "--ham-root", str(ham), "--isic-root", str(tmp_path / "none"),
            "--out", str(splits), "--check")
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("MATCHES") == 3 and "DIFFERS" not in r.stdout

    te = pd.read_csv(splits / "ham10000_test.csv")
    rng = np.random.default_rng(0)
    y = te["label"].values
    for name, strength in [("both_seed42", 0.55), ("base_seed42", 0.2)]:
        d = tmp_path / "runs" / name
        d.mkdir(parents=True)
        prob = np.clip(strength * y + rng.normal(0.25, 0.2, len(y)), 0, 1)
        pd.DataFrame({"id": te["image_id"], "label": y, "prob": prob}).to_csv(d / "pred_ham_test.csv", index=False)
        json.dump({"variant": name.split("_")[0], "seed": 42, "val_auc": 0.9, "threshold": 0.5,
                   "ham_test": point_metrics(prob, y, 0.5)}, open(d / "result.json", "w"))
    r = run("evaluate.py", "--run", "runs/both_seed42", "--set", "ham_test", "--ref", "runs/base_seed42",
            "--splits", str(splits), "--n-boot", "100")
    assert r.returncode == 0, r.stderr
    assert "roc_auc" in r.stdout and "minus reference" in r.stdout
    r = run("summarize_seeds.py", "--runs", "runs", "--out", "results/s.csv")
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "results" / "s.csv").exists()
