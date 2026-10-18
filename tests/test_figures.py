"""Smoke test for scripts/make_figures.py on synthetic predictions (no GPU, no data)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skinfw.metrics import point_metrics  # noqa: E402


def test_make_figures_creates_all_figures(tmp_path):
    rng = np.random.default_rng(0)
    for variant in ("base", "both"):
        for seed in (42, 1):
            d = tmp_path / "runs" / f"{variant}_seed{seed}"
            d.mkdir(parents=True)
            res = {"variant": variant, "seed": seed, "val_auc": 0.9, "threshold": 0.5}
            for name, n, prev in [("ham_test", 800, 0.12), ("isic_final", 1500, 0.02)]:
                y = (rng.random(n) < prev).astype(int)
                p = np.clip(rng.normal(0.25, 0.2, n) + 0.3 * y, 0, 1)
                pd.DataFrame({"id": range(n), "label": y, "prob": p}).to_csv(d / f"pred_{name}.csv", index=False)
                res[name] = point_metrics(p, y, 0.5)
            json.dump(res, open(d / "result.json", "w"))
    out = tmp_path / "figures"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_figures.py"),
                        "--runs", str(tmp_path / "runs"), "--out", str(out),
                        "--ablation", str(ROOT / "results" / "design_part_ablation.csv")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for name in ("fig_roc_internal_external", "fig_pr_internal_external",
                 "fig_internal_external_gap", "fig_variant_ablation_design_part"):
        assert (out / f"{name}.png").exists() and (out / f"{name}.pdf").exists()


def test_make_figures_filters_by_backbone(tmp_path):
    rng = np.random.default_rng(1)
    for backbone, prefix in [("resnet50", "resnet50_")]:
        for variant in ("base", "both"):
            for seed in (42, 1):
                d = tmp_path / "runs" / f"{prefix}{variant}_seed{seed}"
                d.mkdir(parents=True)
                res = {"backbone": backbone, "variant": variant, "seed": seed, "val_auc": 0.9, "threshold": 0.5}
                for name, n, prev in [("ham_test", 600, 0.12), ("isic_final", 1200, 0.02)]:
                    y = (rng.random(n) < prev).astype(int)
                    p = np.clip(rng.normal(0.25, 0.2, n) + 0.3 * y, 0, 1)
                    pd.DataFrame({"id": range(n), "label": y, "prob": p}).to_csv(d / f"pred_{name}.csv", index=False)
                    res[name] = point_metrics(p, y, 0.5)
                json.dump(res, open(d / "result.json", "w"))
    script = str(ROOT / "scripts" / "make_figures.py")
    abl = str(ROOT / "results" / "design_part_ablation.csv")
    # default backbone (efficientnet_b0) finds nothing here, so the script must stop
    r = subprocess.run([sys.executable, script, "--runs", str(tmp_path / "runs"), "--out", str(tmp_path / "f0"),
                        "--ablation", abl], capture_output=True, text=True)
    assert r.returncode != 0
    r = subprocess.run([sys.executable, script, "--runs", str(tmp_path / "runs"), "--out", str(tmp_path / "f1"),
                        "--backbone", "resnet50", "--ablation", abl], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "f1" / "fig_roc_internal_external.png").exists()
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "summarize_seeds.py"), "--runs", str(tmp_path / "runs"),
                        "--out", str(tmp_path / "s.csv")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "resnet50" in r.stdout
