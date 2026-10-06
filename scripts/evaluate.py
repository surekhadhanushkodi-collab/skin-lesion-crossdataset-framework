"""Confidence intervals for a finished run, from its saved predictions (no GPU needed).

  python scripts/evaluate.py --run runs/both_seed42 --set isic_final --ref runs/base_seed42

Intervals come from 1,000 bootstrap resamples of whole lesions (ham_test) or whole
patients (isic_final). --ref adds a paired ROC-AUC difference against another run.
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

import pandas as pd

from skinfw.metrics import grouped_bootstrap

SETS = {  # name: (split file, id column, group column, filter)
    "ham_test": ("ham10000_test.csv", "image_id", "lesion_id", None),
    "isic_final": ("isic2020_design_final_split.csv", "isic_id", "patient_id", ("part", "final")),
}


def load(run, name, splits):
    run = Path(run)
    res = json.load(open(run / "result.json"))
    pred = pd.read_csv(run / f"pred_{name}.csv")
    f, id_col, grp, filt = SETS[name]
    meta = pd.read_csv(Path(splits) / f)
    if filt:
        meta = meta[meta[filt[0]] == filt[1]]
    df = pred.merge(meta[[id_col, grp]], left_on="id", right_on=id_col, how="left")
    if df[grp].isna().any() or len(df) != len(pred):
        raise ValueError("Predictions and split file do not match.")
    return df, res["threshold"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--set", required=True, choices=list(SETS))
    ap.add_argument("--ref")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    df, thr = load(args.run, args.set, args.splits)
    grp = SETS[args.set][2]
    ref = None
    if args.ref:
        rdf, _ = load(args.ref, args.set, args.splits)
        rdf = rdf.set_index("id").loc[df["id"]]
        ref = (rdf["prob"].values, None)
    out = grouped_bootstrap(df["prob"].values, df["label"].values, df[grp].values,
                            thr, n_boot=args.n_boot, ref=ref)
    for m, v in out["point"].items():
        lo, hi = out["ci95"][m]
        print(f"{m}: {v:.3f} (95% CI {lo:.3f} to {hi:.3f})")
    if ref is not None:
        d = out["roc_auc_diff_vs_ref"]
        print(f"ROC-AUC minus reference: {d['point']:.3f} (95% CI {d['ci95'][0]:.3f} to {d['ci95'][1]:.3f})")
    json.dump(out, open(Path(args.run) / f"eval_{args.set}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
