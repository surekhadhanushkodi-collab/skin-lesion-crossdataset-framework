"""Table of mean and SD across seeds, from runs/*/result.json (no GPU needed)."""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/seed_summary.csv")
    args = ap.parse_args()
    rows = []
    for f in sorted(Path(args.runs).glob("*/result.json")):
        r = json.load(open(f))
        row = {"variant": r["variant"], "seed": r["seed"], "val_auc": r["val_auc"],
               "threshold": r["threshold"]}
        for s in ("ham_test", "isic_final"):
            for m, v in r.get(s, {}).items():
                row[f"{s}_{m}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    summary = df.drop(columns="seed").groupby("variant").agg(["mean", "std"]).round(3)
    print(df.round(3).to_string(index=False))
    print()
    print(summary.T.to_string())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
