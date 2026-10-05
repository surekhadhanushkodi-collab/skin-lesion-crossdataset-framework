"""Create (or verify) the lesion-level HAM10000 splits and the patient-level ISIC 2020 split.

  python scripts/make_splits.py --check   # compare with the CSV files stored in splits/
  python scripts/make_splits.py           # write the CSV files into splits/

The CSV files in splits/ are the reference. If --check reports a difference, the cause is
usually a different scikit-learn version, so use the stored files.
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

import pandas as pd

from skinfw.splits import load_ham_metadata, make_ham_splits, make_isic_split


def same(new, old_path, id_col, col):
    old = pd.read_csv(old_path)
    a = new.set_index(id_col)[col].sort_index()
    b = old.set_index(id_col)[col].sort_index()
    return a.index.equals(b.index) and (a.values == b.values).all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ham-root", default="data/ham10000")
    ap.add_argument("--isic-root", default="data/isic2020")
    ap.add_argument("--out", default="splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    ham = make_ham_splits(load_ham_metadata(args.ham_root), args.seed)
    print(ham.groupby("split")["label"].agg(images="count", melanoma="sum"))
    for s in ["train", "val", "test"]:
        f = out / f"ham10000_{s}.csv"
        part = ham[ham["split"] == s]
        if args.check:
            print(f"HAM10000 {s}: ", "MATCHES" if same(part.assign(split=s), f, "image_id", "split")
                  else "DIFFERS from the stored file")
        else:
            part.to_csv(f, index=False)

    isic_meta = Path(args.isic_root) / "train-metadata.csv"
    if isic_meta.exists():
        isic = make_isic_split(pd.read_csv(isic_meta), args.seed)
        print(isic.groupby("part")["target"].agg(images="count", melanoma="sum"))
        f = out / "isic2020_design_final_split.csv"
        if args.check:
            print("ISIC 2020 split: ", "MATCHES" if same(isic, f, "isic_id", "part")
                  else "DIFFERS from the stored file")
        else:
            isic.to_csv(f, index=False)


if __name__ == "__main__":
    main()
