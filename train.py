"""Train one run and score it on the test sets.

  python scripts/train.py --variant both --seed 42
"""
import _bootstrap  # noqa: F401
import argparse

import yaml

from skinfw.data import VARIANTS
from skinfw.engine import run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    for k in ("variant", "seed", "epochs"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    run_training(cfg, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
