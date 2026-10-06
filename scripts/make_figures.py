"""Paper figures from saved predictions (no GPU needed).

  python scripts/make_figures.py --runs runs --out figures

Needs runs/<variant>_seed<seed>/ folders with result.json and pred_*.csv for the variants
base and both, and results/design_part_ablation.csv for the ablation figure.
Every figure is saved as PNG (300 dpi) and as vector PDF.
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import precision_recall_curve, roc_curve

COL = {"base": "#E69F00", "both": "#0072B2"}  # colour-blind-safe palette
NAME = {"base": "base", "both": "both (colour normalisation + augmentation)"}
SETS = {"ham_test": "HAM10000 test (internal)", "isic_final": "ISIC 2020 final part (external)"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9, "legend.fontsize": 8, "figure.dpi": 100})


def load_runs(runs_dir):
    runs = {}
    for d in sorted(Path(runs_dir).glob("*_seed*")):
        f = d / "result.json"
        if not f.exists():
            continue
        r = json.load(open(f))
        preds = {s: pd.read_csv(d / f"pred_{s}.csv") for s in SETS if (d / f"pred_{s}.csv").exists()}
        runs[(r["variant"], int(r["seed"]))] = {"result": r, "preds": preds}
    return runs


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", out / f"{name}.png")


def mean_sd(runs, variant, s, metric):
    v = [r["result"][s][metric] for (vv, _), r in runs.items() if vv == variant and s in r["result"]]
    return np.mean(v), (np.std(v, ddof=1) if len(v) > 1 else 0.0)


def fig_roc(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True)
    for ax, s in zip(axes, SETS):
        ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="grey")
        for (variant, _), r in runs.items():
            if variant not in COL or s not in r["preds"]:
                continue
            df = r["preds"][s]
            fpr, tpr, _ = roc_curve(df["label"], df["prob"])
            ax.plot(fpr, tpr, color=COL[variant], lw=1.0, alpha=0.8)
            m = r["result"][s]
            ax.plot(1 - m["specificity"], m["sensitivity"], "o", ms=4, color=COL[variant],
                    mec="white", mew=0.6, zorder=3)
        txt = "\n".join(f"{v}: AUC {mean_sd(runs, v, s, 'roc_auc')[0]:.3f} ± {mean_sd(runs, v, s, 'roc_auc')[1]:.3f}"
                        for v in COL)
        ax.text(0.97, 0.05, txt, ha="right", va="bottom", fontsize=8)
        ax.set_title(SETS[s])
        ax.set_xlabel("1 - specificity")
        ax.set_aspect("equal")
    axes[0].set_ylabel("Sensitivity")
    handles = [Line2D([], [], color=COL[v], lw=1.5, label=NAME[v]) for v in COL]
    handles.append(Line2D([], [], marker="o", ls="", color="grey", mec="white", label="operating point (validation threshold)"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
    save(fig, out, "fig_roc_internal_external")


def fig_pr(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for ax, s in zip(axes, SETS):
        prev = None
        for (variant, _), r in runs.items():
            if variant not in COL or s not in r["preds"]:
                continue
            df = r["preds"][s]
            prev = df["label"].mean()
            p, rec, _ = precision_recall_curve(df["label"], df["prob"])
            ax.plot(rec, p, color=COL[variant], lw=1.0, alpha=0.8)
        if prev is not None:
            ax.axhline(prev, ls="--", lw=0.8, color="grey")
        txt = "\n".join(f"{v}: PR-AUC {mean_sd(runs, v, s, 'pr_auc')[0]:.3f} ± {mean_sd(runs, v, s, 'pr_auc')[1]:.3f}"
                        for v in COL)
        if s == "ham_test":
            ax.text(0.03, 0.17, txt, ha="left", va="bottom", fontsize=8, transform=ax.transAxes)
        else:
            ax.text(0.97, 0.97, txt, ha="right", va="top", fontsize=8, transform=ax.transAxes)
        ax.set_title(f"{SETS[s]}\nchance level = {prev:.3f}")
        ax.set_xlabel("Recall (sensitivity)")
        ax.set_ylim(0, 1.0 if s == "ham_test" else None)
    axes[0].set_ylabel("Precision")
    handles = [Line2D([], [], color=COL[v], lw=1.5, label=NAME[v]) for v in COL]
    handles.append(Line2D([], [], color="grey", ls="--", lw=0.8, label="chance level (prevalence)"))
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    save(fig, out, "fig_pr_internal_external")


def fig_gap(runs, out):
    fig, ax = plt.subplots(figsize=(3.8, 3.5))
    off = {"base": -0.03, "both": 0.03}
    for variant in COL:
        pts = [(r["result"]["ham_test"]["roc_auc"], r["result"]["isic_final"]["roc_auc"])
               for (v, _), r in runs.items() if v == variant]
        for a, b in pts:
            ax.plot([off[variant], 1 + off[variant]], [a, b], color=COL[variant], alpha=0.35, lw=1)
            ax.scatter([off[variant], 1 + off[variant]], [a, b], color=COL[variant], s=12, zorder=3)
        ma, mb = np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])
        ax.plot([off[variant], 1 + off[variant]], [ma, mb], color=COL[variant], lw=2.2,
                label=f"{variant}: drop {ma - mb:.2f}")
    ax.axhline(0.5, ls="--", lw=0.8, color="grey")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["HAM10000 test\n(internal)", "ISIC 2020 final\n(external)"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("ROC-AUC")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.08))
    ax.set_title("Internal to external drop (thin: seeds, thick: mean)", fontsize=8)
    save(fig, out, "fig_internal_external_gap")


def fig_ablation(csv_path, out):
    if not Path(csv_path).exists():
        print("skipped ablation figure (missing", csv_path, ")")
        return
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    y = np.arange(len(df))[::-1]
    colours = ["#999999" if v in ("original baseline", "base") else "#0072B2" for v in df["variant"]]
    ax.scatter(df["design_roc_auc"], y, c=colours, s=36, zorder=3)
    for yy, v in zip(y, df["design_roc_auc"]):
        ax.text(v + 0.006, yy, f"{v:.3f}", va="center", fontsize=8)
    ax.axvline(0.5, ls="--", lw=0.8, color="grey")
    ax.set_yticks(y)
    ax.set_yticklabels(df["variant"])
    ax.set_xlim(0.48, 0.76)
    ax.set_xlabel("ROC-AUC on the ISIC 2020 design part (one seed)")
    save(fig, out, "fig_variant_ablation_design_part")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--ablation", default="results/design_part_ablation.csv")
    args = ap.parse_args()
    runs = load_runs(args.runs)
    if not runs:
        raise SystemExit(f"No runs found in {args.runs}")
    out = Path(args.out)
    fig_roc(runs, out)
    fig_pr(runs, out)
    fig_gap(runs, out)
    fig_ablation(args.ablation, out)


if __name__ == "__main__":
    main()
