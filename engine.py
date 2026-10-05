"""Training and scoring of one run (one variant, one seed)."""
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .data import ImgDS, make_transforms
from .metrics import point_metrics, youden_threshold
from .model import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(x.to(device)).squeeze(1)
            probs.append(torch.sigmoid(out.float()).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def _loader(paths, labels, tf, batch_size, shuffle, num_workers, pin):
    return DataLoader(ImgDS(paths, labels, tf), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin)


def run_training(cfg, overwrite=False):
    """Train one model, pick the epoch and threshold on HAM10000 validation only,
    then score the HAM10000 test set and (if available) the ISIC 2020 final part.

    Writes best_model.pt, result.json, config_used.json and pred_*.csv to
    <out_dir>/<variant>_seed<seed>/ and returns the result dictionary.
    """
    variant, seed = cfg["variant"], int(cfg["seed"])
    p = cfg["paths"]
    out = Path(p["out_dir"]) / f"{variant}_seed{seed}"
    if (out / "result.json").exists() and not overwrite:
        print(f"Already done: {out} (use --overwrite to redo)")
        return json.load(open(out / "result.json"))
    out.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(out / "config_used.json", "w"), indent=2)

    set_seed(seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    ham_root, splits = Path(p["ham_root"]), Path(p["splits_dir"])
    bs, nw = int(cfg["batch_size"]), int(cfg["num_workers"])

    tr = pd.read_csv(splits / "ham10000_train.csv")
    va = pd.read_csv(splits / "ham10000_val.csv")
    te = pd.read_csv(splits / "ham10000_test.csv")
    full = lambda df: [str(ham_root / q) for q in df["path"]]

    tr_tf, ev_tf = make_transforms(variant, int(cfg["image_size"]))
    tr_dl = _loader(full(tr), tr["label"], tr_tf, bs, True, nw, use_cuda)
    va_dl = _loader(full(va), va["label"], ev_tf, bs, False, nw, use_cuda)

    model = build_model(cfg["backbone"], bool(cfg["pretrained"])).to(device)
    y_tr = tr["label"].values
    pos_weight = torch.tensor([(y_tr == 0).sum() / (y_tr == 1).sum()],
                              dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    epochs = int(cfg["epochs"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]),
                            weight_decay=float(cfg["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    best, history = 0.0, []
    for ep in range(1, epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_cuda):
                loss = criterion(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item() * len(y)
        sched.step()
        pv, yv = predict(model, va_dl, device)
        auc = float(roc_auc_score(yv, pv))
        history.append({"epoch": ep, "train_loss": run / len(tr), "val_roc_auc": auc})
        if auc > best:
            best = auc
            torch.save(model.state_dict(), out / "best_model.pt")
        print(f"[{variant} s{seed}] ep {ep}/{epochs} | loss {run / len(tr):.3f} | "
              f"val ROC-AUC {auc:.3f} | {time.time() - t0:.0f}s")

    model.load_state_dict(torch.load(out / "best_model.pt", map_location=device))
    pv, yv = predict(model, va_dl, device)
    thr = youden_threshold(yv, pv)
    result = {"variant": variant, "seed": seed, "val_auc": best,
              "threshold": thr, "history": history}
    pd.DataFrame({"id": va["image_id"], "label": yv, "prob": pv}).to_csv(
        out / "pred_ham_val.csv", index=False)

    eval_sets = {"ham_test": (te["image_id"], full(te), te["label"])}
    isic_file = splits / "isic2020_design_final_split.csv"
    isic_root = Path(p["isic_root"])
    if isic_file.exists() and isic_root.exists():
        sp = pd.read_csv(isic_file)
        fin = sp[sp["part"] == "final"].reset_index(drop=True)
        eval_sets["isic_final"] = (
            fin["isic_id"],
            [str(isic_root / "train-image" / "image" / f"{i}.jpg") for i in fin["isic_id"]],
            fin["target"])
    for name, (ids, paths, labels) in eval_sets.items():
        dl = _loader(paths, labels, ev_tf, 64, False, nw, use_cuda)
        prob, y = predict(model, dl, device)
        pd.DataFrame({"id": list(ids), "label": y, "prob": prob}).to_csv(
            out / f"pred_{name}.csv", index=False)
        result[name] = point_metrics(prob, y, thr)

    json.dump(result, open(out / "result.json", "w"), indent=2)
    print({k: v for k, v in result.items() if k != "history"})
    return result
