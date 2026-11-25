"""Train a multi-label chest X-ray classifier.

    python -m src.train --config configs/default.yaml --data-dir /kaggle/temp/nih256 --out-dir outputs/densenet121_wbce

Any config key can be overridden from the command line, e.g. `--loss focal --epochs 5`.
Use `--subset 500` for a quick smoke test on a few hundred images.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import CXRDataset, build_transforms, load_metadata, make_splits, pos_weights
from .evaluate import mean_auroc, per_class_auroc, predict
from .labels import CLASSES
from .losses import build_loss
from .model import build_model

DEFAULTS = {
    "data_dir": None,
    "out_dir": "outputs/densenet121_wbce",
    "pretrained": True,
    "img_size": 224,
    "batch_size": 32,
    "epochs": 10,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "loss": "weighted_bce",
    "patience": 3,
    "num_workers": 2,
    "val_frac": 0.1,
    "seed": 42,
    "subset": 0,
    "amp": True,
}


def load_config(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML config file")
    for key, val in DEFAULTS.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(val, bool):
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        else:
            ap.add_argument(flag, type=type(val) if val is not None else str, default=None)
    args = ap.parse_args(argv)

    cfg = dict(DEFAULTS)
    if args.config:
        cfg.update(yaml.safe_load(Path(args.config).read_text()) or {})
    for key in DEFAULTS:
        v = getattr(args, key)
        if v is not None:
            cfg[key] = v
    if not cfg["data_dir"]:
        ap.error("--data-dir is required (or set data_dir in the config)")
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(cfg: dict) -> dict:
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg["amp"] and device.type == "cuda"
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    print(f"device={device}  amp={use_amp}  out_dir={out_dir}")

    # ---- data
    df = load_metadata(cfg["data_dir"])
    train_df, val_df, test_df = make_splits(df, cfg["data_dir"], cfg["val_frac"], cfg["seed"])
    if cfg["subset"]:
        n = cfg["subset"]
        train_df = train_df.sample(min(n, len(train_df)), random_state=cfg["seed"])
        val_df = val_df.sample(min(max(n // 4, 1), len(val_df)), random_state=cfg["seed"])
    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)} images")

    train_loader = DataLoader(CXRDataset(train_df, build_transforms(cfg["img_size"], True)),
                              batch_size=cfg["batch_size"], shuffle=True, drop_last=len(train_df) > cfg["batch_size"],
                              num_workers=cfg["num_workers"], pin_memory=device.type == "cuda")
    val_loader = DataLoader(CXRDataset(val_df, build_transforms(cfg["img_size"], False)),
                            batch_size=cfg["batch_size"] * 2, num_workers=cfg["num_workers"],
                            pin_memory=device.type == "cuda")

    # ---- model / loss / optimiser
    model = build_model(cfg["pretrained"]).to(device)
    pw = pos_weights(train_df).to(device)
    criterion = build_loss(cfg["loss"], pw).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best_auc, bad_epochs = [], -1.0, 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
            loss = criterion(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0); seen += x.size(0)
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {running / seen:.4f}")
        scheduler.step()

        val_prob, val_true = predict(model, val_loader, device)
        val_auc = mean_auroc(val_true, val_prob)
        rec = {"epoch": epoch, "train_loss": running / max(seen, 1), "val_mean_auroc": val_auc,
               "val_auroc": dict(zip(CLASSES, per_class_auroc(val_true, val_prob).tolist())),
               "lr": optimizer.param_groups[0]["lr"], "time_s": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"epoch {epoch}: train_loss={rec['train_loss']:.4f}  val_mean_AUROC={val_auc:.4f}  ({rec['time_s']}s)")

        ckpt = {"model": model.state_dict(), "config": cfg, "classes": CLASSES,
                "epoch": epoch, "val_mean_auroc": val_auc}
        torch.save(ckpt, out_dir / "last_model.pt")
        improved = not np.isnan(val_auc) and val_auc > best_auc
        if improved or best_auc < 0:
            best_auc = val_auc if not np.isnan(val_auc) else best_auc
            torch.save(ckpt, out_dir / "best_model.pt")
            bad_epochs = 0
            print("  saved best_model.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                print(f"early stopping: no improvement for {cfg['patience']} epochs")
                break

        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"done. best val mean AUROC = {best_auc:.4f}")
    return {"best_val_auroc": best_auc, "history": history}


def main(argv=None) -> None:
    train(load_config(argv))


if __name__ == "__main__":
    main()
