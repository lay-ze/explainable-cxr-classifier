"""Metrics, per-class threshold tuning, and evaluation plots.

CLI usage (after training):
    python -m src.evaluate --checkpoint outputs/run/best_model.pt --data-dir /path/to/nih

Writes to the checkpoint's folder:
    thresholds.json   per-class decision thresholds tuned on the validation split
    metrics_test.json per-class AUROC / AUPRC / F1 / sensitivity / specificity on the test split
    roc_curves.png, pr_curves.png
    test_predictions.npz (probabilities + labels, for later analysis)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from .labels import CLASSES


# --------------------------------------------------------------------------- inference
@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (probabilities, labels), both (N, 14)."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


# --------------------------------------------------------------------------- metrics
def _safe(fn, y, p) -> float:
    # AUROC/AUPRC are undefined when a class has no positives (or no negatives) in the split.
    if y.min() == y.max():
        return float("nan")
    return float(fn(y, p))


def per_class_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    return np.array([_safe(roc_auc_score, y_true[:, i], y_prob[:, i]) for i in range(y_true.shape[1])])


def mean_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    aucs = per_class_auroc(y_true, y_prob)
    return float(np.nanmean(aucs)) if not np.all(np.isnan(aucs)) else float("nan")


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Per-class threshold that maximises F1 on the given (validation) data.

    With heavy imbalance the optimal cut-off for a rare class is far below 0.5, so a single
    global 0.5 threshold would almost never predict it.
    """
    thresholds = np.full(y_true.shape[1], 0.5)
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() == 0:
            continue
        prec, rec, thr = precision_recall_curve(y_true[:, i], y_prob[:, i])
        f1 = 2 * prec * rec / np.clip(prec + rec, 1e-8, None)
        best = int(np.argmax(f1[:-1])) if len(thr) else 0  # last P/R point has no threshold
        if len(thr):
            thresholds[i] = float(thr[best])
    return thresholds


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray | None = None) -> dict:
    if thresholds is None:
        thresholds = np.full(y_true.shape[1], 0.5)
    per_class = {}
    for i, name in enumerate(CLASSES):
        y, p = y_true[:, i], y_prob[:, i]
        pred = (p >= thresholds[i]).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        sens = tp / (tp + fn) if tp + fn else float("nan")
        spec = tn / (tn + fp) if tn + fp else float("nan")
        prec = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) and not math.isnan(sens) else 0.0
        per_class[name] = {
            "prevalence": float(y.mean()),
            "n_positive": int(y.sum()),
            "auroc": _safe(roc_auc_score, y, p),
            "auprc": _safe(average_precision_score, y, p),
            "threshold": float(thresholds[i]),
            "f1": float(f1),
            "sensitivity": float(sens),
            "specificity": float(spec),
        }
    vals = lambda k: [v[k] for v in per_class.values() if not math.isnan(v[k])]
    summary = {
        "mean_auroc": float(np.mean(vals("auroc"))) if vals("auroc") else float("nan"),
        "mean_auprc": float(np.mean(vals("auprc"))) if vals("auprc") else float("nan"),
        "mean_f1": float(np.mean([v["f1"] for v in per_class.values()])),
    }
    return {"summary": summary, "per_class": per_class}


def format_table(metrics: dict) -> str:
    lines = [f"{'class':<20}{'prev':>7}{'AUROC':>8}{'AUPRC':>8}{'F1':>7}{'sens':>7}{'spec':>7}"]
    for name, m in metrics["per_class"].items():
        lines.append(f"{name:<20}{m['prevalence']:>7.3f}{m['auroc']:>8.3f}{m['auprc']:>8.3f}"
                     f"{m['f1']:>7.3f}{m['sensitivity']:>7.3f}{m['specificity']:>7.3f}")
    s = metrics["summary"]
    lines.append(f"{'MEAN':<20}{'':>7}{s['mean_auroc']:>8.3f}{s['mean_auprc']:>8.3f}{s['mean_f1']:>7.3f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- plots
def plot_curves(y_true: np.ndarray, y_prob: np.ndarray, out_dir: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    for kind in ("roc", "pr"):
        fig, axes = plt.subplots(3, 5, figsize=(18, 11))
        for i, ax in enumerate(axes.flat):
            if i >= len(CLASSES):
                ax.axis("off")
                continue
            y, p = y_true[:, i], y_prob[:, i]
            ax.set_title(CLASSES[i], fontsize=10)
            if y.min() == y.max():
                ax.text(0.5, 0.5, "no positives", ha="center")
                continue
            if kind == "roc":
                fpr, tpr, _ = roc_curve(y, p)
                ax.plot(fpr, tpr, lw=1.5)
                ax.plot([0, 1], [0, 1], "k--", lw=0.7)
                ax.set_xlabel("FPR", fontsize=8); ax.set_ylabel("TPR", fontsize=8)
                ax.text(0.55, 0.08, f"AUC={roc_auc_score(y, p):.3f}", fontsize=9)
            else:
                prec, rec, _ = precision_recall_curve(y, p)
                ax.plot(rec, prec, lw=1.5)
                ax.axhline(y.mean(), color="k", ls="--", lw=0.7)  # random-classifier baseline
                ax.set_xlabel("Recall", fontsize=8); ax.set_ylabel("Precision", fontsize=8)
                ax.text(0.45, 0.9, f"AP={average_precision_score(y, p):.3f}", fontsize=9)
        fig.suptitle("ROC curves (test)" if kind == "roc" else "Precision-Recall curves (test)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{kind}_curves.png", dpi=120)
        plt.close(fig)


# --------------------------------------------------------------------------- CLI
def main() -> None:
    from .dataset import CXRDataset, build_transforms, load_metadata, make_splits
    from .model import load_checkpoint

    ap = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the NIH test split")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.checkpoint, device)
    cfg = ckpt["config"]
    out_dir = Path(args.checkpoint).parent

    df = load_metadata(args.data_dir)
    _, val_df, test_df = make_splits(df, args.data_dir, cfg["val_frac"], cfg["seed"])
    tf = build_transforms(cfg["img_size"], train=False)
    mk = lambda d: DataLoader(CXRDataset(d, tf), batch_size=args.batch_size, num_workers=args.num_workers)

    print("Tuning thresholds on validation split ...")
    val_prob, val_true = predict(model, mk(val_df), device)
    thresholds = tune_thresholds(val_true, val_prob)
    (out_dir / "thresholds.json").write_text(json.dumps(dict(zip(CLASSES, thresholds.tolist())), indent=2))

    print("Evaluating on test split ...")
    test_prob, test_true = predict(model, mk(test_df), device)
    metrics = compute_metrics(test_true, test_prob, thresholds)
    (out_dir / "metrics_test.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(out_dir / "test_predictions.npz", probs=test_prob, labels=test_true,
                        images=test_df["image"].values)
    plot_curves(test_true, test_prob, out_dir)
    print(format_table(metrics))
    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()
