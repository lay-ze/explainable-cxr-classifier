"""Statistics for the report, computed from the files that evaluate.py and localization.py save.

    python scripts/analyse_results.py
    python scripts/analyse_results.py --runs outputs/densenet121_bce outputs/densenet121_wbce --main outputs/densenet121_wbce

1. Loss comparison. Mean test AUROC of each run with a 95% confidence interval, and the paired
   difference against the first run. Patients are resampled, not images, because most test
   patients have several X-rays and treating those as independent would make the intervals too narrow.
2. Heatmap check split by whether the model flagged the disease (probability >= its tuned
   threshold). A heatmap for a disease the model did not detect is not evidence of a shortcut,
   so the flagged boxes with almost no heat inside are saved as shortcut_candidates.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

RUNS = ["outputs/densenet121_bce", "outputs/densenet121_wbce", "outputs/densenet121_focal"]


def load_run(run: Path):
    d = np.load(run / "test_predictions.npz", allow_pickle=True)  # 'images' holds the file names
    return d["probs"], d["labels"], d["images"]


def mean_auroc(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean([roc_auc_score(y[:, c], p[:, c]) for c in range(y.shape[1])]))


def compare_losses(runs: list[Path], n_boot: int, seed: int) -> None:
    names = [r.name for r in runs]
    data = [load_run(r) for r in runs]
    _, y, images = data[0]
    for _, y2, im2 in data[1:]:
        assert (im2 == images).all() and (y2 == y).all(), "runs were evaluated on different test images"
    probs = [p for p, _, _ in data]

    patient = np.array([s[:8] for s in images])  # NIH file names start with the 8-digit patient ID
    _, inv = np.unique(patient, return_inverse=True)
    groups = [np.flatnonzero(inv == g) for g in range(inv.max() + 1)]
    rare = int(np.argmin(y.sum(0)))  # rarest class in the test set (Hernia)
    print(f"Test set: {len(images)} images from {len(groups)} patients; {n_boot} patient-level resamples")

    rng = np.random.default_rng(seed)
    auc = {n: [] for n in names}
    d_auc = {n: [] for n in names[1:]}
    d_rare_ap = {n: [] for n in names[1:]}
    for _ in range(n_boot):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        yb = y[idx]
        if (yb.sum(0) == 0).any():  # a class with no positives has no AUROC
            continue
        m = [mean_auroc(yb, p[idx]) for p in probs]
        base_ap = average_precision_score(yb[:, rare], probs[0][idx, rare])
        for i, n in enumerate(names):
            auc[n].append(m[i])
            if i:
                d_auc[n].append(m[i] - m[0])
                d_rare_ap[n].append(average_precision_score(yb[:, rare], probs[i][idx, rare]) - base_ap)

    lo_hi = lambda v: np.percentile(v, [2.5, 97.5])
    print(f"\n{'run':22s} {'mean AUROC':>10s}   95% CI")
    for n, p in zip(names, probs):
        lo, hi = lo_hi(auc[n])
        print(f"{n:22s} {mean_auroc(y, p):10.4f}   {lo:.4f} to {hi:.4f}")
    print(f"\nDifference from {names[0]} (an interval containing 0 means no real difference):")
    for i, n in enumerate(names[1:], 1):
        lo, hi = lo_hi(d_auc[n])
        print(f"  {n:20s} mean AUROC {mean_auroc(y, probs[i]) - mean_auroc(y, probs[0]):+.4f}   95% CI {lo:+.4f} to {hi:+.4f}")
    n_rare = len(set(patient[y[:, rare] == 1]))
    print(f"\nRarest class ({int(y[:, rare].sum())} images from {n_rare} patients), AUPRC difference from {names[0]}:")
    for i, n in enumerate(names[1:], 1):
        lo, hi = lo_hi(d_rare_ap[n])
        diff = average_precision_score(y[:, rare], probs[i][:, rare]) - average_precision_score(y[:, rare], probs[0][:, rare])
        print(f"  {n:20s} {diff:+.4f}   95% CI {lo:+.4f} to {hi:+.4f}")


def split_localisation(run: Path) -> None:
    thr = json.loads((run / "thresholds.json").read_text())
    for csv in sorted(run.glob("localization_*/per_box.csv")):
        d = pd.read_csv(csv)
        d["flagged"] = d.prob >= d.label.map(thr)
        f, m = d[d.flagged], d[~d.flagged]
        t = pd.DataFrame({
            "boxes": d.groupby("label").size(),
            "flagged": d.groupby("label").flagged.mean(),
            "pointing_all": d.groupby("label").pointing_hit.mean(),
            "pointing_flagged": f.groupby("label").pointing_hit.mean(),
            "pointing_missed": m.groupby("label").pointing_hit.mean(),
            "energy_in_box": d.groupby("label").energy_in_box.mean(),
            "random_baseline": d.groupby("label").box_fraction.mean(),
        })
        t.loc["ALL"] = [len(d), d.flagged.mean(), d.pointing_hit.mean(), f.pointing_hit.mean(),
                        m.pointing_hit.mean(), d.energy_in_box.mean(), d.box_fraction.mean()]
        print(f"\n=== {csv.parent.name}  (flagged = probability >= tuned threshold)")
        print(t.round(3).to_string())

        outside = f[f.energy_in_box < 0.01].sort_values("prob", ascending=False)
        out = csv.parent / "shortcut_candidates.csv"
        outside.drop(columns="flagged").to_csv(out, index=False)
        counts = ", ".join(f"{k} {v}" for k, v in outside.label.value_counts().items())
        print(f"Flagged with almost no heat inside the box: {len(outside)} of {len(f)} ({counts}) -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", default=RUNS, help="run folders; the first is the reference")
    ap.add_argument("--main", default="outputs/densenet121_wbce", help="run whose heatmap check to split")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    compare_losses([Path(r) for r in a.runs], a.n_boot, a.seed)
    split_localisation(Path(a.main))


if __name__ == "__main__":
    main()
