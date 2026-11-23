"""Does the model look at the right place? Quantitative Grad-CAM check against radiologist boxes.

BBox_List_2017.csv has ~980 boxes for 8 conditions. For every box we compute the CAM of
that box's class and measure:

    pointing_hit   1 if the CAM's peak pixel falls inside the box (pointing game)
    energy_in_box  fraction of total CAM mass inside the box
    iou            IoU between the box and the CAM thresholded at `cam_threshold`
    box_fraction   box area / image area: the score a *random* heatmap would get for
                   both pointing and energy, so it is the baseline to beat

A class whose AUROC is high but whose energy_in_box is near box_fraction is a red flag:
the model may be classifying from something other than the lesion (a shortcut).

    python -m src.localization --checkpoint outputs/run/best_model.pt --data-dir /path/to/nih
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .dataset import index_images, load_image
from .explain import CAMExplainer, overlay, resize_cam
from .labels import BBOX_LABEL_ALIASES, CLASS_TO_IDX
from .model import load_checkpoint


def load_bboxes(data_dir: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(Path(data_dir) / "BBox_List_2017.csv")
    df = raw.iloc[:, :6].copy()  # the file has trailing empty columns
    df.columns = ["image", "label", "x", "y", "w", "h"]
    df["label"] = df["label"].replace(BBOX_LABEL_ALIASES)
    df = df[df["label"].isin(CLASS_TO_IDX)]
    paths = index_images(data_dir)
    df["path"] = df["image"].map(paths)
    return df.dropna(subset=["path"]).reset_index(drop=True)


def box_metrics(cam: np.ndarray, box: tuple[float, float, float, float], cam_threshold: float = 0.5) -> dict:
    """cam: (H, W) in [0, 1] at original resolution. box: x, y, w, h in the same pixel coords."""
    H, W = cam.shape
    x0, y0 = int(max(0, box[0])), int(max(0, box[1]))
    x1, y1 = int(min(W, box[0] + box[2])), int(min(H, box[1] + box[3]))
    mask = np.zeros_like(cam, dtype=bool)
    mask[y0:y1, x0:x1] = True

    py, px = np.unravel_index(np.argmax(cam), cam.shape)
    hot = cam >= cam_threshold
    inter, union = (hot & mask).sum(), (hot | mask).sum()
    return {
        "pointing_hit": float(mask[py, px]),
        "energy_in_box": float(cam[mask].sum() / (cam.sum() + 1e-8)),
        "iou": float(inter / union) if union else 0.0,
        "box_fraction": float(mask.mean()),
    }


def _draw_box(img, box, color=(0, 255, 0), width=4):
    from PIL import ImageDraw
    img = img.copy()
    x, y, w, h = box
    ImageDraw.Draw(img).rectangle([x, y, x + w, y + h], outline=color, width=width)
    return img


def _check_held_out(boxes: pd.DataFrame, data_dir: str | Path) -> None:
    """Box images should all be in the official test split; if any were trained on, say so."""
    trainval = Path(data_dir) / "train_val_list.txt"
    if not trainval.exists():
        return
    n_seen = int(boxes["image"].isin(set(trainval.read_text().split())).sum())
    if n_seen:
        print(f"WARNING: {n_seen}/{len(boxes)} box images are in the training split; "
              "localisation scores on them are optimistic")
    else:
        print(f"OK: all {len(boxes)} box images are in the held-out test split")


def run(checkpoint: str, data_dir: str, method: str = "gradcam", max_images: int = 0,
        n_examples: int = 6, cam_threshold: float = 0.5, bbox_ref_size: int = 1024) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(checkpoint, device)
    explainer = CAMExplainer(model, ckpt["config"]["img_size"], method, device)
    out_dir = Path(checkpoint).parent / f"localization_{method.replace('++', 'pp')}"
    out_dir.mkdir(exist_ok=True)

    boxes = load_bboxes(data_dir)
    if max_images:
        boxes = boxes.head(max_images)
    _check_held_out(boxes, data_dir)
    print(f"Evaluating {len(boxes)} boxes with {method} ...")

    rows = []
    for i, r in boxes.iterrows():
        img = load_image(r.path)
        # Box coordinates are given for 1024px images; rescale if these images were resized.
        s = img.width / bbox_ref_size
        box = (r.x * s, r.y * s, r.w * s, r.h * s)
        c = CLASS_TO_IDX[r.label]
        probs, cams = explainer.explain_image(img, [c])
        cam = resize_cam(cams[c], img.size)
        m = box_metrics(cam, box, cam_threshold)
        rows.append({"image": r.image, "label": r.label, "path": r.path, "box": box,
                     "prob": float(probs[c]), **m})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(boxes)}")
    res = pd.DataFrame(rows)
    res.drop(columns=["path", "box"]).to_csv(out_dir / "per_box.csv", index=False)

    summary = res.groupby("label")[["pointing_hit", "energy_in_box", "iou", "box_fraction", "prob"]].mean()
    summary["n_boxes"] = res.groupby("label").size()
    summary.loc["ALL"] = res[["pointing_hit", "energy_in_box", "iou", "box_fraction", "prob"]].mean().tolist() + [len(res)]
    summary.to_csv(out_dir / "summary.csv")
    (out_dir / "summary.json").write_text(json.dumps(summary.reset_index().to_dict(orient="records"), indent=2))
    print(summary.round(3).to_string())

    # Best and worst examples: worst = heatmap mostly outside the radiologist's box.
    ranked = res.sort_values("energy_in_box")
    for tag, subset in (("worst", ranked.head(n_examples)), ("best", ranked.tail(n_examples))):
        for _, r in subset.iterrows():
            img = load_image(r.path)
            c = CLASS_TO_IDX[r.label]
            _, cams = explainer.explain_image(img, [c])
            vis = _draw_box(overlay(img, cams[c]), r.box)
            vis.save(out_dir / f"{tag}_{r.label}_{Path(r.image).stem}_E{r.energy_in_box:.2f}.png")
    print(f"Saved results and example overlays to {out_dir}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Grad-CAM localisation vs NIH bounding boxes")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--method", default="gradcam", choices=["gradcam", "gradcam++"])
    ap.add_argument("--max-images", type=int, default=0, help="limit number of boxes (0 = all)")
    ap.add_argument("--n-examples", type=int, default=6)
    ap.add_argument("--cam-threshold", type=float, default=0.5)
    ap.add_argument("--bbox-ref-size", type=int, default=1024,
                    help="image size the box coordinates refer to (NIH: 1024)")
    a = ap.parse_args()
    run(a.checkpoint, a.data_dir, a.method, a.max_images, a.n_examples, a.cam_threshold, a.bbox_ref_size)


if __name__ == "__main__":
    main()
