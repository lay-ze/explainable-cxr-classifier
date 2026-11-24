"""Draw Grad-CAM pictures for a list of boxes (e.g. the shortcut candidates) and copy test X-rays for the demo.

    python scripts/render_cases.py --checkpoint trained/densenet121_wbce/best_model.pt \
        --data-dir /kaggle/input/data --cases trained/densenet121_wbce/shortcut_candidates.csv \
        --out-dir /kaggle/working/shortcut_cases

For each row of --cases (columns image, label, prob, energy_in_box, as written by
scripts/analyse_results.py) it saves the X-ray twice, side by side: on the left with only the
radiologist's box (green), so chest drains, wires and text markers are easy to see, and on the
right with the heatmap for that disease. overview.png shows all cases on one page.
With --demo-samples N it also copies N test X-rays per disease, plus N with no finding, into
demo_samples/ for trying the demo app; labels.csv there lists each image's real labels.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import load_image, load_metadata  # noqa: E402
from src.explain import CAMExplainer, overlay  # noqa: E402
from src.labels import CLASS_TO_IDX, CLASSES  # noqa: E402
from src.localization import _draw_box, load_bboxes  # noqa: E402
from src.model import load_checkpoint  # noqa: E402

DEMO_CLASSES = ["Cardiomegaly", "Effusion", "Pneumothorax", "Mass", "Atelectasis", "Emphysema"]


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _captioned(panels: list[Image.Image], caption: str, panel: int, font_size: int) -> Image.Image:
    bar = font_size * 2
    out = Image.new("RGB", (panel * len(panels), panel + bar), "white")
    for i, p in enumerate(panels):
        out.paste(p.convert("RGB").resize((panel, panel), Image.BILINEAR), (i * panel, bar))
    ImageDraw.Draw(out).text((8, font_size // 2), caption, fill="black", font=_font(font_size))
    return out


def render_cases(checkpoint: str, data_dir: str, cases_csv: str, out_dir: Path, method: str,
                 bbox_ref_size: int, cols: int = 4) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(checkpoint, device)
    explainer = CAMExplainer(model, ckpt["config"]["img_size"], method, device)
    thr = json.loads((Path(checkpoint).parent / "thresholds.json").read_text())
    cases = pd.read_csv(cases_csv).sort_values(["label", "prob"], ascending=[True, False])
    boxes = load_bboxes(data_dir)
    print(f"Drawing {len(cases)} cases with {method} on {device} ...")

    thumbs = []
    for i, r in enumerate(cases.itertuples(), 1):
        b = boxes[(boxes.image == r.image) & (boxes.label == r.label)]
        if b.empty:
            print(f"  no box found for {r.image} {r.label}, skipped")
            continue
        img = load_image(b.path.iloc[0])
        c = CLASS_TO_IDX[r.label]
        probs, cams = explainer.explain_image(img, [c])
        plain, heat = img, overlay(img, cams[c])
        s = img.width / bbox_ref_size
        for bb in b.itertuples():
            box = (bb.x * s, bb.y * s, bb.w * s, bb.h * s)
            plain = _draw_box(plain, box, width=max(2, round(4 * s)))
            heat = _draw_box(heat, box, width=max(2, round(4 * s)))
        p, t = float(probs[c]), thr[r.label]
        caption = (f"{r.label}   probability {p:.2f} (threshold {t:.2f})   "
                   f"heat inside box {r.energy_in_box:.0%}   {r.image}")
        _captioned([plain, heat], caption, panel=512, font_size=18).save(
            out_dir / f"{i:02d}_{r.label}_{Path(r.image).stem}.png")
        thumbs.append(_captioned([plain, heat], f"{i:02d} {r.label} p={p:.2f}", panel=256, font_size=14))

    if thumbs:
        w, h = thumbs[0].size
        rows = -(-len(thumbs) // cols)
        grid = Image.new("RGB", (cols * w, rows * h), "white")
        for k, t in enumerate(thumbs):
            grid.paste(t, ((k % cols) * w, (k // cols) * h))
        grid.save(out_dir / "overview.png")
        print(f"Saved {len(thumbs)} pictures and overview.png to {out_dir}")
    else:
        print("No pictures saved: none of the cases matched a box in this dataset")


def copy_demo_samples(data_dir: str, out_dir: Path, n: int) -> None:
    df = load_metadata(data_dir)
    test = set((Path(data_dir) / "test_list.txt").read_text().split())
    df = df[df["image"].isin(test)]
    groups = [(c, df[df[c] == 1]) for c in DEMO_CLASSES] + [("No_Finding", df[df[CLASSES].sum(axis=1) == 0])]
    picks = pd.concat([g.sample(min(n, len(g)), random_state=0).assign(group=name) for name, g in groups])
    picks = picks.drop_duplicates("image")
    dst = out_dir / "demo_samples"
    dst.mkdir(exist_ok=True)
    for r in picks.itertuples():
        shutil.copy(r.path, dst / f"{r.group}_{r.image}")
    picks[["group", "image", "findings"]].to_csv(dst / "labels.csv", index=False)
    print(f"Copied {len(picks)} test X-rays to {dst} (real labels in labels.csv)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True, help="original NIH images (1024px)")
    ap.add_argument("--cases", required=True, help="CSV with columns image, label, prob, energy_in_box")
    ap.add_argument("--out-dir", default="shortcut_cases")
    ap.add_argument("--method", default="gradcam", choices=["gradcam", "gradcam++"])
    ap.add_argument("--bbox-ref-size", type=int, default=1024, help="image size the box coordinates refer to")
    ap.add_argument("--demo-samples", type=int, default=2, help="test X-rays to copy per disease (0 = none)")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    render_cases(a.checkpoint, a.data_dir, a.cases, out, a.method, a.bbox_ref_size)
    if a.demo_samples:
        copy_demo_samples(a.data_dir, out, a.demo_samples)


if __name__ == "__main__":
    main()
