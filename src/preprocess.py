"""Optional: pre-resize NIH images (1024x1024 PNG) to a smaller size once, before training.

Decoding 1024px PNGs is usually the training bottleneck on Kaggle/Colab (few CPU cores).
Resizing once to 256px makes every epoch several times faster and needs ~4-5 GB of disk.

    python -m src.preprocess --src /kaggle/input/data --dst /kaggle/temp/nih256 --size 256

Train with --data-dir pointing to the resized copy. src.localization rescales the 1024px box
coordinates automatically, but the original images give sharper heatmap overlays.
"""
from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from PIL import Image

META_FILES = ["Data_Entry_2017.csv", "BBox_List_2017.csv", "train_val_list.txt", "test_list.txt"]


def _resize_one(src: Path, dst_dir: Path, size: int) -> None:
    out = dst_dir / src.name
    if out.exists():
        return
    Image.open(src).convert("L").resize((size, size), Image.BILINEAR).save(out, optimize=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    img_dir = dst / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for f in META_FILES:
        if (src / f).exists():
            shutil.copy(src / f, dst / f)

    files = list(src.rglob("*.png"))
    print(f"Resizing {len(files)} images to {a.size}px -> {img_dir}")
    with ProcessPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(partial(_resize_one, dst_dir=img_dir, size=a.size), files, chunksize=256), 1):
            if i % 10000 == 0:
                print(f"  {i}/{len(files)}")
    print("done")


if __name__ == "__main__":
    main()
