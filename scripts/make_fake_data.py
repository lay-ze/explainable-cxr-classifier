"""Create a tiny fake dataset in the NIH ChestX-ray14 file layout, for testing the code.

    python scripts/make_fake_data.py data/fake_nih

No real X-rays: 400 noise images with random labels. "Mass" images get a bright square,
recorded in BBox_List_2017.csv, and "Cardiomegaly" images get a bright patch, so those two
classes are learnable and the heatmap check has something to find. Use it to confirm the
code runs on your machine; the scores mean nothing.
"""
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.labels import CLASSES  # noqa: E402


def main(root: Path, n: int = 400, size: int = 256) -> None:
    img_dir = root / "images_001" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng, nrng = random.Random(0), np.random.default_rng(0)
    rows, boxes, names = [], [], []
    for i in range(n):
        pid = i // 2 + 1  # two scans per patient, to exercise the patient-level split
        name = f"{pid:08d}_{i % 2:03d}.png"
        labels = [c for c in CLASSES if rng.random() < 0.12]
        if rng.random() < 0.35 and "Mass" not in labels:
            labels.append("Mass")
        img = nrng.normal(90, 20, (size, size)).clip(0, 255)
        if "Mass" in labels:
            x, y, w = rng.randint(20, size - 76), rng.randint(20, size - 76), 50
            img[y:y + w, x:x + w] = 230
            boxes.append(f"{name},Mass,{x},{y},{w},{w},,,\n")
        if "Cardiomegaly" in labels:
            img[110:170, 90:170] += 60
        Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(img_dir / name)
        rows.append(f"{name},{'|'.join(labels) or 'No Finding'},{i % 2},{pid},50,M,PA,{size},{size},0.14,0.14,\n")
        names.append(name)

    (root / "Data_Entry_2017.csv").write_text(
        "Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Gender,View Position,"
        "OriginalImage[Width,Height],OriginalImagePixelSpacing[x,y],\n" + "".join(rows))
    (root / "BBox_List_2017.csv").write_text("Image Index,Finding Label,Bbox [x,y,w,h],,,,,\n" + "".join(boxes))
    split = int(n * 0.8)
    (root / "train_val_list.txt").write_text("\n".join(names[:split]))
    (root / "test_list.txt").write_text("\n".join(names[split:]))
    print(f"Fake dataset written to {root}: {n} images ({size}px), {len(boxes)} boxes")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/make_fake_data.py <output folder>")
    main(Path(sys.argv[1]))
