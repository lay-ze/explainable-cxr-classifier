"""The 14 NIH ChestX-ray14 conditions and multi-hot label helpers."""
from __future__ import annotations

import numpy as np

CLASSES: list[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# BBox_List_2017.csv spells one label differently from Data_Entry_2017.csv.
BBOX_LABEL_ALIASES = {"Infiltrate": "Infiltration"}


def encode(finding_labels: str) -> np.ndarray:
    """'Effusion|Cardiomegaly' -> 14-dim multi-hot float vector. 'No Finding' -> all zeros."""
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    for name in finding_labels.split("|"):
        name = name.strip()
        if name in CLASS_TO_IDX:
            vec[CLASS_TO_IDX[name]] = 1.0
    return vec


def decode(vec, threshold: float = 0.5) -> list[str]:
    return [CLASSES[i] for i, v in enumerate(vec) if v >= threshold]
