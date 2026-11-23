"""NIH ChestX-ray14 loading: metadata, patient-level splits, transforms and the Dataset class.

Expected data layout (this is how the Kaggle dataset `nih-chest-xrays/data` is organised):

    <data_dir>/Data_Entry_2017.csv
    <data_dir>/BBox_List_2017.csv
    <data_dir>/train_val_list.txt
    <data_dir>/test_list.txt
    <data_dir>/images_001/images/00000001_000.png
    ...
    <data_dir>/images_012/images/...

A flat `<data_dir>/images/*.png` folder also works: images are found by recursive search.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .labels import CLASSES, encode

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- metadata
def index_images(data_dir: str | Path) -> dict[str, str]:
    """Map image file name -> full path by searching data_dir recursively."""
    return {p.name: str(p) for p in Path(data_dir).rglob("*.png")}


def load_metadata(data_dir: str | Path) -> pd.DataFrame:
    """Read Data_Entry_2017.csv, attach image paths and the 14-dim label vectors."""
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "Data_Entry_2017.csv")
    df = df.rename(columns={"Image Index": "image", "Finding Labels": "findings", "Patient ID": "patient_id"})
    paths = index_images(data_dir)
    df["path"] = df["image"].map(paths)
    missing = df["path"].isna().sum()
    if missing:
        print(f"[dataset] warning: {missing} images listed in the CSV were not found on disk; dropping them")
        df = df.dropna(subset=["path"])
    labels = np.stack(df["findings"].map(encode).values)
    df[CLASSES] = labels
    return df.reset_index(drop=True)


def _read_list(path: Path) -> set[str]:
    return set(path.read_text().split())


def make_splits(df: pd.DataFrame, data_dir: str | Path, val_frac: float = 0.1, seed: int = 42):
    """Official train_val/test split, then a patient-level validation split out of train_val.

    Grouping by patient matters: NIH has many follow-up scans per patient, and if the same
    patient is in train and val the validation score is inflated.
    """
    data_dir = Path(data_dir)
    test_names = _read_list(data_dir / "test_list.txt")
    trainval_names = _read_list(data_dir / "train_val_list.txt")

    test_df = df[df["image"].isin(test_names)]
    trainval_df = df[df["image"].isin(trainval_names)]

    rng = np.random.default_rng(seed)
    patients = trainval_df["patient_id"].unique()
    rng.shuffle(patients)
    n_val = max(1, int(len(patients) * val_frac))
    val_patients = set(patients[:n_val])

    val_df = trainval_df[trainval_df["patient_id"].isin(val_patients)]
    train_df = trainval_df[~trainval_df["patient_id"].isin(val_patients)]

    check_no_patient_leak(train_df, val_df, test_df)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def check_no_patient_leak(*dfs: pd.DataFrame) -> None:
    sets = [set(d["patient_id"]) for d in dfs]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            assert not overlap, f"patient leakage between split {i} and {j}: {len(overlap)} patients"


def pos_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """neg/pos ratio per class, used as pos_weight in weighted BCE."""
    pos = train_df[CLASSES].sum().values.astype(np.float64)
    neg = len(train_df) - pos
    return torch.tensor(neg / np.clip(pos, 1, None), dtype=torch.float32)


# --------------------------------------------------------------------------- transforms
def build_transforms(img_size: int = 224, train: bool = True) -> T.Compose:
    """No horizontal flip: it would put the heart on the wrong side, which is itself a shortcut."""
    if train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_image(path: str) -> Image.Image:
    # NIH images are grayscale (some RGBA); pretrained backbones expect 3 channels.
    return Image.open(path).convert("RGB")


# --------------------------------------------------------------------------- dataset
class CXRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: T.Compose):
        self.paths = df["path"].tolist()
        self.labels = df[CLASSES].values.astype(np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = self.transform(load_image(self.paths[idx]))
        return img, torch.from_numpy(self.labels[idx])
