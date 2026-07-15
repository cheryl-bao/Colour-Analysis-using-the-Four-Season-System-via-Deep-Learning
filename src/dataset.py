import json

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src import config


def load_normalization_stats():
    if config.NORM_STATS_PATH.exists():
        stats = json.loads(config.NORM_STATS_PATH.read_text())
        return stats["mean"], stats["std"]
    return config.IMAGENET_MEAN, config.IMAGENET_STD


def get_default_transform(img_size=config.IMG_SIZE):
    """Resize -> tensor -> normalize. Cached files under data/processed/ are
    left at their cropped (variable) resolution -- resizing to a fixed,
    consumer-chosen size happens here instead, so e.g. the CNN (224) and the
    SVM baseline (64) can each use their own resolution off the same cache."""
    mean, std = load_normalization_stats()
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class SeasonDataset(Dataset):
    def __init__(
        self,
        csv_path=config.ANNOTATIONS_PROCESSED,
        processed_root=config.PROCESSED_ROOT,
        partition=None,
        img_size=config.IMG_SIZE,
        transform=None,
    ):
        df = pd.read_csv(csv_path)
        df = df[df["path_processed"].notna()]
        if partition is not None:
            df = df[df["partition"] == partition]

        self.df = df.reset_index(drop=True)
        self.processed_root = processed_root
        self.transform = transform or get_default_transform(img_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self.processed_root / row["path_processed"]).convert("RGB")
        image = self.transform(image)
        label = config.CLASS_TO_IDX[row["class"]]
        return image, torch.tensor(label, dtype=torch.long)
