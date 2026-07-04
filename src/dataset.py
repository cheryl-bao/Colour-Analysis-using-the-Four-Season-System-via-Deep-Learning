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


def get_default_transform():
    mean, std = load_normalization_stats()
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


class SeasonDataset(Dataset):
    def __init__(
        self,
        csv_path=config.ANNOTATIONS_PROCESSED,
        processed_root=config.PROCESSED_ROOT,
        partition=None,
        label_mode="4",
        transform=None,
    ):
        if label_mode not in ("4", "12"):
            raise ValueError(f"label_mode must be '4' or '12', got {label_mode!r}")

        df = pd.read_csv(csv_path)
        df = df[df["path_processed"].notna()]
        if partition is not None:
            df = df[df["partition"] == partition]

        self.df = df.reset_index(drop=True)
        self.processed_root = processed_root
        self.label_mode = label_mode
        self.transform = transform or get_default_transform()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self.processed_root / row["path_processed"]).convert("RGB")
        image = self.transform(image)

        if self.label_mode == "4":
            label = config.CLASS_TO_IDX[row["class"]]
        else:
            label = config.SUBCLASS_TO_IDX[(row["class"], row["sub_class"])]

        return image, torch.tensor(label, dtype=torch.long)
