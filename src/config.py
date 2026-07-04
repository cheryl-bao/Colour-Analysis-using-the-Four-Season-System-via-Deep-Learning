from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

RAW_ROOT = REPO_ROOT / "data" / "raw"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
ANNOTATIONS_RAW = RAW_ROOT / "annotations.csv"
ANNOTATIONS_PROCESSED = PROCESSED_ROOT / "annotations_processed.csv"
MISSING_MASKS_REPORT = PROCESSED_ROOT / "missing_masks_report.csv"
NORM_STATS_PATH = PROCESSED_ROOT / "normalization_stats.json"

IMG_SIZE = 224
CROP_PADDING_FRAC = 0.15
MASK_BINARY_THRESHOLD = 127

CLASSES = ["autunno", "estate", "inverno", "primavera"]

# 12-way — explicit, ordered so label indices stay stable across runs.
SUBCLASS_COMBOS = [
    ("autunno", "deep"), ("autunno", "soft"), ("autunno", "warm"),
    ("estate", "cool"), ("estate", "light"), ("estate", "soft"),
    ("inverno", "bright"), ("inverno", "cool"), ("inverno", "deep"),
    ("primavera", "bright"), ("primavera", "light"), ("primavera", "warm"),
]

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
SUBCLASS_TO_IDX = {combo: i for i, combo in enumerate(SUBCLASS_COMBOS)}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
