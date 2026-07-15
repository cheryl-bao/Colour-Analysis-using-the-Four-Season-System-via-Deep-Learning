"""Offline preprocessing pipeline: crop-to-mask, illumination normalization.

Usage:
    python -m src.preprocessing [--limit N] [--force]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src import config


def load_image(path):
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"could not read mask: {path}")
    return mask


def crop_to_mask(image, mask, padding_frac=config.CROP_PADDING_FRAC):
    """Crop `image` and `mask` to the bounding box of `mask`'s foreground, plus padding.

    `mask` must already be the same height/width as `image`. Returns
    (cropped_image, cropped_mask, status) where status is "cropped" or
    "mask_empty" (the mask has no foreground pixels, so both are returned
    unchanged).
    """
    mask_bin = mask > config.MASK_BINARY_THRESHOLD
    rows = np.any(mask_bin, axis=1)
    cols = np.any(mask_bin, axis=0)
    if not rows.any() or not cols.any():
        return image, mask, "mask_empty"

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    h, w = rmax - rmin + 1, cmax - cmin + 1
    pad_h, pad_w = int(h * padding_frac), int(w * padding_frac)

    height, width = image.shape[:2]
    r0, r1 = max(0, rmin - pad_h), min(height, rmax + 1 + pad_h)
    c0, c1 = max(0, cmin - pad_w), min(width, cmax + 1 + pad_w)

    return image[r0:r1, c0:c1], mask[r0:r1, c0:c1], "cropped"


def gray_world_lab_normalize(image_rgb, mask=None):
    """Neutralize lighting-driven colour cast via gray-world correction in LAB space.

    Shifts the LAB a/b channels so their mean is neutral (128), preserving
    luminance (L). If `mask` is given, statistics are computed over
    foreground-only pixels so background doesn't skew the correction.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    if mask is not None:
        fg = mask > config.MASK_BINARY_THRESHOLD
        if fg.any():
            a_mean, b_mean = lab[..., 1][fg].mean(), lab[..., 2][fg].mean()
        else:
            a_mean, b_mean = lab[..., 1].mean(), lab[..., 2].mean()
    else:
        a_mean, b_mean = lab[..., 1].mean(), lab[..., 2].mean()

    lab[..., 1] = np.clip(lab[..., 1] + (128 - a_mean), 0, 255)
    lab[..., 2] = np.clip(lab[..., 2] + (128 - b_mean), 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def process_one_row(row, raw_root, processed_root):
    """Run the crop -> illumination-normalize pipeline on one row.

    Never raises: any failure is captured in the returned dict's "error" field.
    """
    try:
        rgb_path = raw_root / row["path_rgb_original"]
        mask_path = raw_root / row["path_mask"]

        image = load_image(rgb_path)

        if mask_path.exists():
            mask = load_mask(mask_path)
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(
                    mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
                )
            cropped, cropped_mask, crop_status = crop_to_mask(image, mask)
            mask_for_stats = cropped_mask if crop_status == "cropped" else None
        else:
            cropped, crop_status = image, "mask_missing"
            mask_for_stats = None

        normalized = gray_world_lab_normalize(cropped, mask=mask_for_stats)

        out_rel = Path(row["path_rgb_original"]).with_suffix(".png")
        out_path = processed_root / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), cv2.cvtColor(normalized, cv2.COLOR_RGB2BGR))

        return {
            "path_processed": str(out_rel),
            "crop_status": crop_status,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - one bad row must not kill the batch
        return {"path_processed": None, "crop_status": "error", "error": str(exc)}


def _load_previous_results(processed_csv_path):
    if not processed_csv_path.exists():
        return None
    prev = pd.read_csv(processed_csv_path)
    return prev.set_index("path_rgb_original")


def compute_normalization_stats(df, processed_root):
    """Per-channel mean/std over the processed train partition, saved to JSON."""
    train_df = df[(df["partition"] == "train") & df["path_processed"].notna()]
    if train_df.empty:
        return None

    sum_, sum_sq, count = np.zeros(3), np.zeros(3), 0
    for path_processed in train_df["path_processed"]:
        img = load_image(processed_root / path_processed).astype(np.float64) / 255.0
        sum_ += img.reshape(-1, 3).sum(axis=0)
        sum_sq += (img.reshape(-1, 3) ** 2).sum(axis=0)
        count += img.shape[0] * img.shape[1]

    mean = sum_ / count
    std = np.sqrt(sum_sq / count - mean**2)

    stats = {"mean": mean.tolist(), "std": std.tolist()}
    config.NORM_STATS_PATH.write_text(json.dumps(stats, indent=2))
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N rows")
    parser.add_argument(
        "--force", action="store_true",
        help="reprocess every row, ignoring cached path_processed results (e.g. after a "
        "pipeline change like dropping the resize step)",
    )
    args = parser.parse_args()

    annotations = pd.read_csv(config.ANNOTATIONS_RAW)
    if args.limit is not None:
        annotations = annotations.head(args.limit)

    previous = _load_previous_results(config.ANNOTATIONS_PROCESSED)

    results = []
    for i, (_, row) in enumerate(annotations.iterrows(), start=1):
        row = row.to_dict()

        prev_record = None
        if previous is not None and row["path_rgb_original"] in previous.index:
            prev_record = previous.loc[row["path_rgb_original"]]

        already_done = (
            not args.force
            and prev_record is not None
            and prev_record["crop_status"] == "cropped"
            and pd.notna(prev_record["path_processed"])
            and (config.PROCESSED_ROOT / prev_record["path_processed"]).exists()
        )

        if already_done:
            result = {
                "path_processed": prev_record["path_processed"],
                "crop_status": prev_record["crop_status"],
                "error": None,
            }
        else:
            result = process_one_row(row, config.RAW_ROOT, config.PROCESSED_ROOT)

        results.append({**row, **result})

        if i % 500 == 0 or i == len(annotations):
            print(f"processed {i}/{len(annotations)}")

    out_df = pd.DataFrame(results)
    config.PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

    # A --limit run only processes a head-N subset of annotations -- writing
    # that to the real annotations_processed.csv would clobber the full
    # dataset's accumulated results. Smoke tests get their own file instead,
    # mirroring baseline-model/svm_baseline.py's results_smoketest_*.json.
    if args.limit is not None:
        annotations_out_path = config.ANNOTATIONS_PROCESSED.with_name(
            "annotations_processed_smoketest.csv"
        )
        missing_out_path = config.MISSING_MASKS_REPORT.with_name(
            "missing_masks_report_smoketest.csv"
        )
    else:
        annotations_out_path = config.ANNOTATIONS_PROCESSED
        missing_out_path = config.MISSING_MASKS_REPORT

    out_df.to_csv(annotations_out_path, index=False)

    missing = out_df[out_df["crop_status"] == "mask_missing"]
    missing.to_csv(missing_out_path, index=False)

    print(f"\nwrote {annotations_out_path}")
    print("crop_status summary:")
    print(out_df["crop_status"].value_counts().to_string())

    if args.limit is None:
        stats = compute_normalization_stats(out_df, config.PROCESSED_ROOT)
        if stats:
            print(f"\nnormalization stats written to {config.NORM_STATS_PATH}: {stats}")


if __name__ == "__main__":
    main()
