"""Build an annotations.csv for a raw_root/<class>/<image> folder layout,
so it can be run through src.preprocessing like the main dataset.

No mask images are assumed to exist -- path_mask always points at a
guaranteed-missing path, so preprocessing's existing mask_missing fallback
kicks in (skips cropping, still applies colour normalization).

Usage:
    python -m src.build_folder_annotations RAW_ROOT OUT_CSV [--partition test]
"""

import argparse
from pathlib import Path

import pandas as pd

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def build_annotations(raw_root, partition):
    rows = []
    for class_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rel_path = image_path.relative_to(raw_root)
            rows.append({
                "class": class_dir.name,
                "path_rgb_original": str(rel_path),
                "path_mask": str(Path("NO_MASK") / rel_path),
                "partition": partition,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("out_csv", type=Path)
    parser.add_argument("--partition", default="test")
    args = parser.parse_args()

    df = build_annotations(args.raw_root, args.partition)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print(f"wrote {args.out_csv}")
    print(df["class"].value_counts().to_string())
    print(f"total: {len(df)}")


if __name__ == "__main__":
    main()
