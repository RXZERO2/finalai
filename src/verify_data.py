"""
Dataset Verification Script (`src/verify_data.py`) for 3-Class Coffee Roast Level Classification:
  - Index 0: Dark
  - Index 1: Light
  - Index 2: Medium
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict

from torchvision import datasets

EXPECTED_CLASS_TO_IDX: Dict[str, int] = {
    "Dark": 0,
    "Light": 1,
    "Medium": 2,
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def count_images_in_dir(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for p in directory.rglob("*") if is_image_file(p))


def remove_excluded_classes(data_dir: Path) -> None:
    """Ensure any non-target class folders (such as 'Green') are removed from train/val/test."""
    for split in ("train", "val", "valid", "test"):
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for child in split_dir.iterdir():
            if child.is_dir() and child.name not in EXPECTED_CLASS_TO_IDX:
                print(f"[INFO] Removing excluded class directory: '{child}'")
                shutil.rmtree(child, ignore_errors=True)


def verify_dataset(data_dir: Path = Path("data")) -> bool:
    """
    Verify directory structure, count images per split/class, and assert the 3-class mapping:
    `{'Dark': 0, 'Light': 1, 'Medium': 2}`.
    """
    remove_excluded_classes(data_dir)

    print("=" * 64)
    print("COFFEE ROAST DATASET VERIFICATION REPORT (3 CLASSES)")
    print("=" * 64)
    print(f"Dataset Root : {data_dir.resolve()}")
    print(f"Target Map   : {EXPECTED_CLASS_TO_IDX}\n")

    splits = ["train", "val", "test"]
    split_totals: Dict[str, int] = {}
    all_valid = True

    header = f"{'Split':<10} | {'Dark (0)':>10} | {'Light (1)':>10} | {'Medium (2)':>10} | {'Total':>10}"
    print(header)
    print("-" * len(header))

    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists() and split == "val" and (data_dir / "valid").exists():
            split_dir = data_dir / "valid"

        counts: Dict[str, int] = {}
        for cls_name in EXPECTED_CLASS_TO_IDX:
            cls_dir = split_dir / cls_name
            counts[cls_name] = count_images_in_dir(cls_dir)
            if counts[cls_name] == 0:
                all_valid = False

        total_split = sum(counts.values())
        split_totals[split] = total_split
        print(
            f"{split:<10} | {counts['Dark']:>10} | "
            f"{counts['Light']:>10} | {counts['Medium']:>10} | {total_split:>10}"
        )

    print("-" * len(header))
    print(f"{'TOTAL':<10} | {'':>10} | {'':>10} | {'':>10} | {sum(split_totals.values()):>10}\n")

    # Verify torchvision ImageFolder class_to_idx on each split
    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists() and split == "val" and (data_dir / "valid").exists():
            split_dir = data_dir / "valid"

        if not split_dir.exists() or count_images_in_dir(split_dir) == 0:
            print(f"[FAIL] Split '{split}' is missing or empty at '{split_dir}'.")
            all_valid = False
            continue

        ds = datasets.ImageFolder(root=str(split_dir))
        actual_map = ds.class_to_idx
        if actual_map == EXPECTED_CLASS_TO_IDX:
            print(f"[PASS] '{split}' class_to_idx matches: {actual_map}")
        else:
            print(
                f"[FAIL] '{split}' class_to_idx mismatch!\n"
                f"       Expected: {EXPECTED_CLASS_TO_IDX}\n"
                f"       Actual  : {actual_map}"
            )
            all_valid = False

    print("=" * 64)
    if all_valid:
        print("STATUS: SUCCESS — All splits and 3-class mappings verified!")
    else:
        print("STATUS: WARNING — One or more splits/classes are empty or mismatched.")
    print("=" * 64)
    return all_valid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify 3-class Coffee Roast dataset (Dark, Light, Medium).")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Root data directory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    verify_dataset(data_dir=args.data_dir)
