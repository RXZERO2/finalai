"""
Dataset Extraction, Structure Normalization & Verification Script (`src/verify_data.py`).

Responsibilities:
1. If `archive.zip` is present (in project root or specified via `--zip-path`) and `data/` is not yet
   populated, extracts and organizes it into `data/train`, `data/val`, and `data/test`.
2. Handles common Kaggle Coffee Bean dataset layouts (e.g., nested folders, `valid/` vs `val/`, or
   single split requiring stratified train/val/test split if only `train`/`test` exist).
3. Counts images in each split (`train`, `val`/`valid`, `test`) across all 4 classes.
4. Verifies that `ImageFolder` produces the exact expected class-to-index mapping:
   `{'Dark': 0, 'Green': 1, 'Light': 2, 'Medium': 3}`.
"""

from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from torchvision import datasets

EXPECTED_CLASS_TO_IDX: Dict[str, int] = {
    "Dark": 0,
    "Green": 1,
    "Light": 2,
    "Medium": 3,
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def count_images_in_dir(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for p in directory.rglob("*") if is_image_file(p))


def find_class_root_dirs(search_root: Path) -> List[Path]:
    """Find any directory under `search_root` that contains Dark, Green, Light, Medium subfolders."""
    matches: List[Path] = []
    expected_set = set(EXPECTED_CLASS_TO_IDX.keys())
    for candidate in [search_root] + [p for p in search_root.rglob("*") if p.is_dir()]:
        child_names = {c.name for c in candidate.iterdir() if c.is_dir()}
        if expected_set.issubset(child_names):
            matches.append(candidate)
    return matches


def normalize_extracted_structure(data_dir: Path, seed: int = 42) -> None:
    """
    Inspect `data/` (including `data/_extracted` if present) and organize images into
    `data/train/{Dark,Green,Light,Medium}`, `data/val/{Dark,Green,Light,Medium}`,
    and `data/test/{Dark,Green,Light,Medium}`.
    """
    extracted_dir = data_dir / "_extracted"
    search_root = extracted_dir if extracted_dir.exists() else data_dir

    # Locate directories containing {Dark, Green, Light, Medium}
    class_roots = find_class_root_dirs(search_root)
    if not class_roots and search_root != data_dir:
        class_roots = find_class_root_dirs(data_dir)

    if not class_roots:
        return

    # Map detected split folders (train, val/valid, test)
    split_sources: Dict[str, Path] = {}
    unassigned_roots: List[Path] = []

    for root in class_roots:
        name_lower = root.name.lower()
        if name_lower in ("train", "training"):
            split_sources["train"] = root
        elif name_lower in ("val", "valid", "validation"):
            split_sources["val"] = root
        elif name_lower in ("test", "testing", "eval"):
            split_sources["test"] = root
        else:
            unassigned_roots.append(root)

    # Copy/move files from _extracted (or non-standard roots) into standard data/{train,val,test}
    if split_sources:
        for split_name, src_split_dir in split_sources.items():
            dst_split_dir = data_dir / split_name
            if src_split_dir.resolve() == dst_split_dir.resolve():
                continue
            for cls_name in EXPECTED_CLASS_TO_IDX:
                src_cls = src_split_dir / cls_name
                dst_cls = dst_split_dir / cls_name
                dst_cls.mkdir(parents=True, exist_ok=True)
                if src_cls.exists():
                    for img_file in src_cls.iterdir():
                        if is_image_file(img_file):
                            shutil.move(str(img_file), str(dst_cls / img_file.name))

    elif len(unassigned_roots) == 1:
        # Single flat root containing Dark, Green, Light, Medium -> split 70% train / 15% val / 15% test
        flat_root = unassigned_roots[0]
        rng = random.Random(seed)
        for cls_name in EXPECTED_CLASS_TO_IDX:
            src_cls = flat_root / cls_name
            images = sorted([p for p in src_cls.iterdir() if is_image_file(p)])
            rng.shuffle(images)
            n = len(images)
            n_train = int(0.70 * n)
            n_val = int(0.15 * n)
            splits = {
                "train": images[:n_train],
                "val": images[n_train : n_train + n_val],
                "test": images[n_train + n_val :],
            }
            for split_name, split_files in splits.items():
                dst_cls = data_dir / split_name / cls_name
                dst_cls.mkdir(parents=True, exist_ok=True)
                for img_file in split_files:
                    shutil.move(str(img_file), str(dst_cls / img_file.name))

    # If `valid` exists directly in `data/` and `data/val` is empty, mirror/move to `data/val`
    valid_dir = data_dir / "valid"
    val_dir = data_dir / "val"
    if valid_dir.exists() and count_images_in_dir(val_dir) == 0:
        for cls_name in EXPECTED_CLASS_TO_IDX:
            src_cls = valid_dir / cls_name
            dst_cls = val_dir / cls_name
            dst_cls.mkdir(parents=True, exist_ok=True)
            if src_cls.exists():
                for img_file in src_cls.iterdir():
                    if is_image_file(img_file):
                        shutil.move(str(img_file), str(dst_cls / img_file.name))
        shutil.rmtree(valid_dir, ignore_errors=True)

    # If dataset only had `train` and `test` (common in Kaggle Coffee Bean dataset where train=1200, test=400),
    # and `val` has 0 images, carve out a stratified 15% validation split from `train` so `train`, `val`, `test` are all populated.
    train_dir = data_dir / "train"
    if count_images_in_dir(train_dir) > 0 and count_images_in_dir(val_dir) == 0:
        print(
            "[INFO] No separate 'val'/'valid' folder found in archive; "
            "carving out 15% of 'train' per class into 'data/val/' for validation..."
        )
        rng = random.Random(seed)
        for cls_name in EXPECTED_CLASS_TO_IDX:
            src_cls = train_dir / cls_name
            dst_cls = val_dir / cls_name
            dst_cls.mkdir(parents=True, exist_ok=True)
            images = sorted([p for p in src_cls.iterdir() if is_image_file(p)])
            rng.shuffle(images)
            n_val = max(1, int(0.15 * len(images)))
            for img_file in images[:n_val]:
                shutil.move(str(img_file), str(dst_cls / img_file.name))

    # Clean up temporary `_extracted` folder if present
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir, ignore_errors=True)


def extract_if_needed(data_dir: Path, zip_path: Optional[Path] = None) -> None:
    """Extract zip archive if provided or found in project root and `data/train` is empty."""
    train_count = count_images_in_dir(data_dir / "train")
    extracted_dir = data_dir / "_extracted"

    if extracted_dir.exists():
        normalize_extracted_structure(data_dir)
        return

    if train_count > 0 and (data_dir / "train" / "Green").exists():
        normalize_extracted_structure(data_dir)
        return

    candidates = [
        zip_path,
        Path("archive.zip"),
        Path.home() / "Downloads" / "archive.zip",
    ]
    for cand in candidates:
        if cand is not None and cand.exists():
            print(f"[INFO] Extracting '{cand}' into '{extracted_dir}'...")
            extracted_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(cand, "r") as zf:
                zf.extractall(extracted_dir)
            normalize_extracted_structure(data_dir)
            break


def verify_dataset(data_dir: Path = Path("data"), zip_path: Optional[Path] = None) -> bool:
    """
    Verify directory structure, count images per split/class, and assert class_to_idx mapping.
    """
    extract_if_needed(data_dir, zip_path=zip_path)

    print("=" * 68)
    print("COFFEE BEAN DATASET VERIFICATION REPORT (4 CLASSES)")
    print("=" * 68)
    print(f"Dataset Root : {data_dir.resolve()}")
    print(f"Target Map   : {EXPECTED_CLASS_TO_IDX}\n")

    splits = ["train", "val", "test"]
    split_totals: Dict[str, int] = {}
    all_valid = True

    header = f"{'Split':<10} | {'Dark (0)':>10} | {'Green (1)':>10} | {'Light (2)':>10} | {'Medium (3)':>10} | {'Total':>10}"
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
            f"{split:<10} | {counts['Dark']:>10} | {counts['Green']:>10} | "
            f"{counts['Light']:>10} | {counts['Medium']:>10} | {total_split:>10}"
        )

    print("-" * len(header))
    print(f"{'TOTAL':<10} | {'':>10} | {'':>10} | {'':>10} | {'':>10} | {sum(split_totals.values()):>10}\n")

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

    print("=" * 68)
    if all_valid:
        print("STATUS: SUCCESS — All splits and 4-class mappings verified!")
    else:
        print("STATUS: WARNING — One or more splits/classes are empty or mismatched.")
    print("=" * 68)
    return all_valid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify 4-class Coffee Roast dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Root data directory.")
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="Optional path to archive.zip to extract before verifying.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    verify_dataset(data_dir=args.data_dir, zip_path=args.zip_path)

