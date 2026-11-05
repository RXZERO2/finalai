"""
Dataset and DataLoader utilities for 3-Class Coffee Roast Level Classification:
  - Index 0: Dark
  - Index 1: Light
  - Index 2: Medium
"""

from pathlib import Path
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = (224, 224)


def get_transforms() -> Dict[str, transforms.Compose]:
    """
    Build standard preprocessing and augmentation pipelines for 3-class classification:
    Resize(224, 224), Horizontal/Vertical Flip, ColorJitter, and ImageNet Normalization.
    """
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return {
        "train": train_transform,
        "val": eval_transform,
        "test": eval_transform,
    }


def create_dataloaders(
    data_dir: str | Path = "data",
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[Dict[str, DataLoader], Dict[int, str]]:
    """
    Load train, val (or valid), and test splits using torchvision.datasets.ImageFolder.
    Returns a dictionary of DataLoaders and an index-to-class mapping.
    """
    root = Path(data_dir)
    tfms = get_transforms()

    train_dir = root / "train"
    val_dir = root / "val" if (root / "val").exists() else root / "valid"
    test_dir = root / "test"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Expected 'train' and 'val' (or 'valid') directories inside '{root.resolve()}'. "
            "Run `python src/verify_data.py` first."
        )

    image_datasets = {
        "train": datasets.ImageFolder(root=str(train_dir), transform=tfms["train"]),
        "val": datasets.ImageFolder(root=str(val_dir), transform=tfms["val"]),
    }

    if test_dir.exists() and any(test_dir.rglob("*.*")):
        image_datasets["test"] = datasets.ImageFolder(root=str(test_dir), transform=tfms["test"])

    dataloaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
        for split, ds in image_datasets.items()
    }

    idx_to_class = {idx: cls_name for cls_name, idx in image_datasets["train"].class_to_idx.items()}
    return dataloaders, idx_to_class
