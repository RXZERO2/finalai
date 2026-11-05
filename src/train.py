"""
Model Training Script (`src/train.py`) for 4-Class Coffee Bean Roast Classification:
  - Index 0: Dark
  - Index 1: Green
  - Index 2: Light
  - Index 3: Medium

Features:
- Lightweight Transfer Learning backbone (`mobilenet_v2` or `resnet18`) modified for 4 classes.
- Preprocessing & Augmentation: Resize(224, 224), RandomHorizontalFlip, RandomVerticalFlip,
  ColorJitter, and ImageNet Normalization.
- CrossEntropyLoss + Adam optimizer + Early Stopping.
- Saves best model checkpoint to `models/coffee_model_best.pth`.
- Saves class mapping `{"0": "Dark", "1": "Green", "2": "Light", "3": "Medium"}` to `models/classes.json`.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EXPECTED_CLASS_TO_IDX = {"Dark": 0, "Green": 1, "Light": 2, "Medium": 3}


class EarlyStopping:
    """Stop training early if validation loss stops improving after `patience` epochs."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return False


def resolve_val_dir(data_dir: Path) -> Path:
    """Support either `data/val` or `data/valid` directory naming."""
    val_dir = data_dir / "val"
    valid_dir = data_dir / "valid"
    if val_dir.exists() and any(val_dir.rglob("*.*")):
        return val_dir
    if valid_dir.exists() and any(valid_dir.rglob("*.*")):
        return valid_dir
    return val_dir


def get_dataloaders(
    data_dir: Path,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
    """
    Create train and validation DataLoaders with standard transforms:
    Resize(224, 224), Horizontal/Vertical Flip, ColorJitter, and ImageNet Normalization.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_dir = data_dir / "train"
    val_dir = resolve_val_dir(data_dir)

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_transforms)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(
            f"Empty dataset in '{train_dir}' or '{val_dir}'. "
            "Run `python src/verify_data.py` first."
        )

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }

    return dataloaders, train_dataset.class_to_idx


def build_model(
    arch: str = "mobilenet_v2",
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build Transfer Learning model (`mobilenet_v2` or `resnet18`) with a 4-class head."""
    arch = arch.lower()
    if arch == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        if freeze_backbone:
            for name, param in model.named_parameters():
                if not name.startswith("fc."):
                    param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported architecture '{arch}'. Choose 'mobilenet_v2' or 'resnet18'.")

    return model


def save_class_mapping(class_to_idx: Dict[str, int], output_path: Path) -> None:
    """
    Save exact mapping `{"0": "Dark", "1": "Green", "2": "Light", "3": "Medium"}`
    to `models/classes.json`.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    idx_to_class = {str(idx): cls_name for cls_name, idx in sorted(class_to_idx.items(), key=lambda x: x[1])}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, indent=2)
    print(f"Saved class mapping {idx_to_class} to '{output_path}'.")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, float]:
    """Execute one training or validation epoch and return `(epoch_loss, epoch_acc)`."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels).item()
            total_samples += batch_size

    epoch_loss = running_loss / max(total_samples, 1)
    epoch_acc = running_corrects / max(total_samples, 1)
    return epoch_loss, epoch_acc


def train(
    data_dir: Path = Path("data"),
    model_dir: Path = Path("models"),
    arch: str = "mobilenet_v2",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 5,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    num_workers: int = 0,
) -> None:
    """Train the 4-class Coffee Roast classifier and save `models/coffee_model_best.pth`."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Architecture: {arch}")

    dataloaders, class_to_idx = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    num_classes = len(class_to_idx)
    print(f"Detected {num_classes} classes: {class_to_idx}")

    if class_to_idx != EXPECTED_CLASS_TO_IDX:
        print(
            f"[WARNING] class_to_idx is {class_to_idx}, expected {EXPECTED_CLASS_TO_IDX}."
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    classes_json_path = model_dir / "classes.json"
    best_model_path = model_dir / "coffee_model_best.pth"

    save_class_mapping(class_to_idx, classes_json_path)

    model = build_model(
        arch=arch,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=lr)
    early_stopper = EarlyStopping(patience=patience)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, dataloaders["train"], criterion, optimizer, device
        )
        val_loss, val_acc = run_epoch(
            model, dataloaders["val"], criterion, None, device
        )

        loss_improved = early_stopper.step(val_loss)
        acc_improved = val_acc > best_val_acc

        marker = ""
        if loss_improved or acc_improved:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

            checkpoint = {
                "arch": arch,
                "num_classes": num_classes,
                "class_to_idx": class_to_idx,
                "model_state_dict": best_weights,
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "epoch": epoch,
            }
            torch.save(checkpoint, best_model_path)
            marker = " [Saved Best]"

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}{marker}"
        )

        if early_stopper.early_stop:
            print(
                f"Early stopping triggered at epoch {epoch} "
                f"(no val_loss improvement for {patience} epochs)."
            )
            break

    print(
        f"Training complete. Best Val Loss: {best_val_loss:.4f} | "
        f"Best Val Acc: {best_val_acc:.4f} | Saved to '{best_model_path}'."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 4-Class Coffee Bean Classifier (Dark, Green, Light, Medium)."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Dataset root directory.")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Model output directory.")
    parser.add_argument(
        "--arch",
        type=str,
        default="mobilenet_v2",
        choices=["mobilenet_v2", "resnet18"],
        help="Transfer learning backbone architecture.",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable downloading ImageNet pretrained weights.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze convolutional feature extractor and train only the classifier layer.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        num_workers=args.num_workers,
    )
