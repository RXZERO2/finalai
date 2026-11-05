"""
Evaluation Script (`src/evaluate.py`) for the 4-Class Coffee Roast Level Classifier.

Features:
1. Loads `models/coffee_model_best.pth` and `models/classes.json`, and evaluates on `data/test/`.
2. Calculates and displays Overall Accuracy and per-class Precision, Recall, and F1-Score
   for all 4 classes: Dark (0), Green (1), Light (2), Medium (3).
3. Plots and saves a 4x4 Confusion Matrix heatmap to `reports/confusion_matrix.png` using Seaborn.
4. Saves top-5 misclassified images analysis plot to `reports/top_misclassified.png`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train import IMAGENET_MEAN, IMAGENET_STD, build_model


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom ImageFolder returning `(tensor, label, file_path)`."""

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        img_tensor, label = super().__getitem__(index)
        img_path = self.imgs[index][0]
        return img_tensor, label, img_path


def load_class_names(
    classes_path: Path,
    checkpoint: Dict,
    dataset_classes: List[str],
) -> List[str]:
    """
    Resolve ordered class names from `models/classes.json`
    (format `{"0": "Dark", "1": "Green", "2": "Light", "3": "Medium"}`).
    """
    if classes_path.exists():
        with open(classes_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if all(k.isdigit() for k in data.keys()):
                return [data[str(i)] for i in range(len(data))]
            if "classes" in data and isinstance(data["classes"], list):
                return data["classes"]
            if "idx_to_class" in data:
                idx_map = data["idx_to_class"]
                return [idx_map[str(i)] for i in range(len(idx_map))]

    if isinstance(checkpoint, dict) and "class_to_idx" in checkpoint:
        class_to_idx = checkpoint["class_to_idx"]
        idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}
        return [idx_to_class[i] for i in range(len(idx_to_class))]

    return dataset_classes


def load_trained_model(
    model_path: Path,
    num_classes: int,
    device: torch.device,
    default_arch: str = "mobilenet_v2",
) -> nn.Module:
    """Load trained 4-class model weights from `models/coffee_model_best.pth`."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{model_path}'. "
            "Run `python src/train.py` first to create `models/coffee_model_best.pth`."
        )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        arch = checkpoint.get("arch", default_arch)
        ckpt_num_classes = checkpoint.get("num_classes", num_classes)
        state_dict = checkpoint["model_state_dict"]
    else:
        arch = default_arch
        ckpt_num_classes = num_classes
        state_dict = checkpoint

    model = build_model(arch=arch, num_classes=ckpt_num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    overall_acc: float,
    output_path: Path,
) -> None:
    """Plot and save a 4x4 Confusion Matrix heatmap to `reports/confusion_matrix.png`."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlOrBr",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 13, "weight": "bold"},
        linewidths=0.8,
        linecolor="white",
        ax=ax,
    )

    ax.set_xlabel("Predicted Class", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Ground Truth Class", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(
        f"4-Class Coffee Bean Classification — Confusion Matrix\n(Test Accuracy: {overall_acc * 100:.2f}%)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11, rotation=0)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved 4x4 confusion matrix heatmap to '{output_path}'.")


def plot_top_misclassified(
    misclassified: List[Dict],
    output_path: Path,
    top_k: int = 5,
) -> None:
    """Save a figure of the top-K highest-confidence misclassified test images."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not misclassified:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No Misclassified Images Found on Test Set (100.0% Accuracy)",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="#2e7d32",
        )
        plt.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        print(f"No misclassifications found! Saved summary figure to '{output_path}'.")
        return

    sorted_errors = sorted(misclassified, key=lambda x: x["confidence"], reverse=True)[:top_k]
    num_to_plot = len(sorted_errors)

    fig, axes = plt.subplots(1, num_to_plot, figsize=(3.6 * num_to_plot, 4.4))
    if num_to_plot == 1:
        axes = [axes]

    for ax, item in zip(axes, sorted_errors):
        img_path = Path(item["path"])
        with Image.open(img_path) as img:
            ax.imshow(img.convert("RGB"))

        conf_pct = item["confidence"] * 100.0
        ax.set_title(
            f"True: {item['true_label']}\nPred: {item['pred_label']} ({conf_pct:.1f}%)",
            fontsize=11,
            fontweight="bold",
            color="#b71c1c",
            pad=8,
        )
        ax.set_xlabel(img_path.name, fontsize=9, color="#555555")
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#d32f2f")
            spine.set_linewidth(1.5)

    fig.suptitle(
        f"Top {num_to_plot} Misclassified Test Images (Ranked by Confidence)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved top {num_to_plot} misclassified images plot to '{output_path}'.")


def evaluate(
    data_dir: Path = Path("data"),
    model_path: Path = Path("models/coffee_model_best.pth"),
    classes_path: Path = Path("models/classes.json"),
    cm_output_path: Path = Path("reports/confusion_matrix.png"),
    misclassified_output_path: Path = Path("reports/top_misclassified.png"),
    batch_size: int = 32,
    top_k_errors: int = 5,
) -> None:
    """Run inference on `data/test/`, print metrics, and save confusion matrix & error analysis."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir = data_dir / "test"

    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test directory '{test_dir}' not found. Run `python src/verify_data.py` first."
        )

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    test_dataset = ImageFolderWithPaths(root=str(test_dir), transform=test_transforms)
    if len(test_dataset) == 0:
        raise RuntimeError(f"No images found inside '{test_dir}'.")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = load_class_names(classes_path, checkpoint, test_dataset.classes)
    model = load_trained_model(model_path, num_classes=len(class_names), device=device)

    all_labels: List[int] = []
    all_preds: List[int] = []
    misclassified_records: List[Dict] = []

    with torch.no_grad():
        for inputs, labels, paths in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            confidences, preds = torch.max(probs, dim=1)

            labels_np = labels.numpy()
            preds_np = preds.cpu().numpy()
            confs_np = confidences.cpu().numpy()

            all_labels.extend(labels_np.tolist())
            all_preds.extend(preds_np.tolist())

            for idx in range(len(paths)):
                true_idx = int(labels_np[idx])
                pred_idx = int(preds_np[idx])
                if true_idx != pred_idx:
                    misclassified_records.append({
                        "path": paths[idx],
                        "true_idx": true_idx,
                        "pred_idx": pred_idx,
                        "true_label": class_names[true_idx],
                        "pred_label": class_names[pred_idx],
                        "confidence": float(confs_np[idx]),
                    })

    overall_acc = accuracy_score(all_labels, all_preds)
    report_str = classification_report(
        all_labels,
        all_preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print("=" * 64)
    print("4-CLASS COFFEE ROAST LEVEL CLASSIFICATION — TEST EVALUATION")
    print("=" * 64)
    print(f"Checkpoint Evaluated   : {model_path}")
    print(f"Classes                : { {i: c for i, c in enumerate(class_names)} }")
    print(f"Test Samples Evaluated : {len(all_labels)}")
    print(f"Overall Accuracy       : {overall_acc * 100:.2f}% ({overall_acc:.4f})")
    print(f"Total Misclassified    : {len(misclassified_records)}")
    print("-" * 64)
    print("Classification Report (Precision, Recall, F1-Score per class):")
    print(report_str)
    print("=" * 64)

    plot_confusion_matrix(
        y_true=all_labels,
        y_pred=all_preds,
        class_names=class_names,
        overall_acc=overall_acc,
        output_path=cm_output_path,
    )

    plot_top_misclassified(
        misclassified=misclassified_records,
        output_path=misclassified_output_path,
        top_k=top_k_errors,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate 4-class Coffee Roast Classifier on data/test/."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-path", type=Path, default=Path("models/coffee_model_best.pth"))
    parser.add_argument("--classes-path", type=Path, default=Path("models/classes.json"))
    parser.add_argument("--cm-output", type=Path, default=Path("reports/confusion_matrix.png"))
    parser.add_argument(
        "--misclassified-output",
        type=Path,
        default=Path("reports/top_misclassified.png"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        data_dir=args.data_dir,
        model_path=args.model_path,
        classes_path=args.classes_path,
        cm_output_path=args.cm_output,
        misclassified_output_path=args.misclassified_output,
        batch_size=args.batch_size,
        top_k_errors=args.top_k,
    )
