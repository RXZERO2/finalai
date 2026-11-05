"""
Synthetic dataset generator for Coffee Roast Level Classification.
Generates simulated coffee bean textures/colors for Light, Medium, and Dark roasts
across train, val, and test splits so the training pipeline can be tested immediately.
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# Representative RGB base profiles (R, G, B) for each roast level
ROAST_PROFILES = {
    "Light": {
        "base_rgb": (181, 130, 78),    # Cinnamon / light brown
        "std_rgb": (18, 15, 12),
        "crease_rgb": (215, 175, 125), # Lighter chaff in the center crease
    },
    "Medium": {
        "base_rgb": (118, 72, 42),     # Rich medium brown
        "std_rgb": (14, 12, 10),
        "crease_rgb": (150, 100, 62),
    },
    "Dark": {
        "base_rgb": (52, 34, 24),      # Deep dark espresso brown / near-black
        "std_rgb": (10, 8, 8),
        "crease_rgb": (75, 50, 36),
    },
}


def generate_coffee_image(
    roast_class: str,
    img_size: int = 256,
    rng: np.random.Generator | None = None,
) -> Image.Image:
    """Create a synthetic image featuring roasted coffee beans on a neutral surface."""
    if rng is None:
        rng = np.random.default_rng()

    profile = ROAST_PROFILES[roast_class]
    base_r, base_g, base_b = profile["base_rgb"]
    std_r, std_g, std_b = profile["std_rgb"]

    # Subtle background noise
    bg_val = int(rng.integers(210, 240))
    canvas = np.full((img_size, img_size, 3), bg_val, dtype=np.float32)
    canvas += rng.normal(0, 6.0, size=canvas.shape)
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    img = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(img)

    # Draw multiple coffee beans with natural color jitter
    num_beans = int(rng.integers(18, 32))
    for _ in range(num_beans):
        cx = int(rng.integers(20, img_size - 20))
        cy = int(rng.integers(20, img_size - 20))
        rx = int(rng.integers(16, 28))
        ry = int(rng.integers(12, 22))

        r = int(np.clip(rng.normal(base_r, std_r), 10, 245))
        g = int(np.clip(rng.normal(base_g, std_g), 10, 245))
        b = int(np.clip(rng.normal(base_b, std_b), 10, 245))

        bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
        draw.ellipse(bbox, fill=(r, g, b), outline=(max(r - 20, 0), max(g - 20, 0), max(b - 20, 0)))

        # Center crease line typical of coffee beans
        crease_color = profile["crease_rgb"]
        draw.line([(cx - rx + 4, cy), (cx + rx - 4, cy)], fill=crease_color, width=2)

    # Slight Gaussian blur to blend edges naturally
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    return img


def build_synthetic_dataset(
    data_dir: str | Path = "data",
    train_per_class: int = 40,
    val_per_class: int = 12,
    test_per_class: int = 12,
    img_size: int = 256,
    seed: int = 42,
) -> None:
    """Populate data/{train,val,test}/{Light,Medium,Dark} with synthetic images."""
    root = Path(data_dir)
    rng = np.random.default_rng(seed)

    splits = {
        "train": train_per_class,
        "val": val_per_class,
        "test": test_per_class,
    }

    total_created = 0
    for split_name, count in splits.items():
        for roast_class in ROAST_PROFILES:
            target_dir = root / split_name / roast_class
            target_dir.mkdir(parents=True, exist_ok=True)

            for idx in range(count):
                img = generate_coffee_image(roast_class, img_size=img_size, rng=rng)
                img_path = target_dir / f"{roast_class.lower()}_{idx:04d}.jpg"
                img.save(img_path, quality=92)
                total_created += 1

    print(f"Generated {total_created} synthetic images in '{root.resolve()}'.")
    for split_name, count in splits.items():
        print(f"  - {split_name}: {count} images/class ({count * len(ROAST_PROFILES)} total)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Coffee Roast Level dataset (Light, Medium, Dark)."
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory.")
    parser.add_argument("--train-count", type=int, default=40, help="Train images per class.")
    parser.add_argument("--val-count", type=int, default=12, help="Validation images per class.")
    parser.add_argument("--test-count", type=int, default=12, help="Test images per class.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_synthetic_dataset(
        data_dir=args.data_dir,
        train_per_class=args.train_count,
        val_per_class=args.val_count,
        test_per_class=args.test_count,
        seed=args.seed,
    )

