"""
Production-ready Streamlit application (`app.py`) for 4-Class Coffee Bean Roast Classification:
  - Index 0: Dark
  - Index 1: Green (Unroasted / Raw Coffee Bean)
  - Index 2: Light
  - Index 3: Medium

Features:
1. Loads `models/coffee_model_best.pth` and `models/classes.json` with `@st.cache_resource`.
2. Supports both file upload (JPG/PNG) and live camera capture (`st.camera_input`).
3. Displays predicted class with dynamic color badge, confidence score (%), and a 4-class probability bar chart.
4. Dedicated note/badge for the `Green` class ("Unroasted / Raw Coffee Bean") alongside sensory/tasting
   profiles for `Dark`, `Medium`, and `Light`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train import IMAGENET_MEAN, IMAGENET_STD, build_model

DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "coffee_model_best.pth"
FALLBACK_MODEL_PATH = ROOT_DIR / "models" / "best_model.pth"
DEFAULT_CLASSES_PATH = ROOT_DIR / "models" / "classes.json"
DEFAULT_CLASSES = ["Dark", "Green", "Light", "Medium"]

# Detailed metadata for all 4 classes: Dark, Green, Light, Medium
ROAST_METADATA: Dict[str, Dict[str, str]] = {
    "Green": {
        "badge_bg": "#4F772D",
        "badge_text": "#FFFFFF",
        "border_color": "#31572C",
        "status_pill": "UNROASTED / RAW COFFEE BEAN",
        "tagline": "Unroasted / Raw Agricultural Coffee Seed — Requires Roasting Before Brewing",
        "acidity": "N/A (Raw / Unroasted — Grassy & Astringent if brewed raw)",
        "body": "Thin, vegetal & herbal infusion (Not standard roasted coffee)",
        "flavor_notes": "Raw green pea, fresh-cut grass, hay, herbal tea (Prior to Maillard reaction)",
        "roast_temp": "Ambient / Unroasted (20°C — Moisture content ~10–12%)",
        "brewing": "Must be roasted first! (Or steeped only for specialized green coffee extract)",
        "description": (
            "Green coffee beans are raw, unroasted seeds of the Coffea cherry. They contain high levels "
            "of chlorogenic acid and locked-in sucrose/amino precursors, but have not yet undergone "
            "roasting (Maillard reactions and caramelization) needed to develop soluble coffee aromas."
        ),
        "is_green": "true",
    },
    "Light": {
        "badge_bg": "#D4A373",
        "badge_text": "#1F1308",
        "border_color": "#B5835A",
        "status_pill": "LIGHT ROAST",
        "tagline": "Bright, Complex & Origin-Forward",
        "acidity": "High, crisp & vibrant",
        "body": "Light, tea-like to silky",
        "flavor_notes": "Citrus, berry, floral jasmine, green apple, honey sweetness",
        "roast_temp": "196°C – 205°C (Dropped at First Crack)",
        "brewing": "Pour-Over (V60, Chemex), AeroPress, Filter Drip",
        "description": (
            "Light roasts are dropped shortly after the first crack, preserving the delicate "
            "aromatic compounds and natural terroir of the coffee origin. Expect pronounced "
            "acidity, fruity/floral complexity, dry bean surface, and minimal roast bitterness."
        ),
        "is_green": "false",
    },
    "Medium": {
        "badge_bg": "#8B5A2B",
        "badge_text": "#FFFFFF",
        "border_color": "#6E4720",
        "status_pill": "MEDIUM ROAST",
        "tagline": "Balanced, Sweet & Rounded",
        "acidity": "Medium, well-integrated",
        "body": "Medium, smooth & syrupy",
        "flavor_notes": "Caramel, milk chocolate, roasted hazelnut, stone fruit, brown spice",
        "roast_temp": "210°C – 220°C (Between First & Second Crack)",
        "brewing": "Drip Coffee Maker, Clever Dripper, French Press, Balanced Espresso",
        "description": (
            "Medium roasts strike a harmonious balance between origin character and Maillard "
            "caramelization. Acidity softens while body and natural sugars peak, yielding "
            "chocolatey, nutty, and caramel-forward notes."
        ),
        "is_green": "false",
    },
    "Dark": {
        "badge_bg": "#3B2314",
        "badge_text": "#F5EBE6",
        "border_color": "#23140B",
        "status_pill": "DARK ROAST",
        "tagline": "Bold, Full-Bodied & Intense",
        "acidity": "Low, muted",
        "body": "Heavy, rich & velvety",
        "flavor_notes": "Dark bittersweet chocolate, toasted walnut, molasses, smoky cedar",
        "roast_temp": "225°C – 240°C (At or Beyond Second Crack)",
        "brewing": "Espresso, Moka Pot, French Press, Cold Brew",
        "description": (
            "Dark roasts develop past the second crack, bringing natural oils to the bean surface. "
            "Roast-driven flavors dominate over origin nuances, delivering a low-acid, bold, "
            "bittersweet, and smoky cup that pairs well with milk."
        ),
        "is_green": "false",
    },
}


def get_inference_transform() -> transforms.Compose:
    """Deterministic inference transform matching training normalization."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


@st.cache_resource(show_spinner="Loading 4-Class Coffee Roast Classifier...")
def load_model_and_classes(
    model_path_str: str,
    classes_path_str: str,
) -> Tuple[nn.Module, List[str], Dict[str, str]]:
    """
    Load and cache `models/coffee_model_best.pth` and `models/classes.json`.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(model_path_str)
    classes_path = Path(classes_path_str)

    if not model_path.exists() and FALLBACK_MODEL_PATH.exists():
        model_path = FALLBACK_MODEL_PATH

    if not model_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{model_path}'. Run `python src/train.py` first."
        )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Resolve 4-class index mapping: {"0": "Dark", "1": "Green", "2": "Light", "3": "Medium"}
    class_names: List[str] = DEFAULT_CLASSES
    if classes_path.exists():
        with open(classes_path, "r", encoding="utf-8") as f:
            cls_data = json.load(f)
        if isinstance(cls_data, dict):
            if all(k.isdigit() for k in cls_data.keys()):
                class_names = [cls_data[str(i)] for i in range(len(cls_data))]
            elif "classes" in cls_data and isinstance(cls_data["classes"], list):
                class_names = cls_data["classes"]
            elif "idx_to_class" in cls_data:
                idx_map = cls_data["idx_to_class"]
                class_names = [idx_map[str(i)] for i in range(len(idx_map))]
    elif isinstance(checkpoint, dict) and "class_to_idx" in checkpoint:
        class_to_idx = checkpoint["class_to_idx"]
        idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        arch = checkpoint.get("arch", "mobilenet_v2")
        num_classes = checkpoint.get("num_classes", len(class_names))
        state_dict = checkpoint["model_state_dict"]
        val_acc = checkpoint.get("best_val_acc")
        epoch = checkpoint.get("epoch")
    else:
        arch = "mobilenet_v2"
        num_classes = len(class_names)
        state_dict = checkpoint
        val_acc = None
        epoch = None

    model = build_model(arch=arch, num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    meta = {
        "device": str(device),
        "arch": arch,
        "val_acc": f"{val_acc * 100:.2f}%" if isinstance(val_acc, float) else "N/A",
        "epoch": str(epoch) if epoch is not None else "N/A",
        "checkpoint": model_path.name,
    }
    return model, class_names, meta


def predict_image(
    image: Image.Image,
    model: nn.Module,
    class_names: List[str],
) -> Tuple[str, float, Dict[str, float]]:
    """Run forward inference on a PIL image and return `(pred_class, confidence, prob_dict)`."""
    device = next(model.parameters()).device
    rgb_image = image.convert("RGB")
    transform = get_inference_transform()
    input_tensor = transform(rgb_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    prob_dict = {cls_name: float(probs[idx]) for idx, cls_name in enumerate(class_names)}
    top_idx = int(np.argmax(probs))
    pred_class = class_names[top_idx]
    top_conf = float(probs[top_idx])
    return pred_class, top_conf, prob_dict


def render_confidence_chart(prob_dict: Dict[str, float]) -> plt.Figure:
    """Plot horizontal probability bar chart across all 4 classes (Dark, Green, Light, Medium)."""
    # Present in natural progression or index order
    display_order = ["Green", "Light", "Medium", "Dark"]
    labels = [c for c in display_order if c in prob_dict] + [
        c for c in prob_dict if c not in display_order
    ]
    values = [prob_dict[c] * 100.0 for c in labels]
    colors = [ROAST_METADATA.get(c, {}).get("badge_bg", "#6C757D") for c in labels]

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, height=0.58, edgecolor="#2B2B2B", linewidth=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel("Prediction Probability (%)", fontsize=10, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, val in zip(bars, values):
        ax.text(
            min(val + 1.8, 101.5),
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.1f}%",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#222222",
        )

    plt.tight_layout()
    return fig


def render_roast_badge_and_profile(pred_class: str, confidence: float) -> None:
    """Render dynamic badge, Green bean alert note (when Green), and tasting profile."""
    profile = ROAST_METADATA.get(pred_class, ROAST_METADATA["Medium"])
    title_label = "Green Coffee Bean (Unroasted / Raw)" if pred_class == "Green" else f"{pred_class} Roast"

    badge_html = f"""
    <div style="
        background-color: {profile['badge_bg']};
        color: {profile['badge_text']};
        border: 2px solid {profile['border_color']};
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.12);
    ">
        <div style="font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.3px; opacity: 0.92;">
            {profile['status_pill']}
        </div>
        <div style="display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; margin-top: 4px;">
            <span style="font-size: 1.9rem; font-weight: 800;">{title_label}</span>
            <span style="font-size: 1.2rem; font-weight: 700; padding: 4px 12px; border-radius: 20px; background: rgba(0,0,0,0.20);">
                {confidence * 100:.1f}% Confidence
            </span>
        </div>
        <div style="font-size: 0.96rem; font-style: italic; margin-top: 6px; opacity: 0.95;">
            {profile['tagline']}
        </div>
    </div>
    """
    st.markdown(badge_html, unsafe_allow_html=True)

    if pred_class == "Green":
        st.warning(
            "🌱 **Unroasted / Raw Coffee Bean Detected (`Green`):** "
            "These beans are in their raw agricultural state prior to roasting. "
            "They should go through a roaster (First Crack at ~196°C+) before grinding and brewing as standard coffee."
        )

    st.markdown("#### Sensory & Bean Profile")
    st.write(profile["description"])

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Flavor / Aroma Notes:**\n{profile['flavor_notes']}")
        st.markdown(f"**Acidity:** {profile['acidity']}")
        st.markdown(f"**Body / Mouthfeel:** {profile['body']}")
    with col_b:
        st.markdown(f"**Recommended Handling / Brewing:**\n{profile['brewing']}")
        st.markdown(f"**Temperature Stage:** {profile['roast_temp']}")


def render_sidebar() -> Tuple[Path, Path]:
    """Configure sidebar with 4-class project details and checkpoint paths."""
    with st.sidebar:
        st.header("About This Project")
        st.markdown(
            """
            **Coffee Bean Roast Level Classifier** classifies coffee beans into **4 stages**:
            - `0`: **Dark** Roast
            - `1`: **Green** *(Unroasted / Raw Bean)*
            - `2`: **Light** Roast
            - `3`: **Medium** Roast

            **Model Pipeline:**
            - **Backbone:** `MobileNetV2` / `ResNet18` (4-class head)
            - **Transforms:** `224×224` Resize, Horizontal/Vertical Flip, ColorJitter, ImageNet Norm
            - **Checkpoint:** `models/coffee_model_best.pth`
            """
        )

        st.divider()
        with st.expander("Model Configuration", expanded=False):
            model_path_input = st.text_input(
                "Checkpoint Path",
                value=str(DEFAULT_MODEL_PATH.relative_to(ROOT_DIR)),
            )
            classes_path_input = st.text_input(
                "Classes JSON Path",
                value=str(DEFAULT_CLASSES_PATH.relative_to(ROOT_DIR)),
            )

    return ROOT_DIR / model_path_input, ROOT_DIR / classes_path_input


def main() -> None:
    st.set_page_config(
        page_title="Coffee Bean Roast Classifier (4 Classes)",
        page_icon="☕",
        layout="wide",
    )

    st.title("☕ Coffee Bean Roast Level Classifier")
    st.caption(
        "4-Class Computer Vision Inspection — Identify **Green (Unroasted)**, **Light**, "
        "**Medium**, and **Dark** coffee beans with real-time sensory profiling."
    )

    model_path, classes_path = render_sidebar()

    try:
        model, class_names, meta = load_model_and_classes(str(model_path), str(classes_path))
        with st.sidebar:
            st.success(
                f"**Model Ready** (`{meta['arch']}`)\n\n"
                f"- **Checkpoint:** `{meta['checkpoint']}`\n"
                f"- **Device:** `{meta['device']}`\n"
                f"- **Best Val Acc:** `{meta['val_acc']}`\n"
                f"- **Classes ({len(class_names)}):** `{', '.join(class_names)}`"
            )
    except FileNotFoundError as err:
        st.error(
            f"**Model checkpoint not found:** `{model_path}`\n\n"
            "Verify the dataset and train the 4-class model first:\n"
            "```bash\n"
            "python src/verify_data.py\n"
            "python src/train.py --epochs 15\n"
            "```"
        )
        st.stop()
    except Exception as exc:
        st.error(f"Error loading model checkpoint: {exc}")
        st.stop()

    tab_upload, tab_camera = st.tabs(["📁 Upload Image (JPG / PNG)", "📷 Live Camera Capture"])

    input_image: Image.Image | None = None
    source_label = ""

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload an image of coffee beans (Green, Light, Medium, or Dark)",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file).convert("RGB")
            source_label = f"Uploaded File: `{uploaded_file.name}`"

    with tab_camera:
        st.info("Place coffee beans under clear lighting and take a photo.")
        camera_photo = st.camera_input("Capture coffee beans")
        if camera_photo is not None:
            input_image = Image.open(camera_photo).convert("RGB")
            source_label = "Live Camera Capture"

    st.divider()

    if input_image is not None:
        col_img, col_res = st.columns([1, 1.35], gap="large")

        with col_img:
            st.subheader("Input Sample")
            st.image(input_image, caption=source_label, use_container_width=True)
            w, h = input_image.size
            st.caption(f"Original Resolution: `{w} × {h} px` → Resized to `224 × 224 px`")

        with col_res:
            st.subheader("Prediction & Roast Profile")
            pred_class, top_conf, prob_dict = predict_image(input_image, model, class_names)

            render_roast_badge_and_profile(pred_class, top_conf)

            st.markdown("#### 4-Class Probability Distribution")
            fig = render_confidence_chart(prob_dict)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    else:
        st.markdown(
            """
            <div style="
                border: 1px dashed #8B5A2B;
                border-radius: 10px;
                padding: 28px;
                text-align: center;
                color: #555555;
                background-color: rgba(212, 163, 115, 0.08);
            ">
                <h4 style="margin-bottom: 6px;">Ready for 4-Class Inference</h4>
                <p style="margin: 0;">
                    Choose <b>Upload Image (JPG / PNG)</b> or <b>Live Camera Capture</b> above to classify
                    <b>Dark</b>, <b>Green (Unroasted)</b>, <b>Light</b>, or <b>Medium</b> coffee beans.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
