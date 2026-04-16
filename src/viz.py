"""Visualization: training curves, confusion matrix, Grad-CAM overlays."""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import seaborn as sns
import torch
from PIL import Image


# 按优先级尝试的 CJK 字体列表（跨 Linux / macOS / Windows）
_CJK_FONT_CANDIDATES = [
    "Noto Sans CJK SC",      # Google Noto（Linux 常见）
    "Noto Sans CJK JP",
    "Source Han Sans SC",    # Adobe 思源黑体
    "Source Han Sans CN",
    "WenQuanYi Zen Hei",     # 文泉驿（Linux）
    "WenQuanYi Micro Hei",
    "Microsoft YaHei",       # Windows
    "SimHei",                # Windows
    "PingFang SC",           # macOS
    "Heiti SC",              # macOS
    "STHeiti",               # macOS
    "Arial Unicode MS",      # macOS 兜底
]


def setup_chinese_font(verbose: bool = True) -> Optional[str]:
    """配置 matplotlib 使用中文字体。

    按优先级尝试 `_CJK_FONT_CANDIDATES`，找到就设置并返回字体名。
    如果都没有，打印安装提示并返回 None（图表会回退到英文）。
    """
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    for font_name in _CJK_FONT_CANDIDATES:
        if font_name in available_fonts:
            plt.rcParams["font.family"] = [font_name, "sans-serif"]
            plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示
            if verbose:
                print(f"[viz] 中文字体已配置: {font_name}")
            return font_name

    # 没有找到任何 CJK 字体
    plt.rcParams["axes.unicode_minus"] = False
    if verbose:
        print("[viz] 警告: 未找到中文字体，图表中文会显示为 □□□。")
        print("      Linux (Debian/Ubuntu): sudo apt install fonts-noto-cjk")
        print("      Linux (CentOS/RHEL):  sudo yum install google-noto-sans-cjk-fonts")
        print("      或在 Python 里运行: matplotlib.font_manager.fontManager.addfont(...)")
    return None


def plot_training_curves(history: dict, output_path: Path) -> None:
    """Plot training and validation loss/accuracy curves."""
    epochs_global = list(range(1, len(history["epoch"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs_global, history["train_loss"], "o-", label="Train")
    axes[0].plot(epochs_global, history["val_loss"], "s--", label="Val")
    axes[0].set_xlabel("Global epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss over epochs")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs_global, history["train_acc"], "o-", label="Train")
    axes[1].plot(epochs_global, history["val_acc"], "s--", label="Val")
    axes[1].set_xlabel("Global epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy over epochs")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Shade stage transitions
    stage_2_start = next((i + 1 for i, s in enumerate(history["stage"]) if s == 2), None)
    if stage_2_start is not None:
        for ax in axes:
            ax.axvline(x=stage_2_start - 0.5, color="red", linestyle=":", alpha=0.5, label="Stage 2 start")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Plot a confusion matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        cbar=True, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def make_gradcam_overlay(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    target_layer,
    original_pil: Image.Image,
    device: str = "cuda",
) -> np.ndarray:
    """Generate a Grad-CAM heatmap overlay on the original PIL image.

    Args:
        model: trained model on device
        image_tensor: single preprocessed image tensor (3, 224, 224), no batch dim
        target_class: class index to explain
        target_layer: module reference (e.g. model.layer4[-1])
        original_pil: original PIL image (any size); will be resized to 224x224 for overlay
        device: 'cuda' or 'cpu'

    Returns:
        H×W×3 uint8 numpy array with Grad-CAM overlaid.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    model.eval()
    cam = GradCAM(model=model, target_layers=[target_layer])
    input_tensor = image_tensor.unsqueeze(0).to(device)
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # H×W

    rgb = np.asarray(original_pil.resize((224, 224))).astype(np.float32) / 255.0
    overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
    return overlay
