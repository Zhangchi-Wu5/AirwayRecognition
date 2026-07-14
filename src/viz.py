"""Visualization: training curves, confusion matrix, Grad-CAM overlays."""
from collections import deque
import os
from pathlib import Path
import tempfile
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "airway_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "airway_cache"))

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


def detect_dark_border_mask(
    image: Image.Image | np.ndarray,
    dark_threshold: int = 24,
) -> np.ndarray:
    """Detect edge-connected dark non-image regions such as scope black borders.

    Returns a boolean HxW mask. Only dark pixels connected to the image edge are
    included, so dark anatomical cavities in the middle are not marked by this
    rule.
    """
    rgb = _to_rgb_array(image)
    luminance = (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    )
    dark = luminance <= dark_threshold
    return _edge_connected_mask(dark)


def detect_specular_highlight_mask(
    image: Image.Image | np.ndarray,
    bright_threshold: int = 235,
    color_spread_threshold: int = 35,
    min_area: int = 4,
) -> np.ndarray:
    """Detect near-white specular highlights from bronchoscopy illumination.

    The rule targets very bright pixels whose RGB channels are close together.
    This intentionally detects strong white glare, not all bright tissue.
    """
    rgb = _to_rgb_array(image)
    max_channel = rgb.max(axis=2)
    min_channel = rgb.min(axis=2)
    mask = (
        (max_channel >= bright_threshold)
        & ((max_channel - min_channel) <= color_spread_threshold)
    )
    if min_area > 1:
        mask = _remove_small_components(mask, min_area=min_area)
    return mask


def build_pseudo_feature_masks(
    image: Image.Image | np.ndarray,
    dark_threshold: int = 24,
    bright_threshold: int = 235,
    color_spread_threshold: int = 35,
    min_highlight_area: int = 4,
) -> dict[str, np.ndarray]:
    """Build rule-based masks for easy pseudo-features.

    The returned dictionary contains dark border, specular highlight, and their
    union as boolean HxW masks.
    """
    dark_border = detect_dark_border_mask(image, dark_threshold=dark_threshold)
    specular_highlight = detect_specular_highlight_mask(
        image,
        bright_threshold=bright_threshold,
        color_spread_threshold=color_spread_threshold,
        min_area=min_highlight_area,
    )
    return {
        "dark_border": dark_border,
        "specular_highlight": specular_highlight,
        "combined": dark_border | specular_highlight,
    }


def pseudo_feature_attention_score(heatmap: np.ndarray, mask: np.ndarray) -> float:
    """Return the fraction of heatmap response falling inside a pseudo-feature mask."""
    heatmap = np.asarray(heatmap, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if heatmap.shape != mask.shape:
        raise ValueError(f"heatmap and mask shapes differ: {heatmap.shape} vs {mask.shape}")
    total = float(np.maximum(heatmap, 0).sum())
    if total == 0:
        return 0.0
    return float(np.maximum(heatmap, 0)[mask].sum() / total)


def make_gradcam_heatmap(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    target_layer,
    device: str = "cuda",
) -> np.ndarray:
    """Generate a Grad-CAM grayscale heatmap for a single preprocessed image."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model.eval()
    cam = GradCAM(model=model, target_layers=[target_layer])
    input_tensor = image_tensor.unsqueeze(0).to(device)
    targets = [ClassifierOutputTarget(target_class)]
    return cam(input_tensor=input_tensor, targets=targets)[0]


def make_gradcam_overlay(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    target_layer,
    original_pil: Image.Image,
    device: str = "cuda",
    crop_border: bool = False,
) -> np.ndarray:
    """Generate a Grad-CAM heatmap overlay on the original PIL image.

    Args:
        model: trained model on device
        image_tensor: single preprocessed image tensor (3, 224, 224), no batch dim
        target_class: class index to explain
        target_layer: module reference (e.g. model.layer4[-1])
        original_pil: original PIL image (any size); receives the same deterministic
            geometry used by the model input before overlay
        device: 'cuda' or 'cpu'
        crop_border: match the flag passed to ``get_eval_transforms``

    Returns:
        H×W×3 uint8 numpy array with Grad-CAM overlaid.
    """
    from pytorch_grad_cam.utils.image import show_cam_on_image

    grayscale_cam = make_gradcam_heatmap(
        model=model,
        image_tensor=image_tensor,
        target_class=target_class,
        target_layer=target_layer,
        device=device,
    )
    from src.data import get_eval_geometry_transforms

    aligned_pil = get_eval_geometry_transforms(crop_border=crop_border)(original_pil)
    if aligned_pil.size != (grayscale_cam.shape[1], grayscale_cam.shape[0]):
        raise RuntimeError(
            f"Aligned image has size {aligned_pil.size}, heatmap is "
            f"{grayscale_cam.shape[1]}x{grayscale_cam.shape[0]}"
        )
    rgb = np.asarray(aligned_pil).astype(np.float32) / 255.0
    overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
    return overlay


def _to_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must be an RGB image with shape HxWx3")
    return array.astype(np.uint8, copy=False)


def _edge_connected_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for x in range(w):
        for y in (0, h - 1):
            if mask[y, x] and not visited[y, x]:
                visited[y, x] = True
                queue.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if mask[y, x] and not visited[y, x]:
                visited[y, x] = True
                queue.append((y, x))

    while queue:
        y, x = queue.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((ny, nx))
    return visited


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    output = np.zeros((h, w), dtype=bool)

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            component = []
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                component.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            if len(component) >= min_area:
                ys, xs = zip(*component)
                output[ys, xs] = True
    return output
