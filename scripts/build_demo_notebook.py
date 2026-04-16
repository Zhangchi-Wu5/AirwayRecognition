"""Build the Gradio demo notebook."""
from pathlib import Path
import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "notebooks" / "demo.ipynb"


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    nb.cells = [
        md("""# 气管镜部位识别 — Gradio Demo

上传一张气管镜图片，模型会返回：
- 预测类别（隆突 / 右总支气管 / 左总支气管）
- 三类的 softmax 置信度
- Grad-CAM 热力图

**前置条件**：已经跑过 `bronchoscopy_classifier.ipynb` 并在 `checkpoints/best_model.pt` 生成了模型权重。
"""),
        code("""import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import gradio as gr
from PIL import Image

from src.data import get_eval_transforms, LABEL_NAMES_CN, ID_TO_LABEL
from src.models import build_resnet50
from src.viz import make_gradcam_overlay

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = PROJECT_ROOT / "checkpoints" / "best_model.pt"
assert CKPT.exists(), f"Checkpoint not found: {CKPT}. Run the training notebook first."

model = build_resnet50(num_classes=3, pretrained=False, dropout=0.3).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()

transform = get_eval_transforms()
target_layer = model.layer4[-1]
CLASS_NAMES = ["lt", "yz", "zz"]
print(f"Model loaded. Device: {DEVICE}")
"""),
        code("""def predict(image: Image.Image):
    if image is None:
        return None, None, None
    image = image.convert("RGB")
    tensor = transform(image).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0))
        proba = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_id = int(np.argmax(proba))
    # Gradio Label expects {label: prob}
    label_output = {f"{ID_TO_LABEL[i]} ({LABEL_NAMES_CN[ID_TO_LABEL[i]]})": float(proba[i]) for i in range(3)}
    # Grad-CAM overlay
    overlay = make_gradcam_overlay(
        model=model, image_tensor=tensor, target_class=pred_id,
        target_layer=target_layer, original_pil=image, device=DEVICE,
    )
    return label_output, Image.fromarray(overlay), f"{ID_TO_LABEL[pred_id]} ({LABEL_NAMES_CN[ID_TO_LABEL[pred_id]]}) — 置信度 {proba[pred_id]:.1%}"


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="上传气管镜图片"),
    outputs=[
        gr.Label(num_top_classes=3, label="三类置信度"),
        gr.Image(type="pil", label="Grad-CAM 热力图"),
        gr.Textbox(label="预测结果"),
    ],
    title="气管镜部位识别",
    description="上传一张气管镜 RGB 图片，模型输出分类结果、置信度、和 Grad-CAM 可解释性热力图。",
    examples=None,
)

# 启动本地服务（share=True 可生成公网临时链接）
demo.launch(share=False)
"""),
    ]
    return nb


if __name__ == "__main__":
    nb = build()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Demo notebook written to {OUTPUT_PATH}")
