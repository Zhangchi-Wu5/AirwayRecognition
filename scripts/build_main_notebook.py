"""Build the main teaching notebook (bronchoscopy_classifier.ipynb) using nbformat."""
from pathlib import Path
import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "notebooks" / "bronchoscopy_classifier.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


# ---------------------------------------------------------------------------
# Section functions (must be defined before build())
# ---------------------------------------------------------------------------

def _add_section_1_intro(nb):
    nb.cells.append(md("""# 🫁 气管镜部位识别：深度学习实战教程

## 项目目标
训练一个 ResNet-50 迁移学习模型，识别气管镜图像属于以下三类之一：

| 代码 | 中文 | 解剖含义 |
|------|------|----------|
| `lt` | 隆突（Carina） | 左右主支气管的分叉处，视觉最独特 |
| `yz` | 右总支气管 | 分出右上叶支 + 右中间段（中叶和下叶） |
| `zz` | 左总支气管 | 通向左上下肺开口 |

## 本教程的学习重点
1. 医学影像数据处理（特别是 **Patient Leakage** 陷阱）
2. CNN 迁移学习（两阶段 Fine-tuning）
3. 模型评估（Confusion Matrix / Per-class 指标）
4. **Grad-CAM 可解释性**：看模型到底在关注图片的哪里

> ⚠️ **重要提醒**：本教程使用的气管镜图像是**自然 RGB 图像**（内窥镜摄像头拍摄），不是 CT/MRI 灰度断层影像。所以我们使用的很多技巧（ImageNet 预训练、RGB 归一化、ColorJitter）是自然图像的标准做法。
"""))


def _add_section_2_setup(nb):
    nb.cells.append(md("""## 2. 环境准备
导入依赖、设置项目路径、固定随机种子。"""))
    nb.cells.append(code("""# 标准库
import sys
from pathlib import Path

# 把项目根目录加入 sys.path，这样就能 `from src import ...`
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 第三方库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader

# 本项目源码
from src.data import (
    build_manifest, split_by_patient,
    BronchoscopyDataset, get_train_transforms, get_eval_transforms,
    LABEL_TO_ID, ID_TO_LABEL, LABEL_NAMES_CN,
)
from src.models import build_resnet50, count_trainable_params
from src.train import set_seed, train_two_stage
from src.evaluate import collect_predictions, compute_metrics
from src.viz import (
    plot_training_curves, plot_confusion_matrix, make_gradcam_overlay,
    setup_chinese_font,
)

# 配置 matplotlib 中文字体（服务器上没字体会打印安装指引）
setup_chinese_font()

# 固定所有随机种子（Python / numpy / torch / CUDA）
set_seed(42)

# 设备检查
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch 版本: {torch.__version__}")
print(f"当前设备: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 路径常量
DATASET_DIR = PROJECT_ROOT / "dataset"
SPLITS_DIR = PROJECT_ROOT / "data_splits"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
for d in [SPLITS_DIR, CKPT_DIR, OUTPUT_DIR, OUTPUT_DIR / "gradcam_examples"]:
    d.mkdir(parents=True, exist_ok=True)
"""))


def _add_section_3_explore(nb):
    nb.cells.append(md("""## 3. 数据探索
先看看 `dataset/` 里有什么。每个文件名包含一个病人 ID 和解剖标签。"""))
    nb.cells.append(code("""# 列出前 15 个文件，感受一下命名
files = sorted(p.name for p in DATASET_DIR.iterdir() if p.is_file())
print(f"文件总数: {len(files)}")
print("前 15 个文件:")
for f in files[:15]:
    print(f"  {f}")
"""))
    nb.cells.append(code("""# 可视化：每类取一张样本，并排展示
from PIL import Image

samples = {}
for f in files:
    for label in ["lt", "yz", "zz"]:
        if label in f.lower() and label not in samples:
            samples[label] = DATASET_DIR / f
            break
    if len(samples) == 3:
        break

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (label, path) in zip(axes, samples.items()):
    img = Image.open(path)
    ax.imshow(img)
    ax.set_title(f"{label} ({LABEL_NAMES_CN[label]})\\n{img.size}")
    ax.axis("off")
plt.suptitle("三个类别的样本图片", fontsize=14)
plt.tight_layout()
plt.show()
"""))


def _add_section_4_manifest(nb):
    nb.cells.append(md("""## 4. 数据清洗与 Manifest 生成

直接处理几百个文件名容易出错，我们先用正则解析所有文件，生成一张结构化的 `manifest` 表。
这张表之后驱动所有数据加载。

**已知异常**：
- 部分文件名在 ID 和标签之间有空格（如 `0000028232 zz.png`）
- 少数文件扩展名是 `.jpg` 而非 `.png`

正则 `^(\\d+)\\s*(lt|yz|zz)\\.(png|jpg)$` 可以兼容这些变体。"""))
    nb.cells.append(code("""manifest = build_manifest(DATASET_DIR)
print(f"Manifest: {len(manifest)} 行, {manifest['patient_id'].nunique()} 个病人")
print("\\n标签分布:")
print(manifest['label'].value_counts())
manifest.head()
"""))
    nb.cells.append(code("""# 保存 manifest
manifest.to_csv(SPLITS_DIR / "manifest.csv", index=False)
print(f"保存到 {SPLITS_DIR / 'manifest.csv'}")

# 柱状图：类别分布
fig, ax = plt.subplots(figsize=(8, 5))
counts = manifest['label'].value_counts().reindex(["lt", "yz", "zz"])
ax.bar([LABEL_NAMES_CN[l] for l in counts.index], counts.values,
       color=["#4C72B0", "#DD8452", "#55A868"])
ax.set_ylabel("图片数")
ax.set_title("三类样本数量分布")
for i, v in enumerate(counts.values):
    ax.text(i, v + 2, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.show()
"""))


def _add_section_5_split(nb):
    nb.cells.append(md("""## 5. ⭐ 按病人划分数据集（Patient-level Split）

**这是医学影像 AI 最关键的教学点之一。**

### 错误做法（新手常犯）
把 641 张图随机 shuffle，按 7:1.5:1.5 划分到 train/val/test。

**为什么错？**
- 病人 A 的 3 张图（lt/yz/zz）中，可能 2 张进了训练集，1 张进了测试集
- 模型见过 A 的两张图之后，对 A 的第三张会"认脸"（病人的气管光照、血丝、拍摄角度都高度相似）
- 测试集精度会**虚高 10-20%**，真实部署时对新病人断崖式下跌
- 这叫 **Patient Leakage（患者泄漏）**

### 正确做法
按 `patient_id` 划分：同一病人的所有图片进同一个子集。这样测试集反映的是"遇到陌生病人时的真实能力"。"""))
    nb.cells.append(code("""train_df, val_df, test_df = split_by_patient(manifest, seed=42)

# 保存到 CSV（可复现）
train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

# 打印统计
for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    print(f"{name}: {df['patient_id'].nunique()} 病人, {len(df)} 图片, "
          f"类别分布 {dict(df['label'].value_counts())}")
"""))
    nb.cells.append(code("""# 显式验证：三个集合的 patient_id 没有任何交集
train_p = set(train_df['patient_id'])
val_p = set(val_df['patient_id'])
test_p = set(test_df['patient_id'])
assert len(train_p & val_p) == 0
assert len(train_p & test_p) == 0
assert len(val_p & test_p) == 0
print("✅ 病人级划分验证通过：三个子集无任何 patient_id 重叠")
"""))


def _add_section_6_augmentation(nb):
    nb.cells.append(md("""## 6. 数据增强策略

### 为什么需要数据增强
- 训练集 ~447 张对深度学习来说很小
- 增强用"随机变换"人工扩大训练样本的多样性
- 让模型学到更鲁棒的特征

### 这个项目里哪些增强可以用

| 增强 | 启用? | 理由 |
|------|-------|------|
| 旋转 ±15° | ✅ | 内镜操作自然抖动 |
| 色彩抖动 | ✅ | 不同设备、不同光照 |
| 随机裁剪缩放 | ✅ | 不同景深 |
| **水平翻转** | ❌ **严禁** | **`yz`=右，`zz`=左 — 翻转会破坏标签** |
| 垂直翻转 | ❌ | 内窥镜不会倒置 |

### ⚠️ 关键教学点：为什么不能水平翻转？
- `yz` 标签里的 `y` = **右**支气管
- `zz` 标签里的 `z` = **左**支气管
- 一张右侧图水平翻转后，解剖上就变成左侧了 → 标签应该变成 `zz`
- 如果训练时乱加水平翻转（但不改标签）→ 模型学到"左右不重要" → 真实推理时会把 `yz` 和 `zz` 混淆

**这种"标签语义敏感"的场景出现在很多医学 AI 任务**（左/右手、左/右眼、左/右肾…），记住这条规则比记住代码重要。"""))
    nb.cells.append(code("""train_tf = get_train_transforms()
eval_tf = get_eval_transforms()

print("训练集 transforms:")
print(train_tf)
print("\\n验证/测试集 transforms:")
print(eval_tf)
"""))
    nb.cells.append(code("""# 可视化增强效果：对同一张图跑 6 次训练 transform，看看随机性
from PIL import Image
sample_path = train_df.iloc[0]['path']
original = Image.open(sample_path).convert("RGB")

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes[0, 0].imshow(original)
axes[0, 0].set_title("原图")
axes[0, 0].axis("off")
for i in range(1, 6):
    augmented = train_tf(original).permute(1, 2, 0).numpy()
    # Denormalize for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    augmented = np.clip(augmented * std + mean, 0, 1)
    ax = axes[i // 3, i % 3]
    ax.imshow(augmented)
    ax.set_title(f"增强 #{i}")
    ax.axis("off")
plt.suptitle("同一张训练图的 5 种随机增强", fontsize=13)
plt.tight_layout()
plt.show()
"""))


def _add_section_7_dataloader(nb):
    nb.cells.append(md("""## 7. Dataset 与 DataLoader

`torch.utils.data.Dataset` 定义"怎么取一条样本"，`DataLoader` 定义"怎么批量、打乱、并行加载"。
"""))
    nb.cells.append(code("""BATCH_SIZE = 32
NUM_WORKERS = 4

train_ds = BronchoscopyDataset(train_df, transform=get_train_transforms())
val_ds = BronchoscopyDataset(val_df, transform=get_eval_transforms())
test_ds = BronchoscopyDataset(test_df, transform=get_eval_transforms())

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))

print(f"Train batches: {len(train_loader)}")
print(f"Val batches:   {len(val_loader)}")
print(f"Test batches:  {len(test_loader)}")

# 取一个 batch 看看形状
images, labels = next(iter(train_loader))
print(f"\\nBatch 形状: images={tuple(images.shape)}, labels={tuple(labels.shape)}")
print(f"Labels 示例: {labels[:8].tolist()}")
"""))


def _add_section_8_pytorch_recap(nb):
    nb.cells.append(md("""## 8. ⭐ PyTorch 快速回顾（针对 D 级别学员）

如果你跑过 MNIST 官方教程，下面这些概念应该眼熟。我们在这里系统过一遍：

### Tensor
- `torch.Tensor` 是 PyTorch 的基本数据结构，像 numpy array 但能在 GPU 上运行，且支持自动求导
- `tensor.to("cuda")` 把数据搬到 GPU

### nn.Module
- 所有模型/层都继承自 `nn.Module`
- `forward(x)` 方法定义前向传播
- `.parameters()` 返回所有可训练权重

### 训练循环四步曲
```python
optimizer.zero_grad()      # 1. 清空上一轮梯度
loss = criterion(model(x), y)  # 2. 前向 + 算损失
loss.backward()            # 3. 反向传播（自动求导）
optimizer.step()           # 4. 按梯度更新权重
```

### DataLoader
- 迭代时自动 batching、shuffling、并行加载

### 常用组件
| 组件 | 作用 |
|------|------|
| `nn.CrossEntropyLoss` | 多分类损失（内含 softmax，输入是 logits） |
| `torch.optim.AdamW` | 带权重衰减的 Adam 优化器（比 SGD 调参友好） |
| `torch.optim.lr_scheduler` | 学习率调度器 |

本项目全部训练工具（`train_one_epoch`, `validate`, `train_two_stage`）封装在 `src/train.py`。"""))


def _add_section_9_model(nb):
    nb.cells.append(md("""## 9. 模型搭建（ResNet-50 + 替换分类头）

### 为什么用 ResNet-50
- **有 ImageNet 预训练权重**：相当于模型已经"见过 100 万张自然图"，学到了通用的纹理/形状特征
- 对 641 张医学图来说，从头训练一个 CNN 基本不可能（数据不够），但在预训练基础上微调就非常现实

### 架构改造
- ResNet-50 原本输出 1000 类（ImageNet 类别）
- 我们把最后一层 `fc: Linear(2048 → 1000)` 替换成 `Sequential(Dropout(0.3), Linear(2048 → 3))`
- Dropout 用来缓解小数据集的过拟合"""))
    nb.cells.append(code("""model = build_resnet50(num_classes=3, pretrained=True, dropout=0.3)
model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")
print(f"可训练参数（当前）: {count_trainable_params(model):,}")

# 看一下模型结构的前几行和最后几行
print("\\n分类头 (model.fc):")
print(model.fc)
"""))


def _add_section_10_training(nb):
    nb.cells.append(md("""## 10. ⭐ 两阶段 Fine-tuning

### 阶段 1：冻结主干，只训分类头（5 epochs，lr=1e-3）
- 新的分类头是随机初始化的。如果一开始就训整个网络，大的梯度会破坏预训练主干的特征
- 先冻结主干，让分类头快速收敛到合理位置

### 阶段 2：解冻全部，小学习率微调（最多 20 epochs，lr=1e-4）
- 学习率比阶段 1 小 10 倍，否则会"忘掉"预训练学到的东西（灾难性遗忘）
- 带 Early Stopping：验证集 5 轮不提升就停
- 保存验证集最佳的权重到 `checkpoints/best_model.pt`

**预期**：阶段 1 结束时 val_acc 已经 >80%，阶段 2 会进一步提升到 90%+。"""))
    nb.cells.append(code("""CKPT_PATH = CKPT_DIR / "best_model.pt"

def log_epoch(info):
    print(f"  Stage {info['stage']} epoch {info['epoch']:2d} | "
          f"train_loss={info['train_loss']:.4f} train_acc={info['train_acc']:.4f} | "
          f"val_loss={info['val_loss']:.4f} val_acc={info['val_acc']:.4f}")

print("Starting two-stage fine-tuning...")
history = train_two_stage(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=DEVICE,
    stage1_epochs=5,
    stage1_lr=1e-3,
    stage2_epochs=20,
    stage2_lr=1e-4,
    weight_decay=1e-4,
    early_stopping_patience=5,
    checkpoint_path=CKPT_PATH,
    on_epoch_end=log_epoch,
)

print(f"\\nBest val accuracy: {history['best_val_acc']:.4f}")
print(f"Best checkpoint saved to: {CKPT_PATH}")
"""))


def _add_section_11_curves(nb):
    nb.cells.append(md("""## 11. 训练曲线

Loss 和 accuracy 两张图一起看，能判断：
- **理想**：train 和 val 都往好的方向走，差距小
- **过拟合**：train 继续变好，val 变差
- **欠拟合**：两条都在高位徘徊，说明模型容量不够或训练不足"""))
    nb.cells.append(code("""curves_path = OUTPUT_DIR / "training_curves.png"
plot_training_curves(history, curves_path)

from IPython.display import Image as IPImage
IPImage(str(curves_path))
"""))


def _add_section_12_evaluation(nb):
    nb.cells.append(md("""## 12. 测试集评估

**只在全部训练结束后，使用最佳 checkpoint 做一次性评估。**
测试集 **不参与**任何训练或超参数调整，否则就泄漏了。"""))
    nb.cells.append(code("""# 加载最佳权重
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

y_true, y_pred, y_proba = collect_predictions(model, test_loader, device=DEVICE)
class_names = ["lt", "yz", "zz"]
metrics = compute_metrics(y_true, y_pred, y_proba, class_names=class_names)

print(f"测试集 Accuracy: {metrics['accuracy']:.4f}")
print("\\n分类报告:")
print(metrics["classification_report"])
"""))
    nb.cells.append(code("""cm_path = OUTPUT_DIR / "confusion_matrix.png"
plot_confusion_matrix(
    metrics["confusion_matrix"], class_names=class_names, output_path=cm_path,
    title=f"测试集混淆矩阵 (Accuracy={metrics['accuracy']:.4f})",
)
from IPython.display import Image as IPImage
IPImage(str(cm_path))
"""))
    nb.cells.append(code("""# 置信度分布箱线图
proba_records = []
for true_id, proba_row in zip(y_true, y_proba):
    for class_id, p in enumerate(proba_row):
        proba_records.append({
            "true_label": class_names[true_id],
            "predicted_class": class_names[class_id],
            "probability": float(p),
        })
proba_df = pd.DataFrame(proba_records)

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=proba_df, x="true_label", y="probability", hue="predicted_class", ax=ax)
ax.set_title("按真实类别分组的 softmax 置信度分布")
ax.set_ylabel("Softmax probability")
plt.tight_layout()
plt.show()
"""))


def _add_section_13_gradcam(nb):
    nb.cells.append(md("""## 13. ⭐ Grad-CAM 可解释性分析

### 原理
Grad-CAM（Gradient-weighted Class Activation Mapping）用最后一个卷积层的梯度加权特征图，
生成一张热力图：红=重要，蓝=不重要。告诉你"模型做判断时，图片的哪些区域最关键"。

### 为什么对医学 AI 是刚需
- ✅ 如果热力图指向**解剖结构**（隆突 Y 形分叉、气管开口中心、气管环）→ 模型学对了
- ❌ 如果热力图指向**非解剖因素**（镜头反光、黑边、血丝）→ 模型学到了数据伪特征，真实部署会失败

### 实现
我们用 `pytorch-grad-cam` 库。目标层选 `model.layer4[-1]`（ResNet-50 最后一个残差块）。"""))
    nb.cells.append(code("""from PIL import Image

# 找正确分类 + 分错的样本各取 3 张
correct_indices = np.where(y_pred == y_true)[0]
wrong_indices = np.where(y_pred != y_true)[0]

# 从每类正确分类里各挑 3 张高置信度的
selected_correct = []
for class_id in range(3):
    mask = (y_true == class_id) & (y_pred == class_id)
    idx_in_class = np.where(mask)[0]
    # 按置信度排序，取 top 3
    confidences = y_proba[idx_in_class, class_id]
    top3 = idx_in_class[np.argsort(confidences)[::-1][:3]]
    selected_correct.extend(top3)

# 错分样本挑 3 张
selected_wrong = wrong_indices[:3].tolist()

selected = selected_correct + selected_wrong
print(f"选中 {len(selected)} 张样本做 Grad-CAM（{len(selected_correct)} 正确 + {len(selected_wrong)} 错误）")
"""))
    nb.cells.append(code("""target_layer = model.layer4[-1]
eval_tf = get_eval_transforms()
gradcam_dir = OUTPUT_DIR / "gradcam_examples"
gradcam_dir.mkdir(exist_ok=True)

fig, axes = plt.subplots(len(selected), 2, figsize=(10, 3.5 * len(selected)))
if len(selected) == 1:
    axes = axes.reshape(1, 2)

for row_idx, sample_idx in enumerate(selected):
    path = test_df.iloc[sample_idx]["path"]
    true_id = int(y_true[sample_idx])
    pred_id = int(y_pred[sample_idx])
    conf = float(y_proba[sample_idx, pred_id])

    original = Image.open(path).convert("RGB")
    tensor = eval_tf(original)
    overlay = make_gradcam_overlay(
        model=model,
        image_tensor=tensor,
        target_class=pred_id,
        target_layer=target_layer,
        original_pil=original,
        device=DEVICE,
    )

    # 保存单张
    save_path = gradcam_dir / f"sample_{sample_idx}_true={class_names[true_id]}_pred={class_names[pred_id]}.png"
    Image.fromarray(overlay).save(save_path)

    # 可视化
    tag = "✅" if pred_id == true_id else "❌"
    axes[row_idx, 0].imshow(original.resize((224, 224)))
    axes[row_idx, 0].set_title(f"{tag} 原图 | True={class_names[true_id]} Pred={class_names[pred_id]} ({conf:.2f})")
    axes[row_idx, 0].axis("off")
    axes[row_idx, 1].imshow(overlay)
    axes[row_idx, 1].set_title("Grad-CAM 叠加图")
    axes[row_idx, 1].axis("off")

plt.tight_layout()
plt.show()
print(f"\\n所有 Grad-CAM 图保存到 {gradcam_dir}")
"""))


def _add_section_14_error_analysis(nb):
    nb.cells.append(md("""## 14. 错误案例分析

打印模型分错的样本，看看错在哪。通常有几种情况：
- **视觉相似**：yz 和 zz 都是圆形管腔开口，易混淆
- **数据质量**：某张图模糊、光照差、镜头上有污渍
- **异常样本**：标签错误、角度异常"""))
    nb.cells.append(code("""errors = []
for i in wrong_indices:
    errors.append({
        "index": int(i),
        "path": test_df.iloc[i]["path"],
        "patient_id": test_df.iloc[i]["patient_id"],
        "true": class_names[int(y_true[i])],
        "pred": class_names[int(y_pred[i])],
        "confidence": float(y_proba[i, int(y_pred[i])]),
    })
errors_df = pd.DataFrame(errors).sort_values("confidence", ascending=False)
print(f"错分样本总数: {len(errors_df)}")
errors_df.head(10)
"""))


def _add_section_15_conclusion(nb):
    nb.cells.append(md("""## 15. 总结与下一步

### 做完这本 notebook，你学到了什么
1. ✅ 用正则解析杂乱文件名，生成结构化 manifest
2. ✅ **按病人级别划分数据集**，避免 Patient Leakage
3. ✅ 为医学图像设计合理的数据增强（懂得什么时候 **不能** 水平翻转）
4. ✅ ResNet-50 迁移学习的两阶段 fine-tune 技巧
5. ✅ 用 Confusion Matrix / Per-class 指标全面评估
6. ✅ 用 Grad-CAM 检查模型是不是"真的在看解剖结构"

### 后续可以尝试的方向
- 🎯 换模型架构：EfficientNet、ViT、Swin Transformer
- 🎯 更强的数据增强：AutoAugment、MixUp、CutMix（注意 MixUp 会混合标签，要配合改损失）
- 🎯 更细粒度的分类：在 yz 里识别右上叶 vs 右中间段
- 🎯 处理不平衡数据：Focal Loss、class-weighted loss
- 🎯 部署：导出 ONNX / TorchScript，接入 FastAPI 服务
- 🎯 可解释性升级：SHAP、LIME

### 现在去 `notebooks/demo.ipynb` 跑 Gradio demo 试试模型实时推理！"""))


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    nb.cells = []
    # Append all 15 sections
    _add_section_1_intro(nb)
    _add_section_2_setup(nb)
    _add_section_3_explore(nb)
    _add_section_4_manifest(nb)
    _add_section_5_split(nb)
    _add_section_6_augmentation(nb)
    _add_section_7_dataloader(nb)
    _add_section_8_pytorch_recap(nb)
    _add_section_9_model(nb)
    _add_section_10_training(nb)
    _add_section_11_curves(nb)
    _add_section_12_evaluation(nb)
    _add_section_13_gradcam(nb)
    _add_section_14_error_analysis(nb)
    _add_section_15_conclusion(nb)
    return nb


if __name__ == "__main__":
    nb = build()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook written to {OUTPUT_PATH} ({len(nb.cells)} cells)")
