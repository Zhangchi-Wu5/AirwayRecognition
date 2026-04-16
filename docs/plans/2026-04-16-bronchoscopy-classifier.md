# 气管镜部位识别模型 —— Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 训练一个 ResNet-50 迁移学习模型对气管镜 RGB 图片做三分类（隆突 / 右总支气管 / 左总支气管），产出一本可跑的教学 notebook 和一个 Gradio Web demo。

**Architecture:** 模块化 Python 源码（`src/`）+ 教学 notebook（`notebooks/`）。数据流：原始图片 → manifest.csv → 病人级划分（train/val/test）→ DataLoader → ResNet-50 两阶段 fine-tune → 评估 + Grad-CAM 可解释性 → Gradio demo。

**Tech Stack:** Python 3.10+, PyTorch 2.x, torchvision, pandas, scikit-learn, matplotlib, seaborn, pytorch-grad-cam, gradio, nbformat, pytest。

**Spec reference:** `docs/specs/2026-04-16-bronchoscopy-classifier-design.md`

**Project root:** `/Users/wuzhangchi/PycharmProjects/AirwayRecognition`

---

## Task 1: 项目骨架搭建

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `notebooks/.gitkeep`
- Create: `data_splits/.gitkeep`
- Create: `outputs/.gitkeep`
- Create: `checkpoints/.gitkeep`

- [ ] **Step 1: 创建目录结构**

```bash
cd /Users/wuzhangchi/PycharmProjects/AirwayRecognition
mkdir -p src tests notebooks data_splits outputs/gradcam_examples checkpoints
touch src/__init__.py tests/__init__.py
touch notebooks/.gitkeep data_splits/.gitkeep outputs/.gitkeep checkpoints/.gitkeep
```

- [ ] **Step 2: 写 .gitignore**

内容：

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
ENV/
.pytest_cache/
.ipynb_checkpoints/
*.egg-info/

# 数据集（737MB，不入 git；克隆者另行获取）
dataset/

# ML 产物（大文件/可重新生成，不入 git）
checkpoints/*.pt
checkpoints/*.pth
outputs/*.png
outputs/gradcam_examples/*.png

# IDE
.idea/
.vscode/
*.swp
.DS_Store

# 保留占位文件
!checkpoints/.gitkeep
!outputs/.gitkeep
!outputs/gradcam_examples/.gitkeep
```

把内容写入 `.gitignore`。

- [ ] **Step 3: 写 requirements.txt**

```text
# 深度学习
torch>=2.1.0
torchvision>=0.16.0

# 数据处理
pandas>=2.0.0
numpy>=1.24.0
Pillow>=10.0.0

# 评估与可视化
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0

# 可解释性
grad-cam>=1.4.8

# Demo
gradio>=4.0.0

# Jupyter
jupyterlab>=4.0.0
ipykernel>=6.25.0
nbformat>=5.9.0

# 测试
pytest>=7.4.0
```

把内容写入 `requirements.txt`。

- [ ] **Step 4: 验证目录结构**

```bash
cd /Users/wuzhangchi/PycharmProjects/AirwayRecognition
ls -la
```

Expected: 看到 `src/`, `tests/`, `notebooks/`, `data_splits/`, `outputs/`, `checkpoints/`, `dataset/`, `.gitignore`, `requirements.txt`, `README.md`。

- [ ] **Step 5: Commit**

```bash
cd /Users/wuzhangchi/PycharmProjects/AirwayRecognition
git add .gitignore requirements.txt src/ tests/ notebooks/ data_splits/ outputs/ checkpoints/
git commit -m "chore: initialize project structure with src/, tests/, notebooks/"
```

---

## Task 2: 数据清洗与 Manifest 生成

**Files:**
- Create: `src/data.py`（第一块：`parse_filename`, `build_manifest`）
- Create: `tests/test_data_parsing.py`

- [ ] **Step 1: 写测试 `tests/test_data_parsing.py`**

```python
"""Tests for filename parsing and manifest building."""
import pytest
from pathlib import Path
from src.data import parse_filename, build_manifest


class TestParseFilename:
    def test_parses_standard_filename(self):
        result = parse_filename("0000003926lt.png")
        assert result == {"patient_id": "0000003926", "label": "lt", "ext": "png"}

    def test_parses_filename_with_space(self):
        result = parse_filename("0000028232 zz.png")
        assert result == {"patient_id": "0000028232", "label": "zz", "ext": "png"}

    def test_parses_jpg_extension(self):
        result = parse_filename("0000032620 zz.jpg")
        assert result == {"patient_id": "0000032620", "label": "zz", "ext": "jpg"}

    def test_parses_all_three_labels(self):
        assert parse_filename("1234567890lt.png")["label"] == "lt"
        assert parse_filename("1234567890yz.png")["label"] == "yz"
        assert parse_filename("1234567890zz.png")["label"] == "zz"

    def test_returns_none_for_invalid_filename(self):
        assert parse_filename("not_a_valid_file.png") is None
        assert parse_filename("1234abc.png") is None
        assert parse_filename(".DS_Store") is None


class TestBuildManifest:
    def test_builds_manifest_from_dir(self, tmp_path):
        # Arrange: create fake dataset dir
        (tmp_path / "0000000001lt.png").touch()
        (tmp_path / "0000000001yz.png").touch()
        (tmp_path / "0000000001zz.png").touch()
        (tmp_path / "0000000002lt.png").touch()
        (tmp_path / ".DS_Store").touch()  # should be skipped

        # Act
        df = build_manifest(tmp_path)

        # Assert
        assert len(df) == 4
        assert set(df.columns) == {"patient_id", "label", "label_id", "path"}
        assert set(df["patient_id"].unique()) == {"0000000001", "0000000002"}
        assert set(df["label"].unique()) == {"lt", "yz"}

    def test_label_id_mapping(self, tmp_path):
        (tmp_path / "0000000001lt.png").touch()
        (tmp_path / "0000000001yz.png").touch()
        (tmp_path / "0000000001zz.png").touch()
        df = build_manifest(tmp_path)
        label_to_id = dict(zip(df["label"], df["label_id"]))
        assert label_to_id == {"lt": 0, "yz": 1, "zz": 2}
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd /Users/wuzhangchi/PycharmProjects/AirwayRecognition
pytest tests/test_data_parsing.py -v
```

Expected: FAIL，错误信息类似 `ImportError: cannot import name 'parse_filename' from 'src.data'` 或 `ModuleNotFoundError`。

- [ ] **Step 3: 写 `src/data.py` 第一块**

```python
"""Data loading, manifest generation, and patient-level splitting."""
import re
from pathlib import Path
from typing import Optional

import pandas as pd


LABEL_TO_ID = {"lt": 0, "yz": 1, "zz": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
LABEL_NAMES_CN = {"lt": "隆突", "yz": "右总支气管", "zz": "左总支气管"}

_FILENAME_PATTERN = re.compile(r"^(\d+)\s*(lt|yz|zz)\.(png|jpg)$", re.IGNORECASE)


def parse_filename(filename: str) -> Optional[dict]:
    """Parse a bronchoscopy image filename into components.

    Returns None if the filename doesn't match the expected pattern.
    Pattern: {patient_id}[optional-space]{lt|yz|zz}.{png|jpg}
    """
    match = _FILENAME_PATTERN.match(filename.strip())
    if match is None:
        return None
    patient_id, label, ext = match.groups()
    return {
        "patient_id": patient_id,
        "label": label.lower(),
        "ext": ext.lower(),
    }


def build_manifest(dataset_dir: Path) -> pd.DataFrame:
    """Scan dataset_dir and build a manifest dataframe.

    Columns: patient_id, label, label_id, path
    Skips unparseable files with a warning.
    """
    dataset_dir = Path(dataset_dir)
    rows = []
    skipped = []
    for file_path in sorted(dataset_dir.iterdir()):
        if not file_path.is_file():
            continue
        parsed = parse_filename(file_path.name)
        if parsed is None:
            skipped.append(file_path.name)
            continue
        rows.append({
            "patient_id": parsed["patient_id"],
            "label": parsed["label"],
            "label_id": LABEL_TO_ID[parsed["label"]],
            "path": str(file_path.resolve()),
        })
    if skipped:
        print(f"[build_manifest] Skipped {len(skipped)} unparseable files:")
        for name in skipped[:10]:
            print(f"  - {name}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")
    return pd.DataFrame(rows)
```

- [ ] **Step 4: 跑测试确认通过**

```bash
pytest tests/test_data_parsing.py -v
```

Expected: 全部 PASS（8 个测试）。

- [ ] **Step 5: 用真实数据跑一次 manifest 生成**

创建 `scripts/generate_manifest.py`:

```python
"""Generate manifest.csv from the real dataset."""
from pathlib import Path
from src.data import build_manifest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_CSV = PROJECT_ROOT / "data_splits" / "manifest.csv"

df = build_manifest(DATASET_DIR)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Manifest saved to {OUTPUT_CSV}")
print(f"Total rows: {len(df)}")
print(f"Unique patients: {df['patient_id'].nunique()}")
print(f"Label distribution:\n{df['label'].value_counts()}")
```

先创建 `scripts/` 目录并添加 `__init__.py`:

```bash
mkdir -p scripts
touch scripts/__init__.py
```

写入 `scripts/generate_manifest.py`，然后运行：

```bash
cd /Users/wuzhangchi/PycharmProjects/AirwayRecognition
python -m scripts.generate_manifest
```

Expected 输出（大致）：
```
Manifest saved to .../data_splits/manifest.csv
Total rows: 641
Unique patients: 213 (±2)
Label distribution:
yz    217
lt    216
zz    202
```

打开 `data_splits/manifest.csv` 抽查几行，确认 `patient_id`, `label`, `label_id`, `path` 都正确。

- [ ] **Step 6: Commit**

```bash
git add src/data.py tests/test_data_parsing.py scripts/ data_splits/manifest.csv
git commit -m "feat(data): add filename parser and manifest builder"
```

---

## Task 3: 病人级数据划分

**Files:**
- Modify: `src/data.py`（追加 `split_by_patient` 函数）
- Create: `tests/test_data_split.py`
- Modify: `scripts/generate_manifest.py` → 改名为 `scripts/build_splits.py` 并扩展

- [ ] **Step 1: 写测试 `tests/test_data_split.py`**

```python
"""Tests for patient-level data splitting."""
import pandas as pd
import pytest
from src.data import split_by_patient


def make_fake_manifest(n_patients: int) -> pd.DataFrame:
    """Create a fake manifest with n_patients, each having 3 images (lt/yz/zz)."""
    rows = []
    for i in range(n_patients):
        pid = f"P{i:06d}"
        for label, lid in [("lt", 0), ("yz", 1), ("zz", 2)]:
            rows.append({
                "patient_id": pid,
                "label": label,
                "label_id": lid,
                "path": f"/fake/{pid}{label}.png",
            })
    return pd.DataFrame(rows)


class TestSplitByPatient:
    def test_split_sizes_sum_to_total_patients(self):
        manifest = make_fake_manifest(100)
        train, val, test = split_by_patient(manifest, seed=42)
        total_patients = (
            train["patient_id"].nunique()
            + val["patient_id"].nunique()
            + test["patient_id"].nunique()
        )
        assert total_patients == 100

    def test_no_patient_overlap(self):
        manifest = make_fake_manifest(100)
        train, val, test = split_by_patient(manifest, seed=42)
        train_p = set(train["patient_id"])
        val_p = set(val["patient_id"])
        test_p = set(test["patient_id"])
        assert train_p & val_p == set()
        assert train_p & test_p == set()
        assert val_p & test_p == set()

    def test_split_ratio_approximately_70_15_15(self):
        manifest = make_fake_manifest(200)
        train, val, test = split_by_patient(manifest, seed=42)
        n_train = train["patient_id"].nunique()
        n_val = val["patient_id"].nunique()
        n_test = test["patient_id"].nunique()
        assert 130 <= n_train <= 150  # ~70% of 200
        assert 25 <= n_val <= 35      # ~15% of 200
        assert 25 <= n_test <= 35     # ~15% of 200

    def test_reproducibility_with_same_seed(self):
        manifest = make_fake_manifest(50)
        train1, val1, test1 = split_by_patient(manifest, seed=42)
        train2, val2, test2 = split_by_patient(manifest, seed=42)
        assert set(train1["patient_id"]) == set(train2["patient_id"])
        assert set(val1["patient_id"]) == set(val2["patient_id"])
        assert set(test1["patient_id"]) == set(test2["patient_id"])

    def test_all_images_for_patient_stay_together(self):
        manifest = make_fake_manifest(50)
        train, val, test = split_by_patient(manifest, seed=42)
        # For each patient, all 3 images must be in the same split
        for split_df in [train, val, test]:
            counts = split_df.groupby("patient_id").size()
            assert (counts == 3).all(), "Every patient in a split must have all 3 images"
```

- [ ] **Step 2: 跑测试确认失败**

```bash
pytest tests/test_data_split.py -v
```

Expected: FAIL（`split_by_patient` 未定义）。

- [ ] **Step 3: 实现 `split_by_patient`（追加到 `src/data.py`）**

追加到 `src/data.py` 末尾：

```python
import random


def split_by_patient(
    manifest: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split manifest into train/val/test by patient_id.

    All images from the same patient go to the same split.
    test_ratio is inferred as 1 - train_ratio - val_ratio.

    Returns: (train_df, val_df, test_df)
    """
    assert 0 < train_ratio < 1, "train_ratio must be in (0, 1)"
    assert 0 < val_ratio < 1, "val_ratio must be in (0, 1)"
    assert train_ratio + val_ratio < 1, "train + val must leave room for test"

    patient_ids = sorted(manifest["patient_id"].unique())
    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    n_total = len(patient_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_patients = set(patient_ids[:n_train])
    val_patients = set(patient_ids[n_train:n_train + n_val])
    test_patients = set(patient_ids[n_train + n_val:])

    train_df = manifest[manifest["patient_id"].isin(train_patients)].reset_index(drop=True)
    val_df = manifest[manifest["patient_id"].isin(val_patients)].reset_index(drop=True)
    test_df = manifest[manifest["patient_id"].isin(test_patients)].reset_index(drop=True)
    return train_df, val_df, test_df
```

- [ ] **Step 4: 跑测试确认通过**

```bash
pytest tests/test_data_split.py -v
```

Expected: 5 个测试全部 PASS。

- [ ] **Step 5: 扩展 `scripts/generate_manifest.py` 生成划分**

改名为 `scripts/build_splits.py`（功能合并）：

```python
"""Generate manifest.csv and train/val/test splits from the real dataset."""
from pathlib import Path
from src.data import build_manifest, split_by_patient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
SPLITS_DIR = PROJECT_ROOT / "data_splits"
SPLITS_DIR.mkdir(exist_ok=True)

# 1. Manifest
manifest = build_manifest(DATASET_DIR)
manifest_path = SPLITS_DIR / "manifest.csv"
manifest.to_csv(manifest_path, index=False)
print(f"Manifest saved: {manifest_path} ({len(manifest)} rows, {manifest['patient_id'].nunique()} patients)")

# 2. Splits
train_df, val_df, test_df = split_by_patient(manifest, seed=42)
train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

print(f"Train: {train_df['patient_id'].nunique()} patients, {len(train_df)} images")
print(f"Val:   {val_df['patient_id'].nunique()} patients, {len(val_df)} images")
print(f"Test:  {test_df['patient_id'].nunique()} patients, {len(test_df)} images")

# 3. Sanity check: no patient overlap
train_p = set(train_df["patient_id"])
val_p = set(val_df["patient_id"])
test_p = set(test_df["patient_id"])
assert not (train_p & val_p), "Patient leakage between train and val!"
assert not (train_p & test_p), "Patient leakage between train and test!"
assert not (val_p & test_p), "Patient leakage between val and test!"
print("Sanity check passed: no patient overlap.")
```

删掉旧 `scripts/generate_manifest.py`，创建新 `scripts/build_splits.py`，然后运行：

```bash
cd /Users/wuzhangchi/PycharmProjects/AirwayRecognition
rm scripts/generate_manifest.py
python -m scripts.build_splits
```

Expected: 生成 `manifest.csv`, `train.csv`, `val.csv`, `test.csv`，打印无 patient 重叠。

- [ ] **Step 6: Commit**

```bash
git add src/data.py tests/test_data_split.py scripts/build_splits.py data_splits/*.csv
git rm scripts/generate_manifest.py 2>/dev/null || true
git commit -m "feat(data): add patient-level split and generate train/val/test CSVs"
```

---

## Task 4: Dataset 类与 Transforms

**Files:**
- Modify: `src/data.py`（追加 `BronchoscopyDataset`, `get_train_transforms`, `get_eval_transforms`）
- Create: `tests/test_dataset.py`

- [ ] **Step 1: 写测试 `tests/test_dataset.py`**

```python
"""Tests for BronchoscopyDataset and transforms."""
import pandas as pd
import pytest
import torch
from PIL import Image
from src.data import BronchoscopyDataset, get_train_transforms, get_eval_transforms


@pytest.fixture
def fake_dataset_dir(tmp_path):
    """Create a tiny fake dataset with 3 images of different colors."""
    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        img = Image.new("RGB", (320, 240), color=color)
        img.save(tmp_path / f"000000000{i}lt.png")
    return tmp_path


@pytest.fixture
def fake_manifest_df(fake_dataset_dir):
    """Manifest for the 3-image fake dataset."""
    rows = []
    for i in range(3):
        rows.append({
            "patient_id": f"000000000{i}",
            "label": "lt",
            "label_id": 0,
            "path": str(fake_dataset_dir / f"000000000{i}lt.png"),
        })
    return pd.DataFrame(rows)


class TestTransforms:
    def test_train_transforms_output_shape(self, fake_dataset_dir):
        transform = get_train_transforms()
        img = Image.open(fake_dataset_dir / "0000000000lt.png")
        out = transform(img)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 224, 224)

    def test_eval_transforms_output_shape(self, fake_dataset_dir):
        transform = get_eval_transforms()
        img = Image.open(fake_dataset_dir / "0000000000lt.png")
        out = transform(img)
        assert out.shape == (3, 224, 224)

    def test_train_transforms_are_random(self, fake_dataset_dir):
        # Two runs should produce different tensors because of augmentation
        transform = get_train_transforms()
        img = Image.open(fake_dataset_dir / "0000000000lt.png")
        out1 = transform(img)
        out2 = transform(img)
        assert not torch.allclose(out1, out2), "Train transforms must be stochastic"

    def test_eval_transforms_are_deterministic(self, fake_dataset_dir):
        transform = get_eval_transforms()
        img = Image.open(fake_dataset_dir / "0000000000lt.png")
        out1 = transform(img)
        out2 = transform(img)
        assert torch.allclose(out1, out2), "Eval transforms must be deterministic"


class TestBronchoscopyDataset:
    def test_len(self, fake_manifest_df):
        ds = BronchoscopyDataset(fake_manifest_df, transform=get_eval_transforms())
        assert len(ds) == 3

    def test_getitem_returns_tensor_and_label(self, fake_manifest_df):
        ds = BronchoscopyDataset(fake_manifest_df, transform=get_eval_transforms())
        image, label = ds[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
        assert isinstance(label, int)
        assert label == 0

    def test_label_returns_label_id_not_string(self, fake_manifest_df):
        ds = BronchoscopyDataset(fake_manifest_df, transform=get_eval_transforms())
        _, label = ds[0]
        assert label == 0  # lt → 0
        assert not isinstance(label, str)
```

- [ ] **Step 2: 跑测试确认失败**

```bash
pytest tests/test_dataset.py -v
```

Expected: FAIL（未定义）。

- [ ] **Step 3: 实现 Dataset 和 transforms（追加到 `src/data.py`）**

首先确保 `src/data.py` 文件顶部的 import 块包含以下所有导入（Task 2/3 已有的保留，缺少的补上）：

```python
"""Data loading, manifest generation, and patient-level splitting."""
import random
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
```

然后在文件末尾追加以下代码（这些 import 已经在顶部，代码块里不再重复）：

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224


def get_train_transforms() -> transforms.Compose:
    """Augmentations for training.

    NOTE: Horizontal flip is intentionally disabled because yz (right) and
    zz (left) labels would swap under flipping.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms() -> transforms.Compose:
    """Deterministic preprocessing for validation/test/inference."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class BronchoscopyDataset(Dataset):
    """PyTorch Dataset that reads images listed in a manifest DataFrame.

    Expects columns: path, label_id.
    """

    def __init__(self, manifest: pd.DataFrame, transform=None):
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.manifest.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label_id"])
```

还需要在文件顶部添加 `import torch`。检查并补上。

- [ ] **Step 4: 跑测试确认通过**

```bash
pytest tests/test_dataset.py -v
```

Expected: 7 个测试全部 PASS。

- [ ] **Step 5: 跑所有数据测试**

```bash
pytest tests/test_data_parsing.py tests/test_data_split.py tests/test_dataset.py -v
```

Expected: 20 个测试全部 PASS。

- [ ] **Step 6: Commit**

```bash
git add src/data.py tests/test_dataset.py
git commit -m "feat(data): add BronchoscopyDataset and train/eval transforms"
```

---

## Task 5: 模型构建

**Files:**
- Create: `src/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: 写测试 `tests/test_models.py`**

```python
"""Tests for model construction and freeze/unfreeze utilities."""
import pytest
import torch
from src.models import build_resnet50, freeze_backbone, unfreeze_all, count_trainable_params


class TestBuildResnet50:
    def test_returns_module(self):
        model = build_resnet50(num_classes=3, pretrained=False)
        assert isinstance(model, torch.nn.Module)

    def test_output_shape(self):
        model = build_resnet50(num_classes=3, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 3)

    def test_classifier_head_has_dropout_and_linear(self):
        model = build_resnet50(num_classes=3, pretrained=False)
        # model.fc should be Sequential(Dropout, Linear(2048, 3))
        head = model.fc
        assert isinstance(head, torch.nn.Sequential)
        modules = list(head.modules())
        has_dropout = any(isinstance(m, torch.nn.Dropout) for m in modules)
        linears = [m for m in modules if isinstance(m, torch.nn.Linear)]
        assert has_dropout
        assert len(linears) == 1
        assert linears[0].out_features == 3


class TestFreezeUnfreeze:
    def test_freeze_backbone_only_trains_head(self):
        model = build_resnet50(num_classes=3, pretrained=False)
        freeze_backbone(model)
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        # Only fc layer parameters should be trainable
        for name in trainable:
            assert name.startswith("fc."), f"Unexpected trainable param: {name}"

    def test_unfreeze_all_makes_everything_trainable(self):
        model = build_resnet50(num_classes=3, pretrained=False)
        freeze_backbone(model)
        unfreeze_all(model)
        for name, p in model.named_parameters():
            assert p.requires_grad, f"{name} is still frozen"

    def test_count_trainable_params_decreases_after_freeze(self):
        model = build_resnet50(num_classes=3, pretrained=False)
        before = count_trainable_params(model)
        freeze_backbone(model)
        after = count_trainable_params(model)
        assert after < before
        # fc is (2048 → 3) + bias = 6147; check after is small
        assert after < 10_000
```

- [ ] **Step 2: 跑测试确认失败**

```bash
pytest tests/test_models.py -v
```

Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现 `src/models.py`**

```python
"""Model construction and parameter freezing utilities."""
import torch
import torch.nn as nn
from torchvision import models


def build_resnet50(num_classes: int = 3, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """Build a ResNet-50 with a replaced classifier head.

    Original head: Linear(2048, 1000)
    New head: Sequential(Dropout(p=dropout), Linear(2048, num_classes))
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the classifier head (model.fc)."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")


def unfreeze_all(model: nn.Module) -> None:
    """Mark all parameters as trainable."""
    for param in model.parameters():
        param.requires_grad = True


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

- [ ] **Step 4: 跑测试确认通过**

```bash
pytest tests/test_models.py -v
```

Expected: 6 个测试全部 PASS。（注意：首次运行会下载 ImageNet 权重大约 100MB，这里测试用 `pretrained=False` 所以不需要下载。）

- [ ] **Step 5: Commit**

```bash
git add src/models.py tests/test_models.py
git commit -m "feat(models): add ResNet-50 builder with replaceable head and freeze utils"
```

---

## Task 6: 训练循环

**Files:**
- Create: `src/train.py`
- Create: `tests/test_train.py`

- [ ] **Step 1: 写测试 `tests/test_train.py`**

```python
"""Tests for training utilities (set_seed, train_epoch, validate_epoch)."""
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.train import set_seed, train_one_epoch, validate


def make_tiny_loader(n_samples: int = 16, batch_size: int = 4):
    x = torch.randn(n_samples, 3, 224, 224)
    y = torch.randint(0, 3, (n_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def make_tiny_model():
    # Tiny CNN that accepts (3, 224, 224) and outputs 3 classes
    return torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(3, 3),
    )


def test_set_seed_reproducible():
    set_seed(42)
    a = torch.randn(5)
    set_seed(42)
    b = torch.randn(5)
    assert torch.allclose(a, b)


def test_train_one_epoch_returns_loss_and_acc():
    model = make_tiny_model()
    loader = make_tiny_loader()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    loss, acc = train_one_epoch(model, loader, optimizer, criterion, device="cpu")
    assert 0.0 <= loss < 100.0
    assert 0.0 <= acc <= 1.0


def test_validate_returns_loss_and_acc():
    model = make_tiny_model()
    loader = make_tiny_loader()
    criterion = torch.nn.CrossEntropyLoss()
    loss, acc = validate(model, loader, criterion, device="cpu")
    assert 0.0 <= loss < 100.0
    assert 0.0 <= acc <= 1.0


def test_train_reduces_loss_on_memorizable_data():
    """Sanity check: training for a few epochs on tiny data should reduce loss."""
    set_seed(42)
    model = make_tiny_model()
    loader = make_tiny_loader(n_samples=8, batch_size=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()
    initial_loss, _ = validate(model, loader, criterion, device="cpu")
    for _ in range(5):
        train_one_epoch(model, loader, optimizer, criterion, device="cpu")
    final_loss, _ = validate(model, loader, criterion, device="cpu")
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} → {final_loss}"
```

- [ ] **Step 2: 跑测试确认失败**

```bash
pytest tests/test_train.py -v
```

Expected: FAIL。

- [ ] **Step 3: 实现 `src/train.py`**

```python
"""Training loop with two-stage fine-tuning, early stopping, and checkpointing."""
import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models import freeze_backbone, unfreeze_all


def set_seed(seed: int = 42) -> None:
    """Set seeds across Python random, NumPy, and PyTorch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = "cuda",
) -> tuple[float, float]:
    """Train for one epoch; return (avg_loss, accuracy)."""
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += images.size(0)
    return total_loss / total_count, total_correct / total_count


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str = "cuda",
) -> tuple[float, float]:
    """Evaluate on a loader; return (avg_loss, accuracy)."""
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_count += images.size(0)
    return total_loss / total_count, total_correct / total_count


def train_two_stage(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cuda",
    stage1_epochs: int = 5,
    stage1_lr: float = 1e-3,
    stage2_epochs: int = 20,
    stage2_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 5,
    checkpoint_path: Optional[Path] = None,
    on_epoch_end: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Run two-stage fine-tuning.

    Stage 1: Freeze backbone, train head for stage1_epochs.
    Stage 2: Unfreeze all, train for up to stage2_epochs with early stopping.

    Returns: history dict with lists of per-epoch metrics.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    history = {
        "stage": [], "epoch": [],
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }

    # Stage 1: head only
    freeze_backbone(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=stage1_lr, weight_decay=weight_decay)
    for epoch in range(1, stage1_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        _record(history, 1, epoch, train_loss, train_acc, val_loss, val_acc)
        if on_epoch_end:
            on_epoch_end({"stage": 1, "epoch": epoch, "train_loss": train_loss,
                          "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

    # Stage 2: full network
    unfreeze_all(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=stage2_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage2_epochs)
    best_val_acc = -1.0
    patience_counter = 0
    for epoch in range(1, stage2_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        _record(history, 2, epoch, train_loss, train_acc, val_loss, val_acc)
        if on_epoch_end:
            on_epoch_end({"stage": 2, "epoch": epoch, "train_loss": train_loss,
                          "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at stage 2 epoch {epoch} (best val_acc={best_val_acc:.4f})")
                break
    history["best_val_acc"] = best_val_acc
    return history


def _record(history, stage, epoch, train_loss, train_acc, val_loss, val_acc):
    history["stage"].append(stage)
    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
```

- [ ] **Step 4: 跑测试确认通过**

```bash
pytest tests/test_train.py -v
```

Expected: 4 个测试全部 PASS（最后一个可能慢几秒）。

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "feat(train): add two-stage fine-tune loop with early stopping and seeding"
```

---

## Task 7: 评估

**Files:**
- Create: `src/evaluate.py`
- Create: `tests/test_evaluate.py`

- [ ] **Step 1: 写测试 `tests/test_evaluate.py`**

```python
"""Tests for evaluation: collect_predictions, compute_metrics."""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import collect_predictions, compute_metrics


def make_tiny_loader(n_samples=12, batch_size=4):
    x = torch.randn(n_samples, 3, 224, 224)
    y = torch.randint(0, 3, (n_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def make_tiny_model():
    return torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(3, 3),
    )


def test_collect_predictions_returns_correct_shape():
    model = make_tiny_model()
    loader = make_tiny_loader(n_samples=12, batch_size=4)
    y_true, y_pred, y_proba = collect_predictions(model, loader, device="cpu")
    assert len(y_true) == 12
    assert len(y_pred) == 12
    assert y_proba.shape == (12, 3)
    assert np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-5)


def test_compute_metrics_returns_expected_keys():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])  # 2 wrong
    y_proba = np.eye(3)[y_pred]
    metrics = compute_metrics(y_true, y_pred, y_proba, class_names=["lt", "yz", "zz"])
    assert "accuracy" in metrics
    assert "confusion_matrix" in metrics
    assert "classification_report" in metrics
    assert abs(metrics["accuracy"] - 4 / 6) < 1e-6
    cm = metrics["confusion_matrix"]
    assert cm.shape == (3, 3)
    assert cm.sum() == 6
```

- [ ] **Step 2: 跑测试确认失败**

```bash
pytest tests/test_evaluate.py -v
```

Expected: FAIL.

- [ ] **Step 3: 实现 `src/evaluate.py`**

```python
"""Evaluation utilities: collect predictions and compute metrics."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on loader; return (y_true, y_pred, y_proba) as numpy arrays."""
    model = model.to(device)
    model.eval()
    all_true, all_pred, all_proba = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            proba = torch.softmax(logits, dim=1)
            preds = proba.argmax(dim=1)
            all_true.append(labels.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_proba.append(proba.cpu().numpy())
    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_proba),
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
) -> dict:
    """Compute accuracy, confusion matrix, and classification report."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))),
        "classification_report": classification_report(
            y_true, y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            digits=4,
            zero_division=0,
        ),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
```

- [ ] **Step 4: 跑测试确认通过**

```bash
pytest tests/test_evaluate.py -v
```

Expected: 2 个测试 PASS。

- [ ] **Step 5: Commit**

```bash
git add src/evaluate.py tests/test_evaluate.py
git commit -m "feat(evaluate): add prediction collection and metrics computation"
```

---

## Task 8: 可视化

**Files:**
- Create: `src/viz.py`
- Create: `tests/test_viz.py`

- [ ] **Step 1: 写测试 `tests/test_viz.py`**

```python
"""Tests for visualization helpers."""
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests
import matplotlib.pyplot as plt
import numpy as np
import pytest
from pathlib import Path

from src.viz import plot_training_curves, plot_confusion_matrix


def test_plot_training_curves_saves_png(tmp_path):
    history = {
        "epoch": [1, 2, 3],
        "stage": [1, 1, 1],
        "train_loss": [1.0, 0.5, 0.3],
        "train_acc": [0.4, 0.7, 0.9],
        "val_loss": [1.1, 0.6, 0.4],
        "val_acc": [0.3, 0.6, 0.85],
    }
    output = tmp_path / "curves.png"
    plot_training_curves(history, output)
    assert output.exists()
    assert output.stat().st_size > 1000


def test_plot_confusion_matrix_saves_png(tmp_path):
    cm = np.array([[10, 1, 0], [2, 8, 1], [0, 1, 9]])
    output = tmp_path / "cm.png"
    plot_confusion_matrix(cm, class_names=["lt", "yz", "zz"], output_path=output)
    assert output.exists()
    assert output.stat().st_size > 1000
```

- [ ] **Step 2: 跑测试确认失败**

```bash
pytest tests/test_viz.py -v
```

Expected: FAIL。

- [ ] **Step 3: 实现 `src/viz.py`**

```python
"""Visualization: training curves, confusion matrix, Grad-CAM overlays."""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image


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
```

- [ ] **Step 4: 跑测试确认通过**

```bash
pytest tests/test_viz.py -v
```

Expected: 2 个测试 PASS。（Grad-CAM 函数没写单元测试，会在 notebook 里端到端验证。）

- [ ] **Step 5: 跑全部测试**

```bash
pytest tests/ -v
```

Expected: 24 个测试全部 PASS。

- [ ] **Step 6: Commit**

```bash
git add src/viz.py tests/test_viz.py
git commit -m "feat(viz): add training curves, confusion matrix, and Grad-CAM overlay"
```

---

## Task 9: 主教学 Notebook

**Files:**
- Create: `scripts/build_main_notebook.py`（用 nbformat 生成 `bronchoscopy_classifier.ipynb`）
- Create: `notebooks/bronchoscopy_classifier.ipynb`（由上面脚本生成）

**说明**：主 notebook 一次性写 500+ 行的 JSON 非常痛苦。我们写一个 Python 脚本用 `nbformat` 库程序化生成 notebook。这样：
- notebook 内容在 Python 中维护，易读
- 运行一次生成 `.ipynb`，之后用户可以用 JupyterLab 直接打开修改
- 需要重新生成时再跑脚本

- [ ] **Step 1: 创建 notebook 生成脚本 `scripts/build_main_notebook.py`**

**文件较长，分 3 个子步骤完成：基础框架 → 注入各章节内容 → 运行生成。**

先写骨架（定义好辅助函数 `md()`, `code()` 和主函数）：

```python
"""Build the main teaching notebook (bronchoscopy_classifier.ipynb) using nbformat."""
from pathlib import Path
import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "notebooks" / "bronchoscopy_classifier.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    nb.cells = []
    # Sections will be appended below
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
```

写入 `scripts/build_main_notebook.py`。暂时各 section 函数未定义，下面的步骤补上。

- [ ] **Step 2: 补上章节 1-4 的函数**

追加到 `scripts/build_main_notebook.py`（在 `if __name__` 之前）：

```python
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
from src.viz import plot_training_curves, plot_confusion_matrix, make_gradcam_overlay

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
```

- [ ] **Step 3: 补上章节 5-8 的函数**

追加到 `scripts/build_main_notebook.py`：

```python
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
```

- [ ] **Step 4: 补上章节 9-15 的函数**

追加：

```python
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
```

- [ ] **Step 5: 运行生成 notebook**

```bash
cd /Users/wuzhangchi/PycharmProjects/AirwayRecognition
python -m scripts.build_main_notebook
```

Expected 输出：
```
Notebook written to .../notebooks/bronchoscopy_classifier.ipynb (N cells)
```

- [ ] **Step 6: 在 JupyterLab 中打开验证**

```bash
jupyter lab notebooks/bronchoscopy_classifier.ipynb
```

手动目视检查：
- 所有 15 章节标题显示正确
- 代码 cell 和 markdown cell 交替得当
- 没有损坏的 cell

（不需要跑完整个 notebook，那是交付物使用环节。但可以跑前两章确认 import 没报错。）

- [ ] **Step 7: Commit**

```bash
git add scripts/build_main_notebook.py notebooks/bronchoscopy_classifier.ipynb
git commit -m "feat(notebook): add main teaching notebook with 15 sections"
```

---

## Task 10: Gradio Demo Notebook

**Files:**
- Create: `scripts/build_demo_notebook.py`
- Create: `notebooks/demo.ipynb`（由上面脚本生成）

- [ ] **Step 1: 写 `scripts/build_demo_notebook.py`**

```python
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
        md("""# 🎬 气管镜部位识别 — Gradio Demo

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
    # Gradio Label 期望 {label: prob}
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
    title="🫁 气管镜部位识别",
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
```

- [ ] **Step 2: 运行生成 demo notebook**

```bash
cd /Users/wuzhangchi/PycharmProjects/AirwayRecognition
python -m scripts.build_demo_notebook
```

- [ ] **Step 3: Commit**

```bash
git add scripts/build_demo_notebook.py notebooks/demo.ipynb
git commit -m "feat(demo): add Gradio web demo notebook"
```

---

## Task 11: README 与冒烟测试

**Files:**
- Modify: `README.md`
- Create: `tests/test_smoke.py`

- [ ] **Step 1: 写冒烟测试 `tests/test_smoke.py`**

```python
"""Smoke tests: verify the whole data pipeline works end-to-end on real data."""
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.data import (
    BronchoscopyDataset, build_manifest, get_eval_transforms,
    split_by_patient,
)
from src.models import build_resnet50

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"


@pytest.mark.skipif(not DATASET_DIR.exists(), reason="Real dataset not available")
class TestRealDataPipeline:
    def test_manifest_has_expected_counts(self):
        df = build_manifest(DATASET_DIR)
        # Allow some slack because we verified 641 rows earlier
        assert 600 <= len(df) <= 700
        assert set(df["label"].unique()) == {"lt", "yz", "zz"}
        assert 200 <= df["patient_id"].nunique() <= 230

    def test_split_produces_three_nonempty_sets_without_overlap(self):
        df = build_manifest(DATASET_DIR)
        train, val, test = split_by_patient(df, seed=42)
        assert len(train) > 0 and len(val) > 0 and len(test) > 0
        train_p = set(train["patient_id"])
        val_p = set(val["patient_id"])
        test_p = set(test["patient_id"])
        assert not (train_p & val_p)
        assert not (train_p & test_p)
        assert not (val_p & test_p)

    def test_dataloader_produces_correct_batch(self):
        df = build_manifest(DATASET_DIR)
        train, _, _ = split_by_patient(df, seed=42)
        ds = BronchoscopyDataset(train, transform=get_eval_transforms())
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        images, labels = next(iter(loader))
        assert images.shape == (4, 3, 224, 224)
        assert labels.shape == (4,)
        assert labels.dtype == torch.int64

    def test_model_forward_on_real_batch(self):
        df = build_manifest(DATASET_DIR)
        train, _, _ = split_by_patient(df, seed=42)
        ds = BronchoscopyDataset(train, transform=get_eval_transforms())
        loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
        images, _ = next(iter(loader))
        model = build_resnet50(num_classes=3, pretrained=False)
        model.eval()
        with torch.no_grad():
            logits = model(images)
        assert logits.shape == (2, 3)
```

- [ ] **Step 2: 跑冒烟测试**

```bash
pytest tests/test_smoke.py -v
```

Expected: 4 个测试 PASS。如果 `pretrained=True` 的测试太慢可以跳过，这里我们故意用 `pretrained=False` 避免网络下载。

- [ ] **Step 3: 跑全部测试**

```bash
pytest tests/ -v
```

Expected: 28 个测试全部 PASS。

- [ ] **Step 4: 写 README**

替换 `README.md` 的内容：

```markdown
# 气管镜部位识别 Airway Recognition

用 ResNet-50 迁移学习做气管镜图片三分类：隆突（lt）/ 右总支气管（yz）/ 左总支气管（zz）。

## 项目结构

```
AirwayRecognition/
├── dataset/                # 原始图片（641 张，.png/.jpg）
├── src/                    # Python 模块
│   ├── data.py             # manifest / 病人级划分 / Dataset / transforms
│   ├── models.py           # ResNet-50 + 替换分类头
│   ├── train.py            # 两阶段 fine-tune + early stopping
│   ├── evaluate.py         # 评估指标
│   └── viz.py              # 训练曲线 / 混淆矩阵 / Grad-CAM
├── tests/                  # pytest 单元测试和冒烟测试
├── notebooks/
│   ├── bronchoscopy_classifier.ipynb   # 主教学 notebook（15 章节）
│   └── demo.ipynb                      # Gradio Web demo
├── scripts/
│   ├── build_splits.py
│   ├── build_main_notebook.py
│   └── build_demo_notebook.py
├── data_splits/            # manifest.csv + train/val/test.csv
├── checkpoints/            # 训练好的模型权重（.gitignore）
├── outputs/                # 图表产物（.gitignore）
└── docs/
    ├── specs/              # 设计文档
    └── plans/              # 实施计划
```

## 快速开始

### 1. 环境准备

```bash
# 建议 Python 3.10+，强烈推荐用虚拟环境
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 生成数据划分

```bash
python -m scripts.build_splits
```

会在 `data_splits/` 生成 `manifest.csv`, `train.csv`, `val.csv`, `test.csv`。

### 3. 跑测试（可选但推荐）

```bash
pytest tests/ -v
```

### 4. 运行训练 notebook

```bash
jupyter lab notebooks/bronchoscopy_classifier.ipynb
```

依次运行所有 cell。在 A100 上约 15 分钟完成。训练好的最佳权重保存到 `checkpoints/best_model.pt`。

### 5. 启动 Gradio Demo

```bash
jupyter lab notebooks/demo.ipynb
```

运行所有 cell 后会在 `http://localhost:7860` 打开 web 界面。

## 关键设计要点

- **病人级划分**：同一病人的所有图片进同一子集（train/val/test），防止 Patient Leakage
- **禁用水平翻转**：`yz`=右，`zz`=左，翻转会破坏标签语义
- **两阶段 Fine-tuning**：先冻结主干训分类头，再解冻整网小学习率微调
- **Grad-CAM 可解释性**：检查模型是否在看解剖结构而不是伪特征

## 开发说明

重新生成 notebook：
```bash
python -m scripts.build_main_notebook
python -m scripts.build_demo_notebook
```

跑单个测试文件：
```bash
pytest tests/test_data_split.py -v
```

## 设计与实施文档

- 设计文档：`docs/specs/2026-04-16-bronchoscopy-classifier-design.md`
- 实施计划：`docs/plans/2026-04-16-bronchoscopy-classifier.md`
```

把上面内容写入 `README.md`（替换原有内容）。

- [ ] **Step 5: 最终 Commit**

```bash
git add README.md tests/test_smoke.py
git commit -m "docs: add README with setup and run instructions; add smoke tests"
```

- [ ] **Step 6: 最终验证（sanity check）**

```bash
cd /Users/wuzhangchi/PycharmProjects/AirwayRecognition
# 1. 所有测试通过
pytest tests/ -v
# 2. 目录结构完整
ls src/ tests/ notebooks/ scripts/ data_splits/ docs/specs/ docs/plans/
# 3. git 状态干净
git status
```

Expected:
- 28 个测试全部 PASS
- 所有目录非空
- `git status` 显示 `nothing to commit, working tree clean`

---

## 完成标志

- ✅ 11 个任务全部完成
- ✅ 所有单元测试和冒烟测试通过
- ✅ `data_splits/*.csv` 已生成，按病人划分无泄漏
- ✅ `notebooks/bronchoscopy_classifier.ipynb` 包含 15 个章节
- ✅ `notebooks/demo.ipynb` 可启动 Gradio 服务
- ✅ README 有完整运行指引
- ✅ git 历史每个任务一到多个原子 commit

## 下一步（本计划之外）

用户打开主 notebook 运行全部 cell，期望：
- 阶段 1（冻结主干）5 epochs，val_acc 上升到 ~0.80+
- 阶段 2（全网络微调）10-20 epochs 收敛，val_acc ~0.90+
- 测试集 accuracy ≥ 0.90
- Grad-CAM 热力图聚焦在解剖关键区域

如有偏差，按 spec 第 14 节"风险与应对"调整。
