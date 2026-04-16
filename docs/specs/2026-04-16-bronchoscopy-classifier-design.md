# 气管镜部位识别模型 —— 设计文档

- **作者**: Carver Wu (carverwu1997@gmail.com)
- **日期**: 2026-04-16
- **项目路径**: `/Users/wuzhangchi/PycharmProjects/AirwayRecognition`
- **状态**: Design — Pending Implementation Plan

---

## 1. 目标与范围

### 1.1 目标

训练一个深度学习图像分类模型，识别输入的气管镜 RGB 图片属于以下三个解剖部位之一：

| 代码 | 中文名 | 含义 |
|------|--------|------|
| `lt` | 隆突（Carina） | 左右主支气管的分叉处 |
| `yz` | 右总支气管（Right Main Bronchus） | 分出右上叶支 + 右中间段 |
| `zz` | 左总支气管（Left Main Bronchus） | 通向左上下肺开口 |

### 1.2 范围

**本项目是一个以深度学习学习为核心目标的教学项目**，重点在于：
- 亲手走一遍完整的 CNN 迁移学习流程
- 理解医学影像数据处理中的特有陷阱（尤其是 Patient Leakage）
- 产出可交互的 Gradio Web Demo，可用于科室内部演示

**本项目不涉及**：
- 真实临床系统集成（不接入 PACS、气管镜报告系统等）
- 生产级 MLOps 基础设施（不做 Docker、CI/CD、模型监控）
- 严格的合规/审计流程
- 论文级别的对比实验（后续可作为扩展追加方案 2）

### 1.3 成功标准

| 维度 | 指标 |
|------|------|
| 模型精度 | 测试集 Accuracy ≥ 90%（预期 92-97%） |
| 可解释性 | Grad-CAM 热力图能聚焦在解剖相关区域 |
| 可复现性 | 固定随机种子，多次运行结果一致 |
| 代码质量 | 模块化、有注释，Notebook 带中文教学讲解 |
| 可演示性 | Gradio demo 可在本地运行、上传图片即可预测 |

---

## 2. 数据集

### 2.1 现状

- **位置**: `/Users/wuzhangchi/PycharmProjects/AirwayRecognition/dataset/`
- **总数**: 641 张图片
- **分布**:
  - `lt`（隆突）: ~216 张
  - `yz`（右总支气管）: ~217 张
  - `zz`（左总支气管）: ~202 张
- **图像格式**: 大部分 PNG，少量 JPG
- **图像性质**: 自然 RGB 内窥镜图像（非 CT/MRI 灰度断层）
- **病人数量**: 约 213 人，每人约 3 张（lt + yz + zz 各一张）

### 2.2 命名规范

标准格式：`{patient_id}{label}.{ext}`，其中：
- `patient_id`: 10 位数字（如 `0000003926`）
- `label`: `lt` / `yz` / `zz`
- `ext`: `png` / `jpg`

**已知异常**：
- 部分文件名在 ID 和标签之间有空格（如 `0000028232 zz.png`）
- 少数文件扩展名为 `.jpg` 而非 `.png`
- 需要在数据清洗阶段用正则解析并归一化

**解析正则**：`^(\d+)\s*(lt|yz|zz)\.(png|jpg)$`

### 2.3 数据集的关键约束

- **左右标签语义敏感**：`yz`=右、`zz`=左。任何水平翻转都会翻转解剖左右，破坏标签 → **严禁水平翻转**
- **同一病人的 3 张图高度相关**：光照、设备型号、操作手法一致 → **必须按病人划分数据集**（Patient-level split）

---

## 3. 技术栈

| 组件 | 技术选型 | 版本建议 |
|------|---------|----------|
| 深度学习框架 | PyTorch | 2.x |
| 模型库 | torchvision（基础）、timm（可选） | latest |
| 数据处理 | pandas, numpy, Pillow | latest |
| 评估 | scikit-learn | latest |
| 可视化 | matplotlib, seaborn | latest |
| 可解释性 | pytorch-grad-cam | latest |
| Web Demo | Gradio | 4.x |
| 开发环境 | Jupyter Lab / Notebook | latest |
| 硬件 | NVIDIA A100（用户自有服务器） | - |

---

## 4. 项目目录结构

```
AirwayRecognition/
├── .git/
├── .gitignore                       # 排除 checkpoints/、outputs/、__pycache__ 等
├── README.md                        # 项目说明 + 运行指引
├── requirements.txt                 # Python 依赖清单
│
├── dataset/                         # 原始数据（641 张图，只读）
│
├── src/                             # 可复用 Python 模块
│   ├── __init__.py
│   ├── data.py                      # Dataset、manifest、病人级划分
│   ├── models.py                    # 模型构建
│   ├── train.py                     # 两阶段训练循环
│   ├── evaluate.py                  # 评估指标
│   └── viz.py                       # 可视化（Grad-CAM、混淆矩阵、曲线）
│
├── notebooks/
│   ├── bronchoscopy_classifier.ipynb   # 主教学 notebook（训练 + 评估）
│   └── demo.ipynb                      # Gradio Web demo
│
├── data_splits/                     # 数据集划分（进 git，保证可复现）
│   ├── manifest.csv
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── checkpoints/                     # .gitignore 排除
│   └── best_model.pt
│
├── outputs/                         # .gitignore 排除
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   └── gradcam_examples/
│
└── docs/
    └── specs/
        └── 2026-04-16-bronchoscopy-classifier-design.md   # 本文档
```

---

## 5. 数据管道设计

### 5.1 数据清洗与 Manifest 生成

**目标**：把 641 个文件解析成一张结构化的 manifest 表。

**步骤**：
1. 遍历 `dataset/` 下所有 `.png` / `.jpg` 文件
2. 用正则 `^(\d+)\s*(lt|yz|zz)\.(png|jpg)$` 解析
3. 对每个文件产出一行：`{patient_id, label, label_id, path}`
4. 标签映射：`lt=0, yz=1, zz=2`
5. 无法解析的文件打印警告、排除
6. 保存到 `data_splits/manifest.csv`

### 5.2 病人级数据划分（关键点）

**划分比例**: 70% train / 15% val / 15% test（按病人数，不是按图片数）

**步骤**：
1. 从 manifest 提取唯一的 `patient_id` 列表（约 213 人）
2. 固定随机种子 `42`，对病人列表 shuffle
3. 按 70/15/15 切分病人列表
4. 同一病人的所有图片进同一子集
5. 产出：
   - `data_splits/train.csv`（~149 病人，~447 张图）
   - `data_splits/val.csv`（~32 病人，~96 张图）
   - `data_splits/test.csv`（~32 病人，~96 张图）
6. 在 notebook 里显式验证：`set(train_patients) ∩ set(val_patients) ∩ set(test_patients) == ∅`

**为什么这样划分**（教学点）：
- 如果按图片随机划分，同一病人的 3 张图会被拆散 → 模型通过"病人外观记忆"作弊 → 测试集精度虚高 10-20%
- 这种现象叫 **Patient Leakage**，是医学 AI 常见陷阱
- Notebook 里会用一段代码对比两种划分方式的精度差异，让学员亲眼看到

### 5.3 预处理

**应用于所有子集**：

```
Resize(256, 256)
  ↓
CenterCrop(224, 224)     # ResNet-50 标准输入尺寸
  ↓
ToTensor()               # PIL → [0,1] Tensor
  ↓
Normalize(
    mean=[0.485, 0.456, 0.406],   # ImageNet 均值
    std=[0.229, 0.224, 0.225]     # ImageNet 标准差
)
```

**理由**：用 ImageNet 预训练权重必须用对应的归一化参数。

### 5.4 数据增强（仅训练集）

| 增强操作 | 是否启用 | 理由 |
|---------|---------|------|
| 随机小角度旋转（±15°） | ✅ | 内镜操作自然抖动 |
| 色彩抖动（brightness/contrast/saturation ±0.2） | ✅ | 不同设备、不同光照 |
| 随机裁剪缩放（scale 0.8-1.0） | ✅ | 不同景深 |
| 水平翻转（HorizontalFlip） | ❌ **严禁** | yz↔zz 标签冲突 |
| 垂直翻转（VerticalFlip） | ❌ | 内窥镜不会倒置 |

### 5.5 Dataset 与 DataLoader

- **自定义类**：`BronchoscopyDataset(torch.utils.data.Dataset)` 从 CSV 读取，`__getitem__` 返回 `(tensor, label_id)`
- **DataLoader 配置**：
  - `batch_size = 32`
  - `num_workers = 4`
  - `shuffle = True`（仅训练集）
  - `pin_memory = True`（A100 GPU 加速）

---

## 6. 模型设计

### 6.1 架构

**基础模型**：ImageNet 预训练 ResNet-50

**改造**：替换最后一层分类头

```
输入 [B, 3, 224, 224]
  ↓ ResNet-50 主干（torchvision.models.resnet50(weights="IMAGENET1K_V2")）
  ↓ 全局平均池化 → [B, 2048]
  ↓ Dropout(p=0.3)                    ← 新增，防过拟合
  ↓ Linear(2048 → 3)                  ← 替换原 Linear(2048 → 1000)
输出 logits [B, 3]
  ↓ softmax
概率分布 {lt, yz, zz}
```

### 6.2 选型理由

- **ResNet-50 > ResNet-18**：641 张图足够训练 2300 万参数，更大的模型能学到更精细的气管纹理
- **ResNet-50 < ResNet-101/152**：更大的模型在 641 张图上易过拟合
- **ResNet-50 vs EfficientNet-B3**：两者精度相近，但 ResNet-50 更经典、更好讲，适合学习
- **Dropout(0.3)**：小数据集标配，降低过拟合风险

---

## 7. 训练策略（两阶段 Fine-tuning）

### 7.1 阶段 1：冻结主干，训练分类头

- **训练参数**：仅 `fc` 层（新分类头）
- **Epochs**: 5
- **学习率**: 1e-3
- **Optimizer**: AdamW (`weight_decay=1e-4`)
- **目的**: 新分类头初始化为随机权重，直接 fine-tune 整网会把梯度反传到预训练主干，破坏已有特征。先冻结主干让分类头收敛到合理位置

### 7.2 阶段 2：解冻全部，低学习率 Fine-tune

- **训练参数**：全网络
- **Epochs**: 最多 20（带 Early Stopping）
- **学习率**: 1e-4（比阶段 1 小 10 倍）
- **Optimizer**: AdamW (`weight_decay=1e-4`)
- **Scheduler**: CosineAnnealingLR (`T_max=20`)
- **Early Stopping**: 验证集 Accuracy 5 轮无提升则停，保存 `val_acc` 最佳权重到 `checkpoints/best_model.pt`

### 7.3 公共训练配置

- **Loss**: `CrossEntropyLoss`
- **Batch Size**: 32
- **设备**: CUDA（A100）
- **混合精度**: AMP (`torch.cuda.amp`) 开启（A100 上加速明显）

### 7.4 可复现性

- 固定 `seed = 42`
- 设置 `torch.manual_seed`, `numpy.random.seed`, `random.seed`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

---

## 8. 评估方案

### 8.1 训练过程监控

- 每 epoch 记录：train_loss, train_acc, val_loss, val_acc
- 训练结束后绘制两张曲线图：
  - Loss 曲线（train vs val）
  - Accuracy 曲线（train vs val）
- 保存至 `outputs/training_curves.png`
- 用途：诊断过拟合（val 曲线上扬）、欠拟合（两条都低）

### 8.2 测试集评估指标

**基于最佳 checkpoint，在测试集上一次性计算**：

1. **整体 Accuracy**
2. **Confusion Matrix**（3×3 热力图）
   - 保存至 `outputs/confusion_matrix.png`
3. **Per-class Classification Report**（`sklearn.metrics.classification_report`）
   - Precision / Recall / F1-score per class
   - Macro / Weighted averages
4. **置信度分布**
   - 箱线图：对每个真实类别，绘制模型对该类的 softmax 置信度分布
   - 可看出模型是否过度自信或犹豫
5. **错误案例分析**
   - 挑出被分错的图片
   - 每张打印：真实标签、预测标签、置信度、Grad-CAM 热力图
   - 放到 notebook 最后一个章节讨论

---

## 9. 可解释性（Grad-CAM）

### 9.1 原理说明（notebook 教学章节）

Grad-CAM 通过最后一个卷积层的梯度加权特征图，生成与输入图同尺寸的热力图，显示"模型对该预测结果最依赖图片的哪些区域"。

### 9.2 实现

- **库**: `pytorch-grad-cam`
- **目标层**: `model.layer4[-1]`（ResNet-50 最后一个残差块）
- **采样策略**：
  - 从测试集每类挑选 3 张正确分类高置信度样本
  - 从测试集挑选 3 张被分错的样本
  - 共 12 张样本，各画热力图叠加到原图上
- **输出**: `outputs/gradcam_examples/` 下 12 张 PNG

### 9.3 预期观察（教学引导）

- ✅ 正确判断时，热力图应聚焦在**解剖关键区域**（隆突的 Y 形分叉、支气管开口中心）
- ⚠️ 如果热力图指向**镜头反光、黑边、背景血丝**等非解剖因素 → 说明模型学到了数据伪特征，需要反思数据质量

---

## 10. Gradio Web Demo

### 10.1 功能

- 拖拽或点击上传一张气管镜图片
- 实时显示：
  - 预测类别（隆突 / 右总支气管 / 左总支气管）
  - 三类的 softmax 置信度（条形图）
  - Grad-CAM 热力图覆盖原图

### 10.2 实现

- 独立 notebook `notebooks/demo.ipynb`
- 加载 `checkpoints/best_model.pt`
- 用 `gradio.Interface` 构建，~30 行代码
- 启动后访问 `http://localhost:7860`
- 支持 `share=True` 参数生成临时公网链接（科室演示）

---

## 11. Notebook 结构（教学导向）

`notebooks/bronchoscopy_classifier.ipynb` 章节规划（⭐ = 重点教学章节）：

1. 项目介绍与目标
2. 环境准备（依赖导入、数据路径、随机种子）
3. 数据探索（文件名模式、每类样例图、类别分布柱状图）
4. 数据清洗 & Manifest 生成
5. ⭐ **按病人划分数据集**（讲 Patient Leakage，对比错误做法）
6. 数据增强策略（讲为什么禁止水平翻转）
7. Dataset & DataLoader
8. ⭐ **PyTorch 基础回顾**（Tensor / nn.Module / 训练循环，针对 D 级别学员）
9. 模型搭建（ResNet-50 + 分类头替换）
10. ⭐ **两阶段 Fine-tuning**（讲迁移学习原理）
11. 训练过程可视化（Loss / Accuracy 曲线）
12. 测试集评估（Confusion Matrix + 分类报告 + 置信度分布）
13. ⭐ **Grad-CAM 可解释性分析**
14. 错误案例分析（挑错分样本讨论）
15. 总结与下一步（后续可扩展方向：多模型对比、更强的增强策略、ViT 等）

---

## 12. 交付清单

| 文件 | 说明 |
|------|------|
| `notebooks/bronchoscopy_classifier.ipynb` | 主教学 notebook（15 章节） |
| `notebooks/demo.ipynb` | Gradio Web demo |
| `src/*.py` | 模块化 Python 源码 |
| `data_splits/{manifest,train,val,test}.csv` | 数据划分记录（可复现） |
| `checkpoints/best_model.pt` | 最佳模型权重（~100MB） |
| `outputs/confusion_matrix.png` | 测试集混淆矩阵 |
| `outputs/training_curves.png` | 训练过程曲线 |
| `outputs/gradcam_examples/*.png` | Grad-CAM 示例 12 张 |
| `requirements.txt` | Python 依赖清单 |
| `README.md` | 项目说明与运行指引（中文） |
| `.gitignore` | git 忽略规则 |

---

## 13. 工作量与时间估算

| 阶段 | 在 A100 上耗时 |
|------|----------------|
| 数据清洗 + 划分 | <1 min |
| 阶段 1 训练（5 epochs，冻结主干） | ~1 min |
| 阶段 2 训练（最多 20 epochs，全网络） | ~10 min |
| 评估 + Grad-CAM | ~2 min |
| **总计**（一次完整流程） | **~15 min** |

用户学习时间预估：
- 环境配置 + 跑通一遍：1-2 小时
- 完整吃透 notebook 所有教学点：4-8 小时

---

## 14. 风险与应对

| 风险 | 概率 | 应对方案 |
|------|------|----------|
| 测试精度 < 85%（显著低于预期） | 低 | 检查数据质量、提高数据增强强度、延长训练 |
| 过拟合严重（val 曲线掉头向下） | 中 | 增加 Dropout、更强数据增强、减少 epoch |
| `yz` 和 `zz` 混淆率高 | 中 | 正常现象（两者视觉相似），用 Grad-CAM 分析错误案例 |
| 文件名解析失败 | 低 | 打印异常样本由用户手动处理 |
| A100 服务器权限或数据传输问题 | 低 | 代码支持单机 MPS/CPU 回退 |

---

## 15. 后续扩展方向（Out of Scope，但为将来做铺垫）

- **方案 2**：多模型架构对比实验（ResNet-50 vs EfficientNet-B3 vs ViT-Small）
- **更细粒度分类**：识别右中间段、左上下肺开口等更深层解剖结构
- **数据扩充**：收集更多病人数据、考虑 GAN 生成合成样本
- **生产部署**：FastAPI 服务、Docker 容器、接入 PACS / 报告系统
- **可解释性升级**：SHAP、LIME、Attention Rollout（for ViT）

---

## 16. 下一步

1. 本设计文档由用户审查
2. 审查通过后，使用 `writing-plans` skill 生成详细的**实施计划**（Implementation Plan），拆解为可执行的任务清单
3. 按计划实施
