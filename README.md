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
