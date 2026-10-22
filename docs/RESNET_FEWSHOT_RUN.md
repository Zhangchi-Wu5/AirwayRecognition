# ResNet 小样本筛查：运行与判读

本实验回答：同一个冻结 ImageNet ResNet-50，在每类仅使用 1、5、10、20 张训练图片时，性能和训练患者抽样波动如何变化？每个预算抽样 20 次，与全部 446 张训练图片拟合一次的结果比较。

这是一项固定三分类的低标注迁移筛查，不是对未见类别进行元学习。全量对照也冻结主干，不能直接与历史约 98% 的整体微调成绩混成一条曲线。

## 运行

从项目根目录运行。依赖为项目 `requirements.txt` 中的 PyTorch、torchvision、NumPy、scikit-learn、Pillow、Matplotlib，以及 `threadpoolctl>=3.5,<4`。建议使用独立虚拟环境；不要直接更改已有科研环境。

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m scripts.run_resnet_fewshot \
  --splits-dir data_splits \
  --image-root dataset \
  --output-root outputs/resnet_fewshot_screening_new \
  --device cpu --batch-size 16 --num-workers 0 \
  --shots 1 5 10 20 --repeats 20 --sampling-seed 20261022
```

首次运行会下载 torchvision 官方 ImageNet1K_V2 权重。可通过 `TORCH_HOME` 指定模型缓存。脚本不接受本地微调检查点，不需要服务器或 GPU。所有图像处理均在本地进行。

输出目录必须为空或不存在；脚本拒绝覆盖已有结果。`--image-root dataset` 按文件名将 CSV 中的原路径映射到本机图片目录。复制至服务器时，修改该参数即可。

只检查数据和生成支持集计划时，增加 `--prepare-only`，并使用**另一个输出目录**。该模式不会计算特征或生成性能结论，也不支持随后在同一目录续跑。

## 固定协议

- 保留 `data_splits` 的训练／验证／测试患者划分。只读取训练和验证图像，测试 CSV 仅用于身份隔离检查。
- 每个重复和类别独立排列训练患者，每位患者该类别至多选一图。1、5、10、20 张取同一排列的前缀；类别间允许患者重合。
- 全量参考使用所有训练图像，包含仅有部分部位的患者。
- ResNet-50 完全冻结，保持 `eval()` 和推理模式。检查提取前后模型参数与 BatchNorm 缓冲指纹一致。
- 短边缩放 256、中心裁剪 224、ImageNet 归一化；不随机增强、不去黑边。
- 2048 维特征逐图 L2 归一化。固定 L2 逻辑回归（C=1、LBFGS、最多 5000 次、容差 1e-8），未按验证结果调参。
- 全部 81 个线性头复用一次提取的训练／验证特征。只有各自支持集的标签进入该次拟合；收敛失败则记录失败，不静默接受。

## 输出

| 文件 | 内容 |
|---|---|
| `REPORT.md` | 准确率均值、抽样标准差、范围及解释边界 |
| `learning_curve.png` / `.pdf` | 所有抽样点、均值±SD、同协议全量参考 |
| `summary_metrics.csv` | 准确率、宏 F1、平衡准确率、AUC、损失与类别召回汇总 |
| `per_run_metrics.csv` | 81 次拟合的指标、收敛信息、耗时、图像及患者数 |
| `predictions.csv` | 每次拟合在固定验证集上的逐图概率 |
| `support_plan.csv`、`supports/` | 全部支持集成员及抽样数量 |
| `features/`、`heads/` | 特征缓存与各个线性头，可复核预测 |
| `protocol.json` | 状态、数据与代码哈希、预训练权重指纹、依赖、运行及产物信息 |

支持集和逐图结果含项目内部病例标识，分享汇总图表时无需附带这些明细。

## 结果判断

比较对象首先是本轮冻结特征的全量结果。低预算低于该结果，并随标注增加改善，才说明这一固定方法对标注量敏感；仍需合理微调等基线才能决定是否提出新方法。

若低预算接近同法全量、但两者均低于历史整体微调成绩，应优先调查适配方法。若低预算和全量均很高，优先扩充独立患者、设备或实际困难视图。

20 次标准差表示选择训练患者引起的波动，并不是新患者泛化的置信区间。每次都使用同一批 32 位验证患者，不能将重复次数视为新增病例。固定 C 也不代表已获得最优小样本分类器。

## 本轮可复现记录

2026-10-22 本地运行使用临时虚拟环境 `/private/tmp/airway-fewshot-runtime-20261022`，保留原环境。关键版本：Python 3.11.5、torch 2.3.1、torchvision 0.18.1、NumPy 1.26.4、SciPy 1.17.1、scikit-learn 1.3.0、Matplotlib 3.7.2、threadpoolctl 3.7.0。具体环境及代码指纹以结果中的 `protocol.json` 为准。

测试入口：`python -m pytest tests/test_fewshot.py -q`。测试覆盖嵌套患者抽样、重复身份及内容、测试图像隔离、冻结和 BatchNorm 状态、线性头收敛与端到端指标输出。
