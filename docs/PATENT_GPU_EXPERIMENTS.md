# 专利补充实验：GPU 一键运行说明

目前一次固定测试集的结果仅显示：基线准确率 95.96%，完整方案 97.98%，绝对差 2.02 个百分点；患者级配对 Bootstrap 95% CI 跨过 0。因此不建议把专利优势只写成“干净测试集准确率显著提高”。本流程优先验证更符合方法设计目的的证据：旋转、反光与遮挡下的稳健性，以及跨训练随机种子的稳定性。

## 1. 推荐先跑核心版

核心版只跑 `baseline` 和 `full`，使用 3 个训练随机种子，但所有运行固定同一个患者级划分（`split_seed=42`）：

```bash
cd /path/to/AirwayRecognition
bash scripts/run_patent_gpu_experiments.sh
```

脚本支持断点续跑。SSH 中断后再次执行同一命令，已经完成且结果文件完整的步骤会自动跳过。建议在 `tmux` 或 `screen` 中执行。

若核心版显示完整方案在多数随机种子、尤其在中高强度旋转/反光/遮挡下均有一致优势，再跑五组消融：

```bash
MODE=full bash scripts/run_patent_gpu_experiments.sh
```

五组分别是：

1. `baseline`：ResNet-50；
2. `crop_only`：仅有效视野裁剪；
3. `attention`：仅解剖注意力；
4. `regularized`：注意力 + 旋转等变/伪特征抑制正则；
5. `full`：有效视野裁剪 + 注意力 + 双正则。

## 2. 常用参数

参数通过环境变量覆盖。例如：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHON_BIN=/path/to/conda/env/bin/python \
DATASET_DIR=/path/to/dataset \
BATCH_SIZE=32 \
NUM_WORKERS=8 \
BOOTSTRAP=10000 \
bash scripts/run_patent_gpu_experiments.sh
```

快速冒烟测试可用：

```bash
SEEDS="42" STAGE1_EPOCHS=1 STAGE2_EPOCHS=1 ROBUST_SEEDS=2 BOOTSTRAP=200 \
RUN_AUDIT=0 RUN_ROOT="$PWD/patent_runs_smoke" \
bash scripts/run_patent_gpu_experiments.sh
```

主要环境变量：

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `MODE` | `core` | `core` 为基线/完整方案；`full` 为五组消融 |
| `SEEDS` | `42 2026 3407` | 训练随机种子，空格分隔 |
| `SPLIT_SEED` | `42` | 患者级数据划分种子；正式对比不要改变 |
| `DEVICE` | `cuda` | PyTorch 设备 |
| `RUN_ROOT` | `patent_runs` | 所有实验输出根目录 |
| `ROBUST_SEEDS` | `10` | 每个反光/遮挡强度的随机位置重复数 |
| `ROBUST_BASE_SEED` | `20260714` | 所有模型共用的扰动位置起始种子；正式比较不要改变 |
| `BOOTSTRAP` | `5000` | 患者级 Bootstrap 次数；正式材料可设为 10000 |
| `RUN_AUDIT` | `1` | 是否对 baseline/full 生成对齐后的 Grad-CAM 审计 |

## 3. 输出与判读

运行完成后先看：

- `patent_runs/summary/SUMMARY.md`：所有随机种子与方案的汇总；
- `patent_runs/summary/figures/clean_performance_multiseed.pdf`：干净测试集多种子性能；
- `patent_runs/summary/figures/robustness_baseline_vs_full.pdf`：旋转、反光、遮挡稳健性曲线；
- `patent_runs/summary/run_config_audit.csv`：训练种子、代码版本及三份患者划分指纹核验；
- `patent_runs/seed_*/paired_metric_difference_full_vs_baseline.csv`：准确率和宏 F1 的患者级配对差值及 95% CI；
- `patent_runs/seed_*/auc_paired_difference_full_vs_baseline.csv`：各类别及宏/微平均 AUC 的患者级配对差值。

更适合专利材料的结论是：“在干净数据性能不降低的前提下，完整方案在方向变化和伪特征干扰下表现出更稳定的识别性能。”只有在多随机种子和稳健性曲线一致支持时才应采用该表述。如果稳健性也没有稳定改善，应缩小方法主张或调整算法，而不宜依靠单次 2.02 个百分点的结果作强结论。

Grad-CAM 自动掩膜重叠仅作为探索性质量控制；正式材料中的“关注解剖区域”仍建议由两名以上专家盲法评分，并报告一致性。
