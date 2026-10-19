# 逐图反光配对评估：服务器运行说明

本脚本加载已经训练好的普通 ResNet-50 基线和完整注意力方案，对同一张原图生成相同的合成高光，再分别推理。只做评估，不重新训练、不下载预训练权重、不修改原始图片。

## 1. 运行前准备

在服务器项目根目录运行。需要项目已有依赖、两个模型的 `best_model.pt`、同一患者级划分的 `test.csv` 及对应图片。

同步代码时，至少同时更新 `scripts/evaluate_glare_pairs.py` 和 `src/data.py`；新脚本需要后者提供的固定裁剪框接口。测试文件为 `tests/test_glare_pairs.py`。依赖沿用项目的 `requirements.txt`，包括 `numpy<2`，无需新增包。

默认模型配置：

- 基线：`src.models.build_resnet50`，不裁黑边。
- 完整方案：`src.attention.build_attention_resnet50`，裁黑边，普通分辨率注意力。
- 权重采用本项目保存的纯 `state_dict`，严格检查参数是否匹配。
- 如果完整方案训练时用了高分辨率注意力，必须追加 `--full-attn-hires`；此配置无法仅靠权重键名推断。
- 如果完整方案训练时没有裁黑边，追加 `--no-full-crop-border`。只有基线训练时也裁黑边，才追加 `--baseline-crop-border`。
- `--image-root dataset` 将 CSV 中的图片路径按文件名映射到服务器的平铺数据目录，适用于本项目的数据组织方式。无需修改 CSV 中原有的本机绝对路径；映射后重复图片会被拒绝。

## 2. 一组已训练模型的完整评估

以下路径与 `scripts/run_patent_gpu_experiments.sh` 的默认产物结构一致。若权重保存在其他目录，替换两个 checkpoint 参数。

```bash
python -m scripts.evaluate_glare_pairs \
  --baseline-checkpoint patent_runs/seed_42/baseline/best_model.pt \
  --full-checkpoint patent_runs/seed_42/full/best_model.pt \
  --splits-dir data_splits_patent \
  --split test \
  --image-root dataset \
  --output-dir patent_runs/seed_42/glare_pairs_v1 \
  --device cuda \
  --batch-size 32 \
  --coverages 0 0.05 0.10 0.20 \
  --placements 10 \
  --seed 20261019 \
  --bootstrap 10000
```

输出目录必须不存在或为空，避免将新旧结果混在一起。失败重跑时选一个新的输出目录。显存不足可减小 `--batch-size`；仅有 CPU 时改用 `--device cpu`。

先检查程序、路径和权重能否正常运行，可改用 `--coverages 0 0.05 --placements 1 --bootstrap 0`，并指定单独的临时输出目录。这仍会评估该划分中的全部图像，但推理次数较少；不要将这次快速检查混入正式结果。

## 3. 三个训练种子分别评估

以下示例保持测试集和反光扰动相同，只更换已训练权重：

```bash
for training_seed in 42 2026 3407; do
  python -m scripts.evaluate_glare_pairs \
    --baseline-checkpoint "patent_runs/seed_${training_seed}/baseline/best_model.pt" \
    --full-checkpoint "patent_runs/seed_${training_seed}/full/best_model.pt" \
    --splits-dir data_splits_patent \
    --image-root dataset \
    --output-dir "patent_runs/seed_${training_seed}/glare_pairs_v1" \
    --device cuda --batch-size 32 \
    --coverages 0 0.05 0.10 0.20 \
    --placements 10 --seed 20261019 --bootstrap 10000 || break
done
```

三个训练种子仍然使用同一批患者。不要把三次结果合并后称为三倍数量的独立患者。

## 4. 输出与指标定义

| 文件 | 内容 |
| --- | --- |
| `per_image_predictions.csv` | 原图文件名、每张图/模型/扰动强度/位置重复的干净与扰动预测、三类概率及变化、正确性、实际掩膜覆盖率 |
| `summary.csv` | 每个模型、每个强度的准确率、翻转率、正确变错误率和平均概率变化 |
| `paired_comparison.csv` | 完整方案减基线的配对指标差，以及患者级配对 Bootstrap 区间 |
| `summary.md` | 便于阅读的结果与统计口径 |
| `metadata.json` | 参数、模型权重和划分哈希、类别顺序、软件版本、扰动定义及限制；完成后才写入 |

CSV 中的率用 0–1 表示；乘 100 即百分比。概率差也是 0–1 概率单位，0.05 等于 5 个百分点。

- **预测翻转率**：扰动后预测类别不同于干净图预测类别的比例。翻转可能是正确变错误，也可能是错误变正确，不能单独视作性能变差。
- **正确变错误比例**：原本预测正确的图像—位置观测中，扰动后预测错误的比例。不是除以所有图片；分母为零时输出空值，不写成 0%。另保留以全部观测为分母的比例。
- **共同正确子集对照**：只在基线和完整方案对干净图都预测正确的病例上，比较各自被干扰后变错的比例，避免两个模型的原始正确病例不同。
- **真实类别概率变化**：扰动图的真实类别概率减去干净图概率。负数表示模型对正确类别的支持下降。另提供相反符号的“概率下降量”，并保留各类别的原始概率及差值。
- **配对差值**：固定采用 `full - baseline`。对翻转率、正确变错误率、概率下降量，负差通常更有利于完整方案；对准确率，正差更有利。必须结合完整指标解释。

非零强度会重复多个反光位置，汇总率按图像—位置观测计算；各图重复数相同。零强度只计算一次。输出同时报告独立图像数、患者数和观测数，不把位置重复当作新患者。Bootstrap 按患者重采样，整组保留该患者的图像和位置重复，两个模型使用相同重采样。`--bootstrap 0` 关闭区间估计。

## 5. 配对与几何处理

1. 每张原图的有效视野与裁剪框从干净原图计算一次。
2. 在原始图像坐标上生成同一份合成高光，两个模型共享该扰动版本。
3. 各模型分别按其训练配置预处理。干净和扰动图共用固定裁剪框，防止亮斑改变黑边检测结果。
4. 同时记录原图有效视野中的实际覆盖率，以及各模型最终输入中的覆盖率。即使原始扰动相同，两模型保留的视野不同，最终可见扰动量也可能不同。
5. 扰动种子由稳定的病例标识和位置重复编号生成，改变 CSV 顺序或 batch size 不会改变反光掩膜。不同覆盖档位复用同一随机序列。

## 6. 解释范围

本次采用确定性近白方块，属于合成亮斑/遮挡压力测试，不是经过物理验证的真实反光模拟。覆盖档位是目标强度，方块重叠、有效视野限制等会使实际覆盖率不同，应使用输出的实际值检查。

名义覆盖率按原图面积换算方块数量；`actual_coverage` 是实际掩膜占原图的比例，`actual_fov_coverage` 是占有效视野的比例，`input_coverage` 是各模型最终输入中的比例。三种分母不同，不宜混用。

稳定的错误预测仍然是错误预测。判断抗干扰能力时，应同时查看干净准确率、扰动后准确率、共同正确病例变错率及概率变化。强遮挡可能破坏解剖可见性，结果不能直接外推为临床安全性或专利优势。

本脚本比较完整推理方案。若需要单独归因于注意力、裁剪或某项正则，还需单独设计组件消融。对新模型或超参数的选择使用训练/验证数据，留出测试集用于最终评价。

逐图文件保留本地患者编号用于配对和统计；对外分享优先使用汇总文件。
