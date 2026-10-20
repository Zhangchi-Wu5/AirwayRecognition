# 旋转约束对照：服务器运行说明

本实验回答：在同一注意力模型、裁边、高光约束和普通旋转增强下，加入旋转等变损失（EQ）是否有增量效果。当前仅完成代码和本地流程验证，尚无正式训练结果。

## 1. 首轮只运行种子 42

将本轮代码同步到服务器后，在 `/opt/data/private/AirwayRecognition` 下激活原来的 `airway` 环境，运行：

```bash
python -m scripts.run_rotation_ablation \
  --splits-dir data_splits_patent \
  --image-root dataset \
  --output-root rotation_ablation_v1 \
  --seeds 42 \
  --device cuda \
  --batch-size 32 \
  --num-workers 4
```

沿用已有数据划分，**无需重新生成划分或处理图片**。高光掩膜仍由训练代码生成。默认使用 ImageNet 预训练参数；若服务器尚无缓存，首次会下载。正式实验不要添加本地调试用的 `--no-pretrained`。

两组分别为：

| 目录 | EQ 权重 | 其他条件 |
|---|---:|---|
| `eq_off` | 0 | 仍执行额外旋转前向，匹配 BN 更新次数和随机数消耗 |
| `eq_on` | 0.1 | 与上组相同，启用 EQ 损失梯度 |

两组均裁边、使用原有注意力结构、PF=0.1、普通 ±15° 旋转增强，分别训练固定的 5+20 轮。会严格核对初始化和每轮实际增强输入是否相同；核验失败会停止，不能把失败的一组当作有效配对结果。

训练后自动在**验证集**评价 0°、±5°、±10°、±15°，按患者做 10,000 次配对 bootstrap。读取测试集成员信息仅用于检查划分隔离，测试图像不加载。此轮不是重新比较原 ResNet 与完整方案，也不涉及视频。

输出目录必须不存在或为空。脚本不支持断点续训，也不覆盖已有结果。若显存不足，改用新的输出目录、减小批大小后，将两组一起重新运行；不要仅修改其中一组。

## 2. 输出在哪

以上命令的结果保存在：

```text
/opt/data/private/AirwayRecognition/rotation_ablation_v1/
├── protocol.json
└── seed_42/
    ├── pair_metadata.json
    ├── eq_off/
    │   ├── run_config.json
    │   ├── training_history.csv
    │   ├── training_input_trace.csv
    │   ├── last_model.pt
    │   └── best_val.pt
    ├── eq_on/                  # 同上
    └── rotation_val/
        ├── summary.md
        ├── summary.csv
        ├── paired_comparison.csv
        ├── per_image_predictions.csv
        └── metadata.json
```

主分析使用固定轮次结束的 `last_model.pt`；`best_val.pt` 留作追溯，不看结果后改用另一份权重。

优先查看 `rotation_val/summary.md` 和 `paired_comparison.csv`：

- `scope=nonzero_mean`：预先指定的六个非零角度平均，是主要比较；逐角度结果为补充。
- 差值方向均为 **eq_on − eq_off**：准确率差为正较好，共同原图正确队列的变错率差为负较好。
- 同时看原图准确率、分类别召回率、预测翻转率和真实类别概率变化。低翻转率也可能意味着持续预测错误。
- 注意力误差变小须结合原始均值、空间标准差和熵判断；均匀图、全低值图也可能得到很小的误差。

只有注意力误差下降，不能写成分类更稳或关注部位更正确。若旋转表现改善但原图表现下降，应如实报告取舍。一个种子的结果是探索信号，不能据此确认稳定技术效果。

## 3. 打包结果发回

在项目根目录执行以下命令。无需打包大体积权重：

```bash
tar -czf rotation_ablation_seed42_results.tar.gz \
  rotation_ablation_v1/protocol.json \
  rotation_ablation_v1/seed_42/pair_metadata.json \
  rotation_ablation_v1/seed_42/eq_off/run_config.json \
  rotation_ablation_v1/seed_42/eq_off/training_history.csv \
  rotation_ablation_v1/seed_42/eq_off/training_input_trace.csv \
  rotation_ablation_v1/seed_42/eq_on/run_config.json \
  rotation_ablation_v1/seed_42/eq_on/training_history.csv \
  rotation_ablation_v1/seed_42/eq_on/training_input_trace.csv \
  rotation_ablation_v1/seed_42/rotation_val
```

压缩包位于 `/opt/data/private/AirwayRecognition/rotation_ablation_seed42_results.tar.gz`。

## 4. 两组已训练完成，仅评价中断时

保留权重和训练记录，使用新的评价输出目录单独运行：

```bash
python -m scripts.evaluate_rotation_pairs \
  --eq-off-checkpoint rotation_ablation_v1/seed_42/eq_off/last_model.pt \
  --eq-on-checkpoint rotation_ablation_v1/seed_42/eq_on/last_model.pt \
  --splits-dir data_splits_patent \
  --image-root dataset \
  --split val \
  --output-dir rotation_ablation_v1/seed_42/rotation_val_retry \
  --device cuda \
  --batch-size 32 \
  --angles -15 -10 -5 0 5 10 15 \
  --bootstrap 10000 \
  --seed 20261020
```

若训练尚未完成，应换新训练输出目录重新开始。评价重跑后，打包命令相应改为 `rotation_val_retry`。

## 5. 后续重复验证

若决定继续评价这套固定设计，在不改训练设置的前提下运行另两个预定种子：

```bash
python -m scripts.run_rotation_ablation \
  --splits-dir data_splits_patent \
  --image-root dataset \
  --output-root rotation_ablation_v1_more_seeds \
  --seeds 2026 3407 \
  --device cuda \
  --batch-size 32 \
  --num-workers 4
```

保留所有已跑种子的正负结果，不替换不利种子。三个种子仍是同一批验证患者，不是三项独立患者验证。正式确认新技术效果，还需要未参与方案选择的患者数据。

实验控制及指标定义见 [设计记录](ROTATION_ABLATION_DESIGN_20261020.md)；其他静态图片候选方向见 [候选方向与已有方法](AIRWAY_CANDIDATE_DIRECTIONS_20261020.md)。

## 本地验证记录（2026-10-20）

- 训练对照及原训练测试 16 项、相邻 attention 测试 13 项、旋转评价测试 24 项通过；独立代码审阅未发现阻断问题。
- 使用合成图片和未预训练的真实 ResNet50，在 CPU 完成两组各 1+1 轮的两阶段训练；初始化及逐轮输入流一致，随后成功输出 3 张验证图 × 7 个角度 × 2 个模型的 42 行预测。测试图片文件刻意不存在，流程仍正常完成。
- 本地 macOS 沙箱对 OpenMP 共享内存有限制，4-worker 测试及真实模型评价经自动审批在沙箱外验证通过；评价由相同参数单独重跑成功。
- 以上检查验证实现和接口，不能作为分类效果证据。尚未运行服务器 GPU 正式训练。
