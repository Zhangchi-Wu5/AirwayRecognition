# 候选区域—周围组织—区域间关系：首轮实验

本实验只使用已有静态图片、患者划分和三分类标签。目标是验证：在相同训练条件下，候选区域内外特征以及区域间关系，是否给单图 ResNet 提供额外有效信息。候选是规则产生的视觉区域，不是人工确认的管腔、解剖 ROI 或分割真值；本版本没有证明方法有效或具备专利新颖性。

## 方法和四组对照

保留普通 ResNet-50 的全局分类分支；另外在较高空间分辨率的特征图上提取候选内部和外围组织带，再计算不同候选之间的特征及几何关系。全部模型仅使用分类交叉熵，不加原先的 PF、EQ 或注意力损失。

候选生成不需要新标注：排除边缘相连的近黑背景，在多个灰度阈值下寻找稳定暗区，再按面积、内外亮度差和跨阈值重合程度筛选，最多保留三个。每个候选的外围环带排除其他候选内部。分类分支读取“内部、外围、内部减外围”的特征，关系分支再读取两区域的特征差、中心位置差、面积与宽高差、中心距离，以及内外亮度对比差。

|目录名|行为|回答的问题|
|---|---|---|
|`baseline`|普通全局 ResNet 分类，新增分支不参与预测|同条件基线|
|`regions`|全局分类加候选内部、外围组织特征|局部信息是否有用|
|`relations`|再加入区域之间的关系特征|关系是否提供额外价值|
|`no_geometry_relations`|保留相同区域特征和关系网络，训练时起将几何输入置零|位置、大小等几何信息是否有额外贡献|

`no_geometry_relations` 与完整关系组的网络尺寸相同，但从训练开始就将中心位置、面积、宽高差及中心距离置零，保留内外亮度对比差。区域内部、周围组织及二者差异的特征仍保留，因此该组检验的是显式几何输入的增量；区域特征自身仍可能携带位置信息。应同时查看候选数量和具有至少两个候选的图片比例。

找不到可靠候选时，模型使用全局分支；只有一个候选时，不产生候选间关系贡献。辅助分类输出层从零开始，四组初始分类输出一致。所有组采用同样的初始化、训练图片顺序、实际增强输入、分类损失和训练轮数，并校验输入流指纹。

首轮沿用普通基线的预处理：训练随机裁剪、普通 ±15° 旋转和颜色增强；验证 Resize + CenterCrop。候选从模型实际输入的增强后图片提取，与特征坐标一致。**本版本不是完整视野保留实验**；中心裁剪的限制同样存在于四组。

为避免此前 BN 统计量问题再次混入，本实验全部组固定 BatchNorm 的运行均值和方差；第二阶段仍可学习 BN 的缩放、平移参数。因此这里的 baseline 是本轮匹配重训的普通 ResNet，不能与历史 checkpoint 的数值直接拼接比较。

## 服务器运行

先将本轮代码同步到服务器，再在项目环境执行。首轮只运行种子 42，不需要把旧的 PF/EQ 模型加入。

```bash
cd /opt/data/private/AirwayRecognition
conda activate airway

python -u -m scripts.preview_region_candidates \
  --splits-dir data_splits_patent \
  --image-root dataset \
  --output-dir region_candidate_preview_v1 \
  --split train \
  --per-class 4

python -u -m scripts.run_region_relation_ablation \
  --splits-dir data_splits_patent \
  --image-root dataset \
  --output-root region_relation_ablation_v1 \
  --seeds 42 \
  --device cuda \
  --batch-size 16 \
  --num-workers 4 \
  --stage1-epochs 5 \
  --stage2-epochs 20 \
  --bootstrap 10000
```

预览先用于检查候选算法是否明显失效，不能以“看起来像管腔”代替定位准确性的验证。若训练集上大量样本选中黑边、整片阴影，或几乎没有两候选图片，应先修正候选生成假设，不能等看到验证准确率后再逐张调规则。

预览图为 `region_candidate_preview_v1/candidate_contact_sheet.png`；同目录 `candidate_summary.csv` 包含全部训练图片的候选计数，`preview_config.json` 记录规则参数与来源。

训练固定 5+20 轮，不使用提前停止。主评价使用最后一轮 `last_model.pt`；验证最优权重仅作辅助保存，不参与本轮主结果选择。脚本自动完成验证集评价。不会对测试集图片进行推理；会读取测试集 CSV 身份以检查患者划分是否交叉。

输出目录必须为空或不存在；已有结果不会被覆盖。失败后应保留日志，在新目录重试。本轮没有实现从中断位置续训。

若四组训练已完成，只有自动汇总失败，可单独重跑评价，无需重训；使用尚未存在的评价输出目录：

```bash
python -u -m scripts.evaluate_region_relations \
  --run-dir region_relation_ablation_v1/seed_42 \
  --splits-dir data_splits_patent \
  --image-root dataset \
  --output-dir region_relation_ablation_v1/seed_42/region_val_retry \
  --device cuda --batch-size 16 --bootstrap 10000
```

评价会核对训练时的源码、配置、划分和权重指纹。训练后如改动相关源码，应保留原版本用于复核，不要绕过校验混用结果。

## 查看结果

优先查看 `region_relation_ablation_v1/seed_42/region_val/summary.md`。同目录的 `per_image_predictions.csv` 保存逐图预测及三类概率，`summary.csv` 保存各组指标，`paired_comparison.csv` 保存配对差值与区间。结果包括分类准确率、宏 F1、宏 OVR AUC、左右主支气管互相误判情况，以及候选数量。配对比较以患者为单位重采样，不能把同一患者的多张图当作独立患者。

判断顺序：

1. `regions` 相对 `baseline`：局部候选和周围组织是否真的有贡献。
2. `relations` 相对 `regions`：新增关系是否有额外贡献。
3. `relations` 相对 `no_geometry_relations`：显式几何信息是否提供额外贡献。
4. 同时看总错误、各类别错误、候选覆盖率和成本；不能只选择一个改善的指标或几个成功案例。

如果只有 `regions` 改善而 `relations` 没有改善，应保留局部信息的结果，不把收益归于关系模块。若 `relations` 与去几何组相当，就不能宣称显式几何输入带来了额外作用。

初筛有价值后，再用新输出目录将 `--seeds` 改为 `2026 3407` 做重复验证。种子重复不增加独立患者数。现有验证数据已参与多轮探索，本轮仍属于内部开发结果，不能写成独立外部验证或真实解剖 ROI 定位结果。

## 回传结果

在项目目录执行，压缩包也生成在该目录：

```bash
tar --exclude='*.pt' \
  -czf region_relation_ablation_v1_results.tar.gz \
  region_candidate_preview_v1 \
  region_relation_ablation_v1

gzip -t region_relation_ablation_v1_results.tar.gz
sha256sum region_relation_ablation_v1_results.tar.gz
```

这会保留配置、训练日志、输入流校验、候选示意及逐图评价，排除体积较大的权重。服务器上的原始权重仍应保留，以便后续复核。
