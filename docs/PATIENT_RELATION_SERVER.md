# ResNet-18 同患者类别差约束：5-shot 四组筛选

本轮直接训练三分类模型。每个训练患者提供隆突、右主支气管、左主支气管各一张图片；数据提供者已于2026-11-11确认同编号三图来自同一患者、同一次气管镜检查。这是提供者确认，不是独立临床复核。

训练期比较同患者不同类别的特征差，推理时仍仅输入单张 RGB。此次不使用深度教师、几何缓存、测试患者另外两张图或额外标注。方案有效性和新颖性均未得到本轮实验确认。

## 固定实验

| 方法 | 含义 |
|---|---|
| `ce` | 普通三分类交叉熵 |
| `supcon` | 交叉熵加普通监督对比损失 |
| `paired` | 交叉熵加同患者类别差方向一致性 |
| `regrouped` | 同样的类别差损失，但打乱跨类患者配对 |

仅5-shot：5个患者、每类5张，共15张训练图像。沿用已有患者隔离五折与支持集repeat 0，固定训练种子42、43，总计 **4×5×2=40次训练**。本入口不提供10-shot；首轮通过预定筛选条件后另行安排。

四组每步使用相同15张训练图，每图一幅随机增强；普通SupCon的同类正例来自另外4位训练患者。主干均为ImageNet预训练ResNet-18，分类头为Dropout(0.3)加三类线性层。先140次分类头更新，再560次整体微调；BN运行统计始终冻结。保持原全图预处理及主训练优化设置。辅助损失仅在整体微调使用，权重0.1；SupCon温度0.1；类别差先L2归一化（epsilon=1e-6），再比较不同患者同一种类别对的方向。

每轮完整支持批次与旧实验的单图随机批次不同，因此四组全部重新训练。旧实验数字仅作背景，不能代替本轮CE。两个训练种子共享支持集，不能当成两次独立患者抽样。当前队列已多次用于方案探索，结果属于开发性比较。

`regrouped`只改变辅助损失的关系索引：类别组成、每张图的使用次数和增强保持不变，任意类别对均不得保留原患者匹配。模型结构、初始参数和实际增强数据流均记录指纹以供配对核验。

## 服务器运行

使用原来的 `airway` 环境。在仓库根目录拉取代码：

```bash
cd /opt/data/private/AirwayRecognition
git pull --ff-only
```

先检查并生成固定计划。`--image-root`必须直接包含已核验的93张PNG；若以前使用的是`dataset/renew/气管镜`，下面两条带图片路径的命令均沿用该路径。

```bash
bash scripts/run_patient_relation_server.sh check \
  --image-root dataset/renew \
  --output-root outputs/renew_patient_relation_5shot
```

检查通过后运行40次训练：

```bash
nohup bash scripts/run_patient_relation_server.sh run \
  --image-root dataset/renew \
  --output-root outputs/renew_patient_relation_5shot \
  > renew_patient_relation_5shot.log 2>&1 < /dev/null &
```

查看实时进度：

```bash
tail -f renew_patient_relation_5shot.log
```

`Ctrl+C`只结束`tail`查看，不结束后台训练。正式训练要求CUDA；首次使用时预训练权重可能需要下载。无需启动EndoDAC或准备teacher cache。实际总耗时应根据前几个任务日志估算，本说明不预先承诺时长。

中断后重复相同`run`命令即可。通过验证的完整任务会跳过，未完成任务保留到失败尝试目录并从头重跑；不做训练步数中途续接。不要修改代码、数据或实验参数后继续复用旧输出目录：身份检查将拒绝混合结果。

脚本也提供`--methods`、`--folds`、`--seeds`用于分批执行。它们只选择固定40任务的子集，不改变完整计划；部分任务完成时不能得出整轮结论。

## 汇总与打包

训练结束后查看状态并重新核验汇总：

```bash
bash scripts/run_patient_relation_server.sh status \
  --output-root outputs/renew_patient_relation_5shot
```

`status`使用保存的预测与运行记录，无需原始图片、GPU或模型权重。仅当40次全部通过完整性检查后打包：

```bash
bash scripts/run_patient_relation_server.sh pack \
  --output-root outputs/renew_patient_relation_5shot
```

取回打印出的结果`.tar.gz`及其`.sha256`校验文件。结果包包含实验身份、支持清单、逐图预测、训练历史、指标与继续/停止判断，**不包含图片或模型权重**。权重留在服务器各任务目录。结果包含研究数据及文件编号，请保留在项目研究环境中。

用于工程验证的`--smoke`模式使用随机权重和极短训练，须使用独立输出目录，不能打包为正式结果，也不会给出正式继续结论。工程测试通过不表示候选算法有效。

## 如何决定下一步

主指标是每个种子五折合并的93张留出图片正确数。两个种子必须各自同时满足：

1. `paired`比`ce`至少净多判对2张（约2.15个百分点）。
2. `paired`严格优于`supcon`。
3. `paired`严格优于`regrouped`。

同时报告macro-F1、每类召回、逐折及逐患者变化，包括修正和新增错误。不完整结果保持待定；全部通过后才考虑10-shot与新的预抽支持集验证，脚本不会自动启动后续实验。

这是一条预先固定的工程筛选规则，不是统计显著性或专利授权标准。差分一致不能证明单图模型已经消除患者偏差。未通过时只停止当前具体版本，不将一次结果推广为所有相关方法都无效。
