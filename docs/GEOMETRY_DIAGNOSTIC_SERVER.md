# 几何监督诊断：服务器执行说明

本轮回答两个问题：原始截图暗框是否进入几何监督；真实几何是否比同数量随机筛选、同位置打乱关系更有助于另一个支持编号的分类。它是独立诊断，不是新的分类准确率实验，也不训练自动筛选器。

默认10个单元：原5折×5/10-shot，support repeat0。每单元按既有支持编号顺序取末位编号为B，其余为A；5-shot是A四个编号/12图＋B一个编号/3图，10-shot是A九个编号/27图＋B一个编号/3图。B不参与模型准备，外层测试图不参与诊断更新。没有按错误图片挑B，首轮也不是完整留一轮换。

## 1. 更新和检查

```bash
cd /opt/data/private/AirwayRecognition
git pull --ff-only
conda activate airway

bash scripts/run_geometry_diagnostic_server.sh check
```

| 用途 | 默认目录 |
|---|---|
| 原始93张PNG | `dataset/renew` |
| 已完成seed42的45次结果 | `outputs/renew_geometry_fewshot` |
| 已有教师缓存 | `outputs/renew_geometry_teacher_cache` |
| 本轮独立输出 | `outputs/renew_geometry_diagnostic` |

不需要重新下载或生成EndoDAC缓存。检查将冻结计划、源码、数据和每单元A使用的教师缓存，并保存旧参考的记录快照。若实际图片在子目录，用`--image-root dataset/renew/气管镜`，后续run也使用相同参数；其他目录用`--reference-root`、`--cache-root`、`--output-root`指定。

新输出与图片、缓存、旧结果目录必须分离。遇到哈希或来源不一致先排查，不要删除旧结果绕过校验。正式运行需要CUDA；本轮运行环境独立记录，重启同一输出时必须一致。

## 2. 后台运行

```bash
nohup bash scripts/run_geometry_diagnostic_server.sh run \
  > renew_geometry_diagnostic.log 2>&1 &

tail -f renew_geometry_diagnostic.log
```

Ctrl+C退出tail只停止看日志。运行期间不要同时对同一输出执行第二个run、status或pack。

每单元先用A训练分类头140步，再单独校准几何头140步。两阶段均使用角度0、无颜色扰动的缓存特征，共享主干和BN始终冻结；分类头训练保留dropout。几何头校准使用原plain目标，可能偏向该目标。随后在固定视图、关闭dropout的同一个基点上做临时更新比较，比较完完整恢复模型，不累计训练分支。

比较包括：原plain、排除外围暗框后的plain、三个固定种子的同数量随机筛选、三个固定种子的同位置关系打乱。三次置换不是三组新患者。

临时更新只作用于conv1及layer1–3参数，排除独立bn1、layer4和两个头；所有BN运行统计保持冻结。这是匹配位移强度的SGD方向诊断，不是复现原700步AdamW训练。记录原始梯度范数/夹角，以及C0（纯分类更新后）、Cr（加真实关系后）、Cs（加打乱关系后）的B分类损失。C0−Cr或Cs−Cr为正，仅表示相对该对照更好，不保证低于未更新模型的损失，更不等于测试准确率提升。无可用关系/近零梯度会明确记录跳过。

输出的外围暗框mask是固定像素启发式，不是真组织标注；不删除原图、不改标签、不把内部暗腔直接当黑框。它不去除所有蓝色角标、边缘或反光。

新模块修复了辅助头初始化的CUDA随机状态隔离，历史45/40次源码和结果保持冻结。不能把本轮某个分支与旧CE直接当作同协议配对结果。

## 3. 完成后检查和打包

```bash
bash scripts/run_geometry_diagnostic_server.sh status
bash scripts/run_geometry_diagnostic_server.sh pack
```

默认应完成10/10。下载pack输出的`.tar.gz`和同名`.sha256`回传分析。结果包只含记录、表格及参考快照，不含患者原图、模型权重、教师NPZ或第三方源码。

已完成单元通过核验后跳过；未完成单元归档后从头运行，不是优化器中途续训。离线status/pack使用已存快照和完整性记录，不要求原始图片、缓存和历史结果目录仍在本机。

## 4. 结果如何决定下一步

先看真实有效pair触及暗框的比例，再看真实关系相对纯分类和打乱的损失差分是否在不同单元呈现一致方向；同时检查同数量随机筛选，避免把减少监督误当筛选有效。

单个B仅3张图，10单元不是10个独立临床样本，也不能据此直接挑选阈值或写成新算法已有效。如果存在可重复的信号，再预先固定更多B轮换/支持抽样和正式训练对照；如果真实关系不能稳定胜过匹配打乱，停止扩展当前几何机制。

可用`--folds`和`--budgets`限制执行范围，但默认pack要求全部10单元完成；小范围包必须显式给相同过滤，包内会标记范围。`--smoke`仅用于合成/随机权重的工程验证，不能打包为正式诊断结果。
