# ResNet-18 几何实验：seed43 配对复现

本轮只复核训练随机性：相同5折、support repeat0、5/10-shot、CE/plain/stable/shuffled、教师和训练预算，训练种子由42改为43，共40次。打乱关系的规则种子固定42，辅助头和关系候选池的独立种子保持原值。不重新训练全量基线，不增加新模块，不改几何权重0.1。

这是新增一次诊断性复现，不是新患者验证，也不能用两个种子证明支持抽样稳定性。两次种子的两个预算均需报告，不能只保留更好的一个。5-shot为5个训练编号/15张图，10-shot为10个编号/30张图。

## 1. 更新并核对已有结果

```bash
cd /opt/data/private/AirwayRecognition
git pull --ff-only
conda activate airway

bash scripts/run_geometry_replication_server.sh check
```

默认目录：

| 用途 | 服务器项目内路径 |
|---|---|
| 已核验图片 | `dataset/renew` |
| 旧seed42的45次完整结果，只读核验 | `outputs/renew_geometry_fewshot` |
| 已有教师缓存，复用 | `outputs/renew_geometry_teacher_cache` |
| 新seed43的40次结果 | `outputs/renew_geometry_seed43` |

检查包括旧45次完整性、原数据/分折/支持集、已执行源码、教师缓存对应关系和新40次计划。默认使用现有缓存，不需要重新下载教师、生成视差或加载`teacher_paths.env`。如果目录不同，可以用`--image-root`、`--reference-root`、`--cache-root`、`--output-root`明确指定；后续训练命令保持相同参数。新输出目录不得与旧结果、图片或缓存目录重叠。

`check`会建立新计划及只含JSON/CSV的seed42参考快照。它不会启动训练。检查通过后再运行下一步；如果拒绝缓存或源码不一致，应先查看具体错误，不要删旧结果强行重建。

## 2. 后台运行40次

```bash
nohup bash scripts/run_geometry_replication_server.sh run \
  > renew_r18_seed43.log 2>&1 &

tail -f renew_r18_seed43.log
```

前几次任务是fold0、5-shot的四种方法，然后继续10-shot及其他折。`head`阶段只训练分类头，几何损失为0；`finetune`阶段几何组才使用辅助损失。CE组整个过程几何损失均为0。

按Ctrl+C退出`tail`只停止查看日志，不会停止`nohup`后台训练。不要同时向同一输出目录启动两个训练、状态检查或打包进程。训练结束后再执行状态检查。

脚本核验已完成run后跳过；未完成run归档后从头重训，不能当作优化器中途续训。源码、配置、参考结果或缓存变化时拒绝混用结果。正式训练要求CUDA和与旧实验匹配的运行环境。

## 3. 完成后打包

```bash
bash scripts/run_geometry_replication_server.sh status
bash scripts/run_geometry_replication_server.sh pack
```

状态应为40/40。打包会输出结果`.tar.gz`及同名`.sha256`，下载这两个文件回传分析即可。结果包包括新预测、训练记录和固定的seed42参考快照，不含原图、权重、教师缓存NPZ或第三方源码；旧结果目录保持原样。

状态检查和打包可使用新输出中的参考快照，在没有原图、教师缓存、旧参考目录和权重的机器上完成，仍需要兼容的指标计算依赖。

## 4. 分批运行与解释范围

可用`--methods ce plain stable shuffled`、`--budgets 5 10`、`--folds 0 1 2 3 4`限制本次执行范围；这些过滤不会改变40次总计划。默认打包要求40次完整，不能把一小部分当成整个复现完成。建议本轮直接完整运行，避免只看有利预算。

新种子改变分类头初始化、采样、图像增强和dropout等训练随机性。打乱目标规则固定，但增强后的有效关系集合仍可变化，不能说所有逐步监督标签完全不变或只隔离优化器噪声。

比较时按预算分别查看每个种子的准确率、宏F1、纠错和新增错误，重点是plain相对CE、stable相对plain的差异是否再次出现。几何损失较低不等于分类更好，shuffled也不是梯度强度完全匹配的负控。两个种子仍是小规模诊断，之后的支持集重复或新机制需要另行固定协议。
