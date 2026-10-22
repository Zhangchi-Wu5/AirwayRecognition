# 全量强基线复现：服务器运行说明

本轮先运行5次新的全量训练，核对原强基线和当前小样本流程的区别。完成后打包回本地分析，再确定5-shot、10-shot的方法比较。入口不会自动启动下一轮小样本实验。

| 流程 | 训练种子 | 本次操作 |
|---|---|---|
| 历史强基线原流程 | 42、2026、3407 | 3次重新训练 |
| 当前few-shot流程的full | 42 | 校验并引用已完成结果 |
| 当前few-shot流程的full | 2026、3407 | 补2次训练 |

历史三个种子的最终验证准确率分别为96.875%、100%、97.9167%，均值98.2639%。本轮记录真实复现结果，不以重新达到98%作为完成条件。

## 1. 拉取代码并检查

在服务器原项目目录执行：

```bash
cd /opt/data/private/AirwayRecognition
git status --short
git pull --ff-only
conda activate airway
nvidia-smi
```

如果Git提示本地修改冲突，先保留并核对修改，不使用强制覆盖。沿用已经完成上一轮实验的环境，不为这次复现主动升级依赖。

历史实验实际使用`data_splits_patent`，本轮当前流程使用`data_splits`。优先保留这两份服务器原文件。确认0号GPU可用后执行：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_baseline_reproduction_server.sh check \
  --historical-splits-dir data_splits_patent
```

如使用其他空闲GPU，统一更换编号。默认参考目录是上一轮服务器的`outputs/resnet_fewshot_suite_20261022`，图片目录是`dataset`。

检查会要求Python及关键依赖版本、GPU型号与被引用的seed42运行一致；不一致时先核对原环境，不跳过校验。新运行还会记录实际数值设置和线程设置，恢复时检查是否变化。

如果原`data_splits_patent`已不在服务器，先查找原CSV。仅有`data_splits`时，可在所有命令中改为`--historical-splits-dir data_splits`；工具会区分“成员一致的重跑”与原始输入流是否复现，不能仅凭患者集合一致宣称精确复现。

## 2. 运行全量对照

检查成功后执行：

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_baseline_reproduction_server.sh run \
  --historical-splits-dir data_splits_patent \
  > baseline_reproduction.log 2>&1 &

tail -f baseline_reproduction.log
```

`Ctrl+C`只退出日志查看。中断后用相同参数重新执行`run`，可更换日志文件名保留旧日志。已完成并通过校验的组会复用；中断组归档后从初始权重重新训练。

默认输出目录为`outputs/resnet_baseline_reproduction_20261022`。不要同时运行两个指向同一输出目录的任务。训练期间不要修改脚本或更新环境，否则恢复时会遇到来源校验不一致。

历史原流程单次约3.5分钟；本轮5次新训练预计十几至数十分钟，尚未实测。建议预留1小时窗口和至少5GB磁盘。

## 3. 打包并回传

全部完成后执行：

```bash
bash scripts/run_baseline_reproduction_server.sh pack \
  --historical-splits-dir data_splits_patent
```

下载以下文件：

```text
outputs/resnet_baseline_reproduction_20261022_results.tar.gz
outputs/resnet_baseline_reproduction_20261022_results.tar.gz.sha256
```

模型权重继续留在服务器。结果包包含六组对照的指标、逐图预测、日志、配置和核验记录，包括原current seed42的来源证据。

打包可离线执行，不需要GPU、原数据或旧suite目录；权重若仍在结果目录中，会校验其哈希。恢复训练则必须保留完整权重。

## 怎样判断是否复现

核验分三个层次：

1. 源码、配置、患者划分、CSV与来源是否匹配。
2. 初始化和25轮实际训练输入流是否与历史一致。
3. 最终模型状态、预测和准确率是否与历史一致。

CSV字节不同可能只是路径改写，也可能影响图片顺序；报告会记录差别，并用实际训练输入流继续核对。历史没有保存TF32等全部数值设置，不能补写成已知。两种训练流程分别在独立进程中执行，防止全局设置相互影响。

原流程仍使用原模型构造、按图片索引隔离的随机增强、末批14张和最终checkpoint。当前流程仍使用满批重复采样、原有BN与学习率规则。未改写上一轮已完成结果，也没有读取锁定测试图像。

如果流程匹配但准确率不同，保留真实结果并分析原因；如果来源或输入流不匹配，先解释差异。两种情况都不会自动扩展到小样本方法比较。
