# 仅使用renew新数据：服务器实验

本轮按用户决定完全使用 `dataset/renew/` 第一层93张PNG；不使用旧数据或旧 `data_splits/`。原数据指纹和31个患者编号已技术核验，文件名标签不代表额外临床复核。

模型均为ImageNet V2预训练ResNet-50，两阶段整体微调；比较 `original` 原裁剪与 `full_frame` 完整原图等比例缩放补边。完整原图方案没有自动识别内镜ROI，也未删除截图文字。不得把它称为已经证实有效的改进算法。

## 1. 通过Git同步代码

本轮代码提交并推送到仓库后，在服务器执行：

```bash
cd /opt/data/private/AirwayRecognition
git pull --ff-only origin master
conda activate airway
```

Git同步本轮新增模块、入口、测试和文档；已上传的 `dataset/renew/` 原地使用。无需上传或解压源码工具包，也无需重新安装服务器原airway环境。确认pull成功后继续数据检查。

## 2. 数据检查与输入预览

```bash
bash scripts/run_renew_server.sh check
bash scripts/run_renew_server.sh preview
nvidia-smi
```

默认数据位置 `dataset/renew`，默认新输出 `outputs/renew_resnet_cv`。check校验93张/31编号、完整图像指纹、每位三类、患者五折隔离和固定支持计划；不训练，也不会改旧划分。check在无CUDA环境也能审查数据，正式run必须有CUDA。

预览位于：

```text
outputs/renew_resnet_cv/preview/inputs.png
outputs/renew_resnet_cv/preview/rotation_qa.png
```

inputs依次是完整原图、原评估裁剪、完整原图补边；rotation_qa为新方案−15/0/+15度的训练几何。仅作输入检查，实际训练还有颜色增强。首6张按文件名固定选择，无错误样本挑选。原图中既有文字未脱敏，预览只供本地研究使用。

## 3. 第一阶段：10次全训练池对照

先查看nvidia-smi，选择空闲GPU。下面的0只在0号卡空闲时使用；其他卡改为相应编号。

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_renew_server.sh run --phase full \
  > renew_resnet_full.log 2>&1 &
tail -f renew_resnet_full.log
```

Ctrl+C退出tail不会停止后台训练。第一阶段共2方法×5fold=10次。五折按患者分组：每折训练24或25人、留出6或7人，同患者三张图不会跨训练/留出。full是该折训练池全量，不是拿全部93张训练后再测同一批图。

每run从相同ImageNet权重和分类头种子42重新开始；固定140次头部更新＋560次整体微调，batch16、BN运行统计冻结。每折只评末轮，不用留出折挑epoch、阈值或checkpoint。GPU耗时以首个run实际记录为准；建议full至少预留2GB，全部110次至少15GB输出磁盘空间。

## 4. 打包全量阶段，统一回传分析

全10次完成后执行：

```bash
bash scripts/run_renew_server.sh pack
```

生成：

```text
outputs/renew_resnet_cv_full_results.tar.gz
outputs/renew_resnet_cv_full_results.tar.gz.sha256
```

在Windows PowerShell下载：

```powershell
scp -P 25137 "root@172.20.35.182:/opt/data/private/AirwayRecognition/outputs/renew_resnet_cv_full_results.tar.gz*" "C:\Users\40763\Downloads\ToDesk\"
```

结果包不含权重、患者图片或输入图册；包含患者编号/文件名的清单与逐图预测，仅供授权研究分析。权重保留在服务器。pack不需要图片或CUDA，但需要匹配的统计依赖；会重算并核对结果，拒绝不完整的阶段或覆盖已有结果包。

## 5. 第二阶段：5/10位训练患者

全量阶段运行正常并完成后，可以在同一版本代码、同一环境与输出目录继续：

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_renew_server.sh run --phase fewshot \
  > renew_resnet_fewshot.log 2>&1 &
tail -f renew_resnet_fewshot.log
```

此阶段新增100次：2方法×5fold×2预算×5支持重复。

- 5-shot：从该折训练池选5位患者，每位三类，共15张。
- 10-shot：同一排列的前10位，共30张，包含对应5-shot支持集。
- 两方法共享支持集、初始化、更新次数；增强随机序列因输入方案不同而不相同。

完整数据方案是否在full胜出，不作为删除few-shot负结果或筛选有利支持集的理由。所有结果保留。两阶段全部110次结束后再次pack，得到独立的 `renew_resnet_cv_all_results.tar.gz` 与校验文件。

## 6. 恢复、隔离与结果解释

中断后原命令重跑：已完成且通过哈希验证的run会跳过；未完成run移入 `incomplete_attempts/` 后整次重算。改变源码、数据、训练设置或环境时拒绝继续旧结果，需明确新输出目录。不要在同一输出目录同时启动两个进程，也不要在训练期间更换代码。

若指定其他路径，check/preview/run使用相同的 `--image-root` 与 `--output-root`，pack使用对应 `--output-root`。路径可迁移，但来源文件指纹和协议必须一致。

主要查看每个method/budget/repeat拼接5fold后的93张OOF预测。每位患者只在一个fold被评估；五个支持重复共享这31位患者，不是155位独立患者。full只有1份完整OOF结果，fewshot各5份，报告的是支持抽样波动。五折训练池重叠，不把fold当独立人群做普通显著性检验。

这是一轮开发性交叉验证，没有额外独立外部验证队列。候选同时改变裁剪、比例形变、有效分辨率与旋转边缘处理；完整输入的作用必须由结果检验，不能预先承诺提高。
