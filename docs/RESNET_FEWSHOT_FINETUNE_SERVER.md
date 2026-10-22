# ResNet 小样本微调：服务器运行

> 当前推荐使用 [全部实验统一在服务器运行](RESNET_FEWSHOT_SUITE_SERVER.md)。下文保留旧的独立微调包用法；统一流程会先在服务器重跑冻结实验，再复用服务器生成的支持集进行微调。

上一轮冻结特征实验已在本机完成，无需在服务器重跑。这里运行的是下一轮：在完全相同的支持集上微调 ResNet，检查低标注下的下降是否仍然存在。

## 上传与解压

将交付包 `airway_fewshot_server_20261022.tar.gz` 上传至此前的服务器项目目录 `/opt/data/private/AirwayRecognition`。数据仍使用服务器已有的 `dataset/`，不需要重新上传图片。

```bash
cd /opt/data/private/AirwayRecognition
tar -xzf airway_fewshot_server_20261022.tar.gz
cd fewshot_server_20261022
conda activate airway
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
```

交付包解压到独立子目录，不覆盖服务器已有训练代码、划分或结果。请保留约 10 GB 以上空闲磁盘，81 个最终模型的权重预计合计约 8 GB。使用此前能运行 ResNet 的 GPU 环境即可；首次使用当前模型缓存时需要下载官方 ImageNet 权重。

若你的服务器项目目录不同，只修改第一条 `cd`。以下命令在 `fewshot_server_20261022` 中执行，`../dataset` 指向服务器原有图片目录。

## 先核对数据与抽样

```bash
bash scripts/run_fewshot_finetune_server.sh check
```

这一步检查患者隔离、图片内容及原支持集，生成计划，不训练模型。必须使用包内 `fewshot_splits_20261022`，不要自行替换为旧服务器的其他划分。核对失败时先查看错误，不能绕过检查继续训练。

接着建议在选定空闲 GPU 上做一次很短的运行检查，验证服务器依赖、CUDA、训练与保存均正常：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_fewshot_finetune_server.sh smoke
```

这只运行一份支持集、4 次更新，输出到独立的 `outputs/resnet_fewshot_finetune_smoke_20261022`，不进入正式结果。

## 正式运行与断点恢复

确认 `nvidia-smi` 显示 0 号 GPU 可用后，执行下面命令；如果使用其他空闲 GPU，修改 `CUDA_VISIBLE_DEVICES`。

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_fewshot_finetune_server.sh run \
  > finetune_server.log 2>&1 &
```

查看进度：

```bash
tail -f finetune_server.log
```

`Ctrl+C` 退出日志查看，不会停止后台训练。断开终端后后台任务仍可继续。不要在任务仍运行时重复启动同一输出目录。

若任务确实中断，再执行相同命令即可。恢复会校验源代码、配置、数据及完成结果，保留已完成的支持集；中断的一组重新训练，并非从组内某一轮继续。恢复日志建议改为 `finetune_resume.log`，保留原日志。

如希望先完成全量参考，将正式命令中的 `run` 改成 `full`；运行结束后用 `run` 完成其余 80 组。全量不会重复训练。仅跑全量时输出属于部分结果，不能作为完整小样本结论。

GPU 型号和读取图片速度会影响耗时，预计应预留数小时窗口。先运行全量组可以获得本机时间参考；不要把本地冻结特征实验约三分钟的耗时套用到这一轮。

快捷脚本固定使用 CUDA、batch16、workers4，并在 `run` 中自动传入 `--resume`。若图片目录不同，在每条命令前一致设置 `AIRWAY_IMAGE_ROOT=/实际图片目录`。如需调整 batch 或 workers，请用完整 Python 入口，并保持准备与正式运行配置一致；已有结果不能通过恢复切换训练配置：

```bash
python -u -m scripts.run_resnet_fewshot_finetune \
  --reference-root fewshot_reference_20261022 \
  --splits-dir fewshot_splits_20261022 \
  --image-root ../dataset \
  --output-root outputs/另一个结果目录 \
  --device cuda --batch-size 16 --num-workers 4
```

## 本轮协议

- 每类 1／5／10／20 张各 20 份支持集，另有全量 446 张一次；与上一轮逐一对应。
- 每组从官方 ImageNet ResNet-50 重新初始化，训练随机种子固定为 42，支持集抽样种子沿用 20261022。
- 先训练分类头，再解冻整体微调。AdamW、学习率 1e-3／1e-4、权重衰减 1e-4、dropout 0.3。
- 固定 BatchNorm 运行统计；第二阶段允许更新其仿射参数。普通裁剪、旋转及颜色增强，不做左右翻转。
- 默认 batch16，每轮 28 个更新步骤、448 次图像取样。小支持集通过重复洗牌补满批次。分类头 5 轮共 140 步，整体微调 20 轮共 560 步。
- 这里的“轮”是固定更新步骤，不是对小支持集自然遍历一次。这避免 1-shot 只更新极少次数；同时意味着小支持集每张图片会重复出现更多次。
- 使用固定最后一轮权重评价，不根据验证分数选最佳轮或提前停止。只评价原来的验证患者，不读取测试图片。

这回答“相同更新预算下，减少标注能达到怎样的表现”。它不是计算效率实验，也不是历史约98%成绩的逐位复现。与冻结线性头相比，微调、增强及分类头优化方式同时改变，不能将差值全部归因于单一因素。

支持集抽样标准差仍不等于患者泛化置信区间。固定优化种子旨在先测量训练患者选择的影响，不覆盖全部训练随机性。

## 回传结果

保留服务器上的模型权重。在 `fewshot_server_20261022` 目录中执行：

```bash
tar --exclude='*.pt' --exclude='*.pth' \
  -czf resnet_fewshot_finetune_results_20261022.tar.gz \
  outputs/resnet_fewshot_finetune_20261022 finetune_server.log
gzip -t resnet_fewshot_finetune_results_20261022.tar.gz
sha256sum resnet_fewshot_finetune_results_20261022.tar.gz
```

将 `resnet_fewshot_finetune_results_20261022.tar.gz` 回传，即可分析各预算的均值、抽样波动及相同支持集上的微调与冻结差异。如果有恢复日志，也一并加入压缩包。

本交付只完成运行工具与本地测试；服务器上的正式 81 组微调尚未启动，没有新的微调性能结论。
