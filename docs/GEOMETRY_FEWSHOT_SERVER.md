# ResNet-18 选择性相对几何蒸馏：服务器运行

此脚本独立于旧renew实验。正常预训练ResNet-18、完整视野输入、相同患者分折和repeat 0支持集；5次全量CE＋40次5/10-shot四组＝45次。第一轮只是开发筛查，不证明新颖性、深度精度或外部泛化。这里不包含后续“任务相关性联合筛选”。

脚本通过Git同步，在服务器项目目录执行`git pull --ff-only`即可更新，不需要另外上传或解压源码包。依赖沿用现有airway环境，至少需要兼容的torch/torchvision、numpy/scipy/scikit-learn、Pillow；教师setup只安装缺少的小型依赖，不替换torch。正式训练必须CUDA；`--smoke`是CPU随机权重两步调试模式，不能打包为正式结果。

## 1. 检查数据并生成固定计划

```bash
conda activate airway
cd /opt/data/private/AirwayRecognition
git pull --ff-only
bash scripts/run_geometry_server.sh check \
  --image-root dataset/renew \
  --output-root outputs/renew_geometry_fewshot
```

`dataset/renew`应直接包含已核验的93个PNG；若图片位于子目录，所有命令统一使用真正的PNG目录。检查会验证原始93图指纹、31个编号、患者隔离与嵌套5/10支持集。不会下载权重或训练。

生成`dataset.csv`、`supports/*.csv`、`support_plan.csv`和`suite.json`。默认训练140次分类头更新＋560次整体微调，batch16、seed42、BN统计冻结、固定末轮评估，几何权重0.1且只在整体微调阶段启用。所有R18组使用同一初始化、图像增强和更新预算；数据加载workers=0便于严格重复。ResNet18 ImageNet V1与旧ResNet50 V2预训练配方不同，核心比较应在R18各组之间。

## 2. 下载并准备冻结教师

```bash
bash scripts/setup_geometry_teacher.sh --install-deps
source pretrained/EndoDAC/teacher_paths.env

python -m scripts.prepare_endodac_teacher \
  --image-root dataset/renew \
  --suite-root outputs/renew_geometry_fewshot \
  --scope qa \
  --cache-root outputs/renew_geometry_teacher_cache \
  --teacher-repo "$GEOMETRY_TEACHER_REPO" \
  --teacher-weights "$GEOMETRY_TEACHER_WEIGHTS"
```

setup固定官方EndoDAC源码提交并下载完整checkpoint，默认不需要另下DepthAnything初始化权重。Google Drive不可访问时，脚本给出官方手动下载地址与本地放置路径，不能用随机权重冒充教师。

QA仅处理预定fold 0、5-shot支持图，生成局部QA图册及缓存统计。确认没有非有限值、常数输出、明显空间错位后生成各正式支持集的并集缓存：

```bash
python -m scripts.prepare_endodac_teacher \
  --image-root dataset/renew \
  --suite-root outputs/renew_geometry_fewshot \
  --scope all-supports \
  --cache-root outputs/renew_geometry_teacher_cache \
  --teacher-repo "$GEOMETRY_TEACHER_REPO" \
  --teacher-weights "$GEOMETRY_TEACHER_WEIGHTS"

bash scripts/run_geometry_server.sh check \
  --image-root dataset/renew \
  --output-root outputs/renew_geometry_fewshot \
  --phase fewshot \
  --cache-root outputs/renew_geometry_teacher_cache
```

教师完全冻结，按图独立推理，不拟合并集统计。每个训练run只读取自己的支持图片；留出折推理只读取RGB。官方输出是视差代理，排序稳定不代表真实深度准确。图册只供本地查看，可能包含原截图文字，不进入结果包。

## 3. 后台运行45次正式训练

```bash
nohup bash scripts/run_geometry_server.sh run \
  --image-root dataset/renew \
  --output-root outputs/renew_geometry_fewshot \
  --cache-root outputs/renew_geometry_teacher_cache \
  --phase all > renew_geometry_fewshot.log 2>&1 &

tail -f renew_geometry_fewshot.log
```

`ce`是正常R18分类微调；`plain`是普通相对排序辅助；`stable`筛掉在预定外观扰动中不稳定的关系；`shuffled`保留相同筛选位置和目标分布但打乱空间对应。

所有组训练时都有同样的轻量辅助头，CE不使用它。每次训练保存`model.pt`供严格恢复、`student.pt`供单图部署。后者已经删除geometry_head，并核验重载输出一致；可以装入标准ResNet18（分类头为Dropout0.3＋Linear(512,3)）。结果包不包含任何权重。

脚本自动跳过已完成且核验通过的run。中断中的run从头重训，旧半成品移入`incomplete_attempts`；不假称恢复了优化器中途状态。源码、配置、数据或教师变更时拒绝混用旧结果，需新输出目录。不能同时在同一输出目录运行两个训练/打包进程。

若教师下载暂未成功，可以先跑15次CE基线，不需要几何缓存：

```bash
nohup bash scripts/run_geometry_server.sh run \
  --image-root dataset/renew \
  --output-root outputs/renew_geometry_fewshot \
  --phase all --methods ce > renew_r18_ce.log 2>&1 &
```

CE结束、教师缓存准备好后，接着运行：

```bash
nohup bash scripts/run_geometry_server.sh run \
  --image-root dataset/renew \
  --output-root outputs/renew_geometry_fewshot \
  --cache-root outputs/renew_geometry_teacher_cache \
  --phase fewshot --methods plain stable shuffled > renew_r18_geometry.log 2>&1 &
```

这些过滤只改变本次执行范围，不改变全局45次计划。也可先`--phase fewshot --budgets 10`再跑5-shot；两个预算都需报告。不要根据单张错误或已有留出结果改损失权重、选种子。

## 4. 检查完成状态、打包结果

训练运行时看日志；结束或停止后执行状态检查（同一输出目录有互斥锁）：

```bash
bash scripts/run_geometry_server.sh status \
  --output-root outputs/renew_geometry_fewshot

bash scripts/run_geometry_server.sh pack \
  --output-root outputs/renew_geometry_fewshot \
  --phase all
```

会输出一个带时间标识的`*_all_*_results.tar.gz`及同名`.sha256`。默认all要求全部45次完成。若只跑完全量五折，可用`pack --phase full`；只跑完CE可用`pack --phase all --methods ce`。局部范围包通过`package_scope.json`明确标记，不能当作整个实验完成。

打包可以在没有图片、教师缓存、GPU和模型权重的机器上执行，但仍需要兼容的指标计算依赖。每次打包均从逐图预测重新计算指标；不相信手改的汇总表。包含计划、教师来源摘要、训练历史、逐图概率、指标与校验，不包含图像、伪深度、模型权重或教师源码。编号和图像名仍属于研究记录，回传请使用现有私下通道。

## 5. 如何看结果

- `REPORT.md`和`oof_metrics.csv`：仅完整五折产生合并OOF指标；每组93张，不能把重复预算当新患者。
- `oof_comparison.csv`：各几何组相对同预算R18 CE的差。
- `history.csv`：分类/几何损失、有效与保留关系数、打乱实际改变的关系比例、辅助头梯度量。
- 如果打乱未实际改变关系，报告会提示负对照没有辨别力；不能把它解释为几何无效。
- 首轮只有一个支持抽样重复，尚不能声称减少抽样波动。后续任务相关性选择、普通特征蒸馏和多重复需要另行版本化。
- 既有ResNet50全量和few-shot结果仍保留。这里不重训或覆盖ResNet50；匹配的repeat 0参考可从原renew套件结果复用。

正式权重加载、真实患者训练和GPU显存/速度均需在服务器实际验证。本地合成测试不代表真实实验效果。
