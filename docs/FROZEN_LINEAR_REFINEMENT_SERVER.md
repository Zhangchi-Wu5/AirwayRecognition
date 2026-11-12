# 冻结ResNet-18线性头：一次有限的正则范围补充

本实验对应冻结特征筛查的第二版。第一版505个线性任务均在支持患者内部验证中选择了C=10上限，且平均验证损失随C=0.1、1、10依次下降。因此这次仅补查更弱正则的两个候选值，保留第一版结果。

## 固定内容

- 复用第一版服务器运行目录中的93张图特征缓存、逐图L2归一化、患者隔离五折、原支持集以及100套固定抽样。
- C候选一次性固定为 `0.1、1、10、100、1000`；仍在所选5位患者内部做留一患者验证，按平均交叉熵选参，差值不超过 `1e-12` 时取较小C。
- 保持带截距的多项逻辑回归、L2、`lbfgs`、容差 `1e-8`、最多10000次迭代；不收敛则报错，不保存为成功结果。
- 仅拟合505个最终线性分类器；各任务包含25次内部验证拟合及1次最终拟合，共13130次小型CPU拟合。不会重新提取ResNet特征，也不会微调主干。
- 每个支持集仅评价内部选出的最终模型。外层测试图不参与C选择，不输出按外层分数挑C的结果。
- 原有C=0.1、1、10的内部验证损失必须在固定数值容差内复现第一版；若不一致，脚本停止并报告，避免把环境或流程变化当作扩大范围的效果。

这是依据第一版内部记录提出的开发性方案修订，不是原先预注册的确认性实验，也不是新算法或专利效果证明。更大C可能改善概率损失，但不保证分类准确率提高。本轮之后不会自动继续扩展候选范围。

## 服务器运行

沿用完成第一版时的 `airway` 环境。第二版只需要CPU，不需要GPU或原始图片；需要服务器原运行目录保留的特征数组。下载的第一版结果包不含这些数组，因此引用路径应指向原运行目录。

```bash
cd /opt/data/private/AirwayRecognition
git pull --ff-only
```

先检查第一版结果、缓存及环境，并建立独立第二版输出：

```bash
bash scripts/run_frozen_linear_refinement_server.sh check \
  --reference-root outputs/renew_frozen_screen \
  --output-root outputs/renew_frozen_linear_v2
```

看到 `PASS` 后运行：

```bash
nohup bash scripts/run_frozen_linear_refinement_server.sh run \
  --reference-root outputs/renew_frozen_screen \
  --output-root outputs/renew_frozen_linear_v2 \
  > renew_frozen_linear_v2.log 2>&1 < /dev/null &
```

查看日志：

```bash
tail -f renew_frozen_linear_v2.log
```

`Ctrl+C`只结束日志查看。若中断，重复相同运行命令即可，完整任务核验后跳过。继续使用相同输出时应保持代码和数值计算环境不变；不要直接改旧实验的C配置或覆盖旧输出。

检查会把核验所需的第一版文本记录和小型特征数组复制到第二版目录，后续运行使用这份固定副本。原目录保持只读；不要手动改动第二版保存的参考记录或数组。

默认运行全部505个任务；`--phase reference`或`--phase draws`仅用于分批执行已固定的任务。实际耗时以新日志为准，较大的C可能需要更多迭代。

## 打包与回传

全部任务完成后：

```bash
bash scripts/run_frozen_linear_refinement_server.sh status \
  --output-root outputs/renew_frozen_linear_v2

bash scripts/run_frozen_linear_refinement_server.sh pack \
  --output-root outputs/renew_frozen_linear_v2
```

正式包需要505个任务、101组完整五折结果、9393条新预测全部通过核验。取回打印出的 `.tar.gz` 和 `.sha256`。

结果保存新旧方案的来源关系、相同支持下的分类变化、内部C损失曲线、选中C频次、原支持结果以及100套抽样分布。结果包包括离线核验与新旧比较所需的第一版文本记录，不含图片、ResNet权重或特征矩阵。编号和结果属于研究数据，不推送公共Git。

## 回传后重点看什么

1. 内部选参是否仍集中在最大C，还是已经出现损失的转折。
2. 相同支持下，第二版修正多少图片、同时新增多少错误；区分概率损失改善与准确率改善。
3. 原支持集是否接近既有整体微调CE；100套抽样的均值和波动如何变化。旧CE只在原支持集上比较，不能与100套新抽样的均值当作配对实验。

保留所有结果。如果新网格仍大量选上限，报告这一限制，不自动启动第三轮追参。100套抽样仍是同一31位开发患者的重复使用，不是独立临床验证。
