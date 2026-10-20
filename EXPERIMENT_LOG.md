# 实验记录

## 2026-10-20：训练期注意力可学习性对照

- 协议：`attention_training_control_v1`。
- 研究问题：从相同常数注意力起点出发，允许注意力学习是否改善分类表现。
- 预定主比较：`learned_ce − fixed_ce`；两组 EQ=PF=0，均保留旋转辅助前向。
- 可选比较：`learned_regularized − learned_ce`，检验 EQ+PF 联合正则。
- 预定服务器种子：42、2026、3407；每模型固定5+20轮，以末轮权重为主结果。
- **患者数据实验状态：待服务器运行，无新效果结论。** 后续需完整保留正向、零差异和负向结果。
- 详细设计：[独立评审](docs/ATTENTION_TRAINING_DESIGN_REVIEW_20261020.md)。
- 执行方法：[服务器运行与打包说明](docs/ATTENTION_TRAINING_ABLATION_RUN.md)。

### 本地实现验证

- 新增训练／评价测试与相关旧逻辑回归：**89项通过**。覆盖模型冻结与更新、两阶段训练、输入和前向随机状态、BN调用、检查点重载、数据隔离、配对统计、来源验证及拒绝覆盖。
- 真实 ResNet-50 CPU 合成数据校验：6张训练图、6张验证图、3组、每组1+1轮，无预训练权重；测试集仅有成员记录，测试图像故意不存在。训练后保存、重载和配对评价完成，共126行预测。
- 初始状态、各轮输入及执行记录跨组一致；固定组全部42个图像／角度组合的注意力均为0.5、空间标准差为0。
- 合成校验不属于患者数据实验，不用于报告识别性能或技术效果。
- 临时校验目录：`/private/tmp/airway_attention_training_smoke_bi_qk9yu/`，不作为永久研究数据存储。
- 本地多进程测试首次受macOS沙箱的OpenMP共享内存限制中断；在沙箱外最终回归全部通过。合成训练的自动评价子进程也遇到该环境限制，随后以单线程单独评价完成。未为此修改训练或统计逻辑。
- 本机未验证CUDA运行；实际服务器训练和三种子效果汇总仍待完成。

测试范围：`test_attention_training_ablation.py`、`test_attention_training_pairs.py`、`test_rotation_ablation_training.py`、`test_rotation_pairs.py`、`test_attention.py`、`test_train.py`。
