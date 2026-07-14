# 专利补充实验跟踪表

所有正式训练均固定 `split_seed=42`。`seed` 仅表示训练随机种子。

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R001 | M0 | 冒烟测试 | baseline/full，seed=42，各 1+1 epoch | 固定患者 test | 输出完整性 | MUST | TODO | 单独使用 `patent_runs_smoke` |
| R002 | M1 | 核心基线 | baseline，seed=42 | 固定患者 test | Acc/F1/AUC/robustness | MUST | TODO | 与 R003 配对 |
| R003 | M1 | 核心方案 | full，seed=42 | 固定患者 test | Acc/F1/AUC/robustness | MUST | TODO | 与 R002 配对 |
| R004 | M1 | 跨种子稳定性 | baseline，seed=2026 | 固定患者 test | Acc/F1/AUC/robustness | MUST | TODO | 与 R005 配对 |
| R005 | M1 | 跨种子稳定性 | full，seed=2026 | 固定患者 test | Acc/F1/AUC/robustness | MUST | TODO | 与 R004 配对 |
| R006 | M1 | 跨种子稳定性 | baseline，seed=3407 | 固定患者 test | Acc/F1/AUC/robustness | MUST | TODO | 与 R007 配对 |
| R007 | M1 | 跨种子稳定性 | full，seed=3407 | 固定患者 test | Acc/F1/AUC/robustness | MUST | TODO | 与 R006 配对 |
| R008 | M2 | 核心结论门控 | 聚合 R002–R007 | 固定患者 test | 配对差值 CI、扰动曲线 | MUST | TODO | 决定是否进入 M3 |
| R009 | M3 | 有效视野消融 | crop_only，seed=42/2026/3407 | 固定患者 test | Acc/F1/AUC/robustness | MUST | BLOCKED | 等待 R008 通过 |
| R010 | M3 | 注意力消融 | attention，seed=42/2026/3407 | 固定患者 test | Acc/F1/AUC/robustness | MUST | BLOCKED | 等待 R008 通过 |
| R011 | M3 | 双正则消融 | regularized，seed=42/2026/3407 | 固定患者 test | Acc/F1/AUC/robustness | MUST | BLOCKED | 等待 R008 通过 |
| R012 | M3 | 五组总汇 | 五变体 × 三种子 | 固定患者 test | 消融均值/标准差/曲线 | MUST | BLOCKED | baseline/full 自动复用 |
| R013 | M4 | 解释性 QC | baseline/full 对齐 Grad-CAM | 固定患者 test | enrichment/样例 | MUST | TODO | 自动指标仅作探索性 QC |
| R014 | M4 | 专家盲评 | baseline/full 匿名热图 | 分层测试样本 | 相关性、伪特征依赖、κ/ICC | MUST | TODO | 至少两名专家 |
| R015 | M4 | 失败案例审计 | baseline/full | 错误及鲁棒性退化样本 | 错误类型/置信度 | MUST | TODO | 限定最终表述范围 |
