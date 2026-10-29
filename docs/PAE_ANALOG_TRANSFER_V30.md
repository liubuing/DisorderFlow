# 类似物迁移验证 v30 结果（TCR-pMHC）

日期：2026-10-08。上游门控链 v23–v27 与预注册（`docs/PAE_ANALOG_PREREGISTRATION_V27.md`，sha256 65831f97…）在任何候选生成与 AF2 标签之前冻结。本文件是迁移验证的冻结结果入口；召回数字只来自 `results/pae_analog_evaluation_v30/evaluation.json`。

## 执行摘要

预注册主终点达成，为**正向但温和**的迁移信号：在 6 个暴露干净的 TCR-pMHC 评估家族上，组成驱动 Ridge 配方对教师（AF2-Multimer PAE）最优 20% 的召回为 **50.0%**（每家族 1/3 或 2/3），随机精确平局期望 21.4%；配对组件差异 **+28.6 个百分点，95% bootstrap CI [17.5, 39.7] pp**，6/6 家族不低于随机。仅组成特征与全特征集完全一致（50.0%）——几何/长度特征不改变家族内排序。ProteinMPNN 生成时似然 33.3%，训练均值基线 21.4%。

## 运行事实

- 教师：AF2-Multimer v3，模型 1+2，种子 7103/7111/7121，recycles 3，单序列 MSA、无模板；1,080/1,080 槽验证通过，零失败批次（model1 45/45 批，model2 45/45 批）。
- 实体组成：β（注入候选 CDR3-β）+ α + 肽；MHC 不入实体（类似物迁移声明，与抗体配方同为“候选链+伙伴链+抗原”三链结构）。
- 差异日志（写作时按方法学调度附注披露）：`results/pae_analog_deviation_log/` DEV-001（准入利用率阈值 20%→60%）、DEV-002（显存守护线 18.5→19.5 GB，耗尽批次跳过续扫）、DEV-003（会话中断一周后续跑，无数据损失）。均为调度层，不触碰任何冻结科学输入、估计量或分析规则。

## 队列与拆分（冻结数字）

v23：TCR 266/272 通过输入审计；v24：265 条通过链级身份编号（ANARCII TCR 模型 IMGT，CDR3-β 中位 12 aa）；v25：265 条全部暴露干净（对 4,749 条参考联合记录）；v26：856/4,567 组合通过坐标门控（TCR 849 + VHH 7），主淘汰为 4.5 Å 直接接触门（3,710）；v27：去重后确认性总体 170 个 TCR 组件，评估 6 组件/6 家族、训练 12 组件/10 家族，家族零重叠。VHH 仅 1 个干净组件，记录为描述性试点，不参与任何确认性主张。

## 主结果（冻结，来自 evaluation.json）

| 方法 | 组件等权召回（20% 预算，3/14） | 每家族值 |
|---|---:|---|
| Ridge 配方（全特征） | **50.0%** | 1/3 ×3，2/3 ×3 |
| 仅组成消融 | 50.0% | 与全特征逐家族一致 |
| ProteinMPNN NLL（符号训练侧固定） | 33.3% | 0, 0, 2/3, 1/3, 2/3, 1/3 |
| 训练均值 | 21.4% | 随机期望 |
| 随机（精确平局期望） | 21.4% | — |

配对组件 bootstrap（10,000 次，种子 20260918）：均值差 +0.2857，CI [0.1746, 0.3968]。正组件 6/6。

## 边界（必须保留）

- **仅组成驱动**：native 与组成打乱对照在教师层不可区分（臂差 0.0004–0.013，重复 SD 0.005–0.045；打乱在 5/6 家族不劣于 native）。单序列 MSA、固定骨架下 AF2-Multimer PAE 目标由组成主导，native 序列顺序无可检测的教师偏好。**不得**写成序列特异性识别、结合、亲和力或实验富集。
- 队列小（6 组件/6 家族），召回量化粗（每家族 3 个参考正例），bootstrap 为描述性，无多重比较显著性主张。
- 类似物迁移不是抗体验证；pMHC 肽两端受 MHC 槽约束，界面几何与自由表位不同；MHC 未入 AF2 实体是有意的配方镜像。
- 无 wall-clock 加速主张。
- 组件级 PDB 由坐标手写器生成（连续重编号）；受体链 MSE 记作 M，肽段严格标准残基。

## 证据索引

- 预注册：`docs/PAE_ANALOG_PREREGISTRATION_V27.md`；机器可读 `results/pae_analog_preregistration_v27/preregistration.json`。
- 门控链：`results/pae_analog_audit_v23/`、`pae_analog_chain_identity_v24/`、`pae_analog_overlap_v25/`、`pae_analog_structures_v26/`、`pae_analog_family_split_v27/`。
- 生成与教师：`results/pae_analog_generation_v28/`（1,080 槽作业清单）、`results/pae_analog_teacher_v29/`（全部批次与验证状态）。
- 评估：`results/pae_analog_evaluation_v30/evaluation.json`、`summary.json`、`predictions.json`、`geometry.json`。
- 手稿：`publication/pae_analog_manuscript_draft_v1.md`（PLOS Computational Biology 路由：`publication/pae_journal_routing_decision_v1.json`）。
