# 类似物队列泛化验证预注册 v27（TCR-pMHC / VHH-肽）

状态：**已冻结（2026-09-30）**。本文件在任何 candidate generation、任何 AF2 教师评分、任何模型拟合之前冻结。冻结后不得修改；偏差必须记录在 `results/pae_analog_deviation_log/` 并在报告中披露。机器可读状态：`results/pae_analog_preregistration_v27/preregistration.json`（含本文件 sha256）。

上游门控链（全部冻结、全部先行于本文件；已确认数字为 2026-09-30 执行结果）：

1. `data/pae_analog_scope_v22/`（HARVEST_RECEIPT.json 校验）：TCR 272 条 / VHH 52 条元数据收割。
2. v23 `results/pae_analog_audit_v23/`：离线输入完整性/化学审计。TCR 266 条、VHH 38 条通过输入检查。
3. v24 `results/pae_analog_chain_identity_v24/`：链级身份编号（ANARCII 2.0.8，TCR 模型 IMGT / 抗体模型 Chothia）。**TCR 265 条**（≥1 条编号成功的 β 链且 CDR3-β 可提取，中位 CDR3-β 长 12 aa，β 链评分中位 32.0）、**VHH 24 条**（编号为 H 且无 K/L 轻链）通过。
4. v25 `results/pae_analog_overlap_v25/`：对 v3 参考联合清单（4,749 条记录 + 576 个精确暴露 PDB）的回顾性暴露审查。命中判定：精确 PDB、序列精确/包含、4-mer 种子引导的 80% 双向覆盖 best_overlap（抗原轴 0.3、受体轴 0.9）。结果：**TCR 门控通过条目全部暴露干净（265/265）**；**VHH 仅 3/24 干净**（11 精确暴露、7 仅联合清单、3 受体序列命中，其中 5HM1 的 gp41 肽 AVERYLKDQQLLGIW 为精确命中）。
5. v26 `results/pae_analog_structures_v26/`：坐标门控。289 个条目全部解析；4,567 个 (α,β,肽)/(VHH,肽) 组合中 **856 个通过全部坐标门控（TCR 849 + VHH 7）**，写入规范组件 PDB。93 个条目无可解析组合；组合级门控失败：接触门 3,710、肽段骨架不完整 33。
6. v27 `results/pae_analog_family_split_v27/`：去重后 173 个组件；暴露排除 2 个；**确认性总体 171 个（TCR 170 + VHH 1）**。确认性拆分仅限 TCR（VHH 唯一干净组件记为描述性试点）。**评估侧 6 组件 / 6 肽段家族**（1lp9、3dxa、5brz、5bs0、6mtm、9k2i），**训练侧 12 组件 / 10 家族**，家族零重叠。最小规模（≥6 组件、≥3 家族）满足。

## 队列定义与排除规则（冻结）

- **确认性总体**：v26 全门控通过组件，且其所属条目在 v25 中无任何暴露命中（`pdb_exact_exposed`、`pdb_in_reference_union`、`receptor_axis_flags` 均为否）；仅限 TCR 队列。
- **TCR 队列**：v24 通过 265 条在 v25 全部暴露干净；确认性组件 170 个。
- **VHH 队列**：v24 通过 24 条中仅 3 条暴露干净，v26 坐标门控后仅 1 个干净组件——不满足任何最小规模，**降级为记录性试点**，不做任何确认性主张。

## 冻结的估算对象（estimand）

教师目标：AF2-Multimer v3（models 1 与 2，种子 7103/7111/7121，recycles 3，单序列 MSA、无模板），候选区（TCR：CDR3-β IMGT 105–117；VHH：Chothia 93–102 CDR3）到全部肽段位置的**方向性平均归一化 PAE**，六个矩阵（2 模型 × 3 种子）平均，越低越好。序列索引来自 v26 冻结的 `candidate_region.indices_zero_based`，不使用任何 AF2 依赖的补丁选择。

实体组成（镜像抗体配方：候选链 + 伙伴链 + 抗原）：

- TCR：α + β（注入候选 CDR3-β 序列）+ 肽段，三链。**MHC 不进入 AF2 实体**——这与生物学语境不同（pMHC 肽两端受 MHC 槽约束），属于明确的类似物迁移，任何写作必须声明。
- VHH：VHH（注入候选 CDR3）+ 肽段，两链。

## 候选生成（冻结）

ProteinMPNN（可溶权重，温度 0.3，`num_seq_per_target` 按下表，种子 20260919），固定骨架 = v26 冻结 PDB 的受体链（TCR 仅 B 链；VHH 仅 V 链），候选区可变、其余全部固定（fixed_positions_jsonl）。候选序列注入受体序列的候选区索引。对照实体：每评估组件 2 个（native、组成打乱，打乱种子 7307，要求打乱后非天然）。

| 组 | 组件数 | 每组件候选 | 每组件对照 | 每候选重复 | AF2 预测槽 |
|---|---:|---:|---:|---:|---:|
| 评估（确认性，TCR） | 6 | 14 | 2（native、组成打乱） | 6 | 6×16×6 = 576 |
| 训练（配方拟合，TCR） | 12 | 7 | 0 | 6 | 12×7×6 = 504 |

总计 1,080 个 AF2 预测槽，分 90 个守护批次（每批 12 槽）。

## 拆分（v27 冻结）

- 肽段家族：v25 方法的 4-mer 种子引导 best_overlap，identity ≥0.3、双向覆盖 ≥0.8，连通分量为家族。偏保守（把相似肽并族）。
- 家族不相交拆分：评估家族与训练家族零重叠。评估侧选 ≥6 组件 ≥3 家族；其余为训练侧。
- 组件去重：同一 (PDB, 受体实体, 肽段实体) 多 auth 链拷贝只保留一个代表实例。

## 模型（同配方重训，冻结）

- 主模型：Ridge（alpha=10，固定，无超参搜索、无校准选择），特征 = 候选区氨基酸组成（20 维）+ 肽段氨基酸组成（20 维）+ 长度特征（候选区、肽段）+ 设计骨架几何特征（沿用 v4/v7 定义的骨架距离特征，由 v26 PDB 计算）。标准化拟合于训练侧。
- 预注册对照：ProteinMPNN 生成时 NLL（排序方向以训练侧符号确定，不得用评估侧选符号）、随机期望、仅候选区组成 Ridge 消融、仅几何消融。
- 不使用冻结抗体模型做任何评分（V22 边界）。

## 主要终点与统计（冻结）

- 主终点：20% 预算（14 取 3，ceil）下教师最优 20% 召回，组件等权平均；随机期望 21.4%（3/14 精确随机-平局期望另报）。
- 统计：配对组件 bootstrap 10,000 次（种子 20260918）对随机期望；报告家族级正组件计数；候选行与 AF2 重复不作为独立生物学样本。
- 无多重比较显著性主张；无加速倍数主张（不计端到端 wall-clock 优势）；无结合/亲和力/实验收益主张。
- 失败与零结果、以及全部对照行为，必须保留在报告中。

## 完成边界

本预注册只冻结设计。科学文本与本地结果不等于期刊投稿或 DOI。任何对本文件的偏离在执行前不可发生；执行中偏差写入 deviation log。类似物迁移不是抗体验证，也不是 TCR 生物学主张。

## 运行手册（冻结后按序执行）

```
venv/Scripts/python.exe scripts/split_pae_analog_families_v27.py        # v27 拆分冻结
venv/Scripts/python.exe scripts/generate_pae_analog_candidates_v28.py   # MPNN 候选 + 作业清单
venv/Scripts/python.exe scripts/run_pae_analog_teacher_v29.py prepare   # Windows 侧分批冻结
wsl.exe -d Ubuntu-24.04-D -- bash -c 'cd /mnt/d/biological/DisorderFlow &&
  venv_wsl/bin/python scripts/run_pae_analog_teacher_v29.py run'        # WSL 侧守护运行（长时）
venv/Scripts/python.exe scripts/evaluate_pae_analog_v30.py              # 教师提取 + 重训 + 评估
```

（冻结时回填：v26 组件计数、v27 拆分明细、本文件 sha256。）
