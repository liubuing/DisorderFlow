# DisorderFlow PAE 分支四项整改结果

本文仅评估 PAE 代理排序分支。当前主论文和发布主线是 ECLS，入口为 `publication/MANUSCRIPT_DRAFT.md`；PAE 的修订结论不替代 ECLS 已冻结的最终评估。统一定位见 `docs/PUBLICATION_MAP.md`。

结论：已完成证据修正、同目标基线与三种模块消融、筛选收益实测，以及新外部数据的可行性审计。**尚未补成可支撑正刊创新性的新证据：完整模型没有稳定优于简单基线，新外部独立队列为 0。不能据此宣称已经达到 Bioinformatics 投稿标准。**

| 工作项 | 实际完成 | 结果与边界 |
|---|---|---|
| 创新性/基线 | Ridge、Extra Trees、同目标 ProteinMPNN；完整模型及三种消融各 3 种子，共 12 次训练 | 完整模型 rho 0.638–0.824；Ridge 0.839；去抗原上下文 0.850–0.867。复杂模块优势未证实 |
| 外部覆盖 | 检索 45 个新发布结构，处理 5 个候选，2 个结构合格，执行五轴同源隔离 | 2 个均与参考集同源，新增独立组件 0；未伪装为盲测，也未降低隔离门槛 |
| 筛选收益 | 10%/20%/50% 预算，实际取 2/3/7 个候选；并实测原始 PDB 到评分耗时 | 取 3/14 时保留最优 3 个的 61.1–77.8%，少运行 78.6% 的候选 AF2 评估；这是计算代理召回率 |
| 论文纠错 | 新 v4 稿、逐项纠错账本、自动生成结果表 | 修正数量、方向、目标不一致、输入来源、阈值、手填表、成本、测试集状态与拒判含义 |

额外发现的关键问题：v3 数据构造读取 AF2 输出 PDB，因此它的结果不能直接支持运行 AF2 前的筛选。v4 已另建只使用原始设计骨架和候选序列的流程；AF2 矩阵只作为监督标签。

| Method | Median scaffold rho | Top-20% recall at 3/14 budget | Top-20% recall at 7/14 budget |
|---|---:|---:|---:|
| Ridge | 0.839 | 0.722 | 0.944 |
| Extra Trees | 0.822 | 0.722 | 0.944 |
| ProteinMPNN NLL | -0.738 | 0.000 | 0.278 |
| full, seed 2041 | 0.824 | 0.778 | 0.944 |
| full, seed 2053 | 0.822 | 0.722 | 0.972 |
| full, seed 2069 | 0.638 | 0.611 | 0.889 |
| no_antigen_context, seed 2041 | 0.850 | 0.722 | 0.944 |
| no_antigen_context, seed 2053 | 0.859 | 0.778 | 1.000 |
| no_antigen_context, seed 2069 | 0.867 | 0.722 | 0.889 |
| no_noise_weight, seed 2041 | 0.829 | 0.722 | 0.944 |
| no_noise_weight, seed 2053 | 0.807 | 0.778 | 0.972 |
| no_noise_weight, seed 2069 | 0.749 | 0.667 | 0.944 |
| no_grouped_difference, seed 2041 | 0.813 | 0.722 | 0.889 |
| no_grouped_difference, seed 2053 | 0.788 | 0.667 | 1.000 |
| no_grouped_difference, seed 2069 | 0.728 | 0.611 | 0.944 |

新训练使用 15 个训练组件的 240 个实体、2 个校准组件的 32 个实体；描述性外部评估为 6 个组件的 84 个实体。已有 GP2 最终测试集完全没有在 v4 中重新使用。外部旧数据已被查看，不能称为新的确认性验证。

完整模型的原始 PDB 到评分实测为每候选 0.0190–0.0280 秒（每骨架 14 个候选分摊；包含解析、特征与评分，不含模型载入）。未对旧 AF2 耗时做同硬件复测，因此撤回“2900–4900 倍”端到端加速主张。

另一个结构性限制：新标量分支对 H3 残基嵌入取均值，固定骨架上的组成相同、顺序不同序列得到几乎相同的分数。18 个原生/打乱对照的最大评分差为 6e-08。这不能被解释为具备序列顺序特异性的抗原识别能力；该项为事后结构诊断。

配对组件 bootstrap 区间和噪声阈值敏感性见 `experiment_report.json`；这些只作描述，不作多重比较后的显著性声明。消融针对新 v4 标量预测头，不冒充原 v3 联合训练的消融。

下一步科研条件很明确：取得真正独立、跨抗原家族的数据，并证明方法稳定优于 Ridge 等简单监督基线；否则应按基准/方法边界研究重新定位，而不是继续强化复杂模型优越性的叙述。此次没有制造正结果，也没有发布或替换上传包。

产物：

- `publication/af2_interface_pae_surrogate_manuscript_v4.md`：当前修订稿。
- `results/pae_surrogate_revision_v4/corrections.json`：原稿错误与更正依据。
- `results/pae_surrogate_revision_v4/experiment_report.json`：基线、消融、区间和筛选收益。
- `results/pae_surrogate_revision_v4/timing.json`：实测耗时。
- `data/pae_surrogate_external_v4/isolation_audit.json`：新增外部队列隔离结果。
