# DisorderFlow 仓库只读审计 —— 资产、已验证结果与阻塞项

- 审计日期：2026-11-01
- 审计对象：`D:\biological\DisorderFlow`，git 分支 `release`，HEAD `13160681`（2026-10-29 10:30:12 +0800，`feat: add PAE analog transfer campaign v24-v30`）
- 审计方式：只读。未运行训练/推理/AF2/BFN 前向；未执行 `pytest`（因此本文所有"测试通过"字样均转述仓库记录，不是本次实测）；未改动 git 状态；除本文件外未写任何仓库文件。
- 唯一写入：`.dsh-research/flex-distance/repo-audit.md`（本文）。
- 依据格式：`文件:行号`。JSON 用 `文件:行号（键名）`。凡本文自行由冻结 JSON 重算的数字，均标注 **【本审计重算】**。

---

## 0. 一句话定位与线层级关系

**一句话**：DisorderFlow 是一个"抗体–柔性肽/CDR-H3 计算评分 + BFN 序列设计实验"的科研仓库，**当前唯一主稿是已冻结的 ECLS v1**（一个 ProteinMPNN 似然差指标，只在"native vs 组成打乱"上为正），目标 Bioinformatics；其余各线或为独立分支、或已判负、或为 2026-10-29 才开的新战役。

依据：`README.md:3-6`（主稿与发布状态）、`README.md:19-22`（冻结主张仅限 native-vs-shuffle）、`publication/ECLS_SCOPE_FREEZE.yml:2-3`（`frozen_at: 2026-08-09`）、`PUBLICATION_PROTOCOL.md:9`（目标 Bioinformatics Original Paper）、`docs/PUBLICATION_MAP.md:16`（"No journal submission is asserted"）。

### 层级关系（谁是谁的上游/下游）

```
             ┌─ 主稿：ECLS v1（已冻结，唯一正向主推断）── Bioinformatics，作者审阅中，未投稿
             │    证据：31 结构 / 15 抗原簇，均值 0.172281
             │
并联、互不互证 ─┼─ PAE 分支（独立改稿 v20 + 类似物迁移 v30）── 已改投 PLOS Comp Biol
             │    历史：v3 已被否，v4 纠错为强制上下文；v21 数据墙"新增合格家族 = 0"
             │
             ├─ 已判负的历史线：T2.1 接触机制、multiscaffold 从头设计、通用跨生成器 reranking、
             │    ECLS+PAE 等权融合、AAYL 亲和力排序
             │
             ├─ disorder 头（独立预测任务，非设计能力）── 外部可判别但劣于 metapredict、未校准
             │
             └─ FLEX-BINDER 战役 v1（2026-10-29 新开）── 明确声明"不接 PAE v31 编号"
                   Phase A 数据门（通过）→ Phase B J0a/J0b 裁判门（通过）→ Phase C1 排序门（**失败**）
                   → 按 M8 条款重定位为"边界研究"；边界手稿草案已立（本地、未冻结）
                   → Phase D（binder-vs-nonbinder）仅草案，被可行性门阻塞
```

| 线 | 层级 | 状态一句话 | 关键依据 |
|---|---|---|---|
| ECLS v1 | 主稿 | 已冻结、终局、禁止重跑；作者声明与投稿均未完成 | `publication/ECLS_SCOPE_FREEZE.yml:2-17`；`PUBLICATION_PROTOCOL.md:21-23` |
| PAE 单一轴置信度 | 冻结部署资产（不是主稿） | 三种子部署合同已冻结，只承担"跨骨架相对排序" | `publication/candidate_interface_pae_deployment_v1.json`（`status=frozen_pae_single_axis_deployment`） |
| PAE 回溯筛选 v20 | 独立分支 | 回溯效用有，独立家族数据墙 | `docs/PAE_EVIDENCE_AUDIT_V20.md:14-21` |
| PAE 类似物迁移 v30 | 独立分支（正向但温和） | 组成驱动召回收敛，序列顺序无教师偏好 | `docs/PAE_ANALOG_TRANSFER_V30.md:7,33` |
| T2.1 接触恢复 | 已判负 | 接触特异性机制被拒，终局 | `publication/t21_mechanism_terminal_decision.json`（`mechanism_supported=false`） |
| multiscaffold 从头设计 | 已判负 | 2400 次尝试全部过门失败 | `publication/idp_platform_capability_v1.json`（`key_negative_results[0]`） |
| FLEX-BINDER v1 | 新线（当前活跃） | A/B 通过，C1 失败，转边界研究 | `docs/FLEX_BINDER_CAMPAIGN_V1.md:1-5,80-82,295-341` |
| Phase D | 新线的下一站 | 草案未冻结，被数据可行性阻塞 | `docs/FLEX_PHASE_D_BINDER_TASK_PREREG_DRAFT_V1.md:1-15,84-96` |
| 边界研究手稿 | FLEX 的产物 | 本地 v1 草案，未冻结、未投稿 | `publication/flex_boundary_manuscript_draft_v1.md:3-11` |

---

## 1. 能力清单表

> 口径：本表只写"仓库里存在什么 + 已知指标从哪来"。存在≠能力，指标≠实测（见 §6）。
> 权威能力自评表在 `publication/idp_platform_capability_v1.json`（`classification=development_only`，`purpose` 明确要求"tool existence is never conflated with design effectiveness"）——本文与它冲突时以文件为准。

### 1.1 模型 / checkpoint

| # | 名称 | 路径（相对仓库根） | 体量 | 训练数据规模 | 已知指标 | sha256 记录位置 |
|---|---|---|---|---|---|---|
| C1 | **BFN disorder v4 s2032（当前主力）** | `logs/bfn_disorder_v4_afdb_supervised_2026_08_05__17_22_45_disorder_v4_balanced_long_s2032/checkpoints/best.pt` | 68,508,625 B | AFDB 监督集：828 训练结构 / 799 UniRef50 簇 + 97 验证结构 / 93 簇；59,698 实验监督残基；另 76,164 高置信 AFDB 残基作 0.25 权重代理 | dev ROC-AUC 0.8974 / 0.8929 / 0.8975（均值 0.8959）；CAID2 回归 0.7435 / 0.7417 / 0.7400；CAID3 macro ROC-AUC **0.7551**（CI [0.6945, 0.8119]）vs metapredict **0.7927**，配对差 −0.0376（CI [−0.0786, 0.0056]）；Brier 0.2991 / ECE 0.3794 **未过校准门** | `results/flex_j0_v1/j0a_summary.json:55`（`ed9984504d…`）；本次审计独立复算该文件 sha256 = `ed9984504d799187b7660c1d40c3dc68c01a0ce9b3cf92e25af1273195931252`，**与记录一致**。指标见 `docs/DISORDER_SUPERVISION_PROTOCOL.md:47-55,85-88,123-127` |
| C2 | **PAE 单轴置信度三种子（冻结部署）** | `logs/bfn_candidate_interface_multiscaffold_dual_sem_v2_2026_09_01__12_13_52_dual_sem_v2_s2041/checkpoints/1000.pt`（s2041）<br>`…_12_29_06_dual_sem_v2_s2053/checkpoints/800.pt`（s2053）<br>`…_21_10_10_dual_sem_v2_s2069_retry/checkpoints/800.pt`（s2069） | 46,019,512 B / 46,019,073 B / 46,019,073 B | 候选-界面置信度头；校准集 6 个独立外部骨架 / 84 条实体记录 | corrected MAE 0.0245 / 0.0223 / …；median scaffold Spearman 0.866 / 0.725 / …；pairwise 可靠对 236 / 6 骨架；pair accuracy 0.953 / 0.886 / … | `publication/candidate_interface_pae_deployment_v1.json`（`seeds[*].checkpoint_sha256`）。**本次审计逐文件复算 3/3 全部哈希匹配** |
| C3 | **v5.1 四级训练产物（stage4）** | `logs/v5_1_pipeline/stage4_idp_conditioned/bfn_v5_1_stage4_idp_conditioned_2026_07_21__02_02_51_v5_1/checkpoints/best.pt` | 123,855,881 B | v5.1 冻结 LMDB 1,289 条（867/217/205）；Phase3 2,952/328 对；disorder profile 覆盖 717/82 | **无任何下游评测记录**；不在 `configs/CHECKPOINT_TRUTH.md`；没有任何 config 指向它 | `logs/v5_1_pipeline/state/final_checkpoint.txt`（路径）；`…/state/readiness.json`（数据面） |
| C4 | 历史 v6_phase2 置信度头 | `logs/bfn_confidence_combined_v6_phase2_2026_05_18__14_05_03/` | — | — | 曾是 `demo_design.yml` 默认（2026-05-18 ~ 2026-09-27），已让位 | `configs/CHECKPOINT_TRUTH.md:25` |
| C5 | 公开 AntibodyDesignBFN `best.pt`（**非本项目 checkpoint**） | 外部（HuggingFace `YueHuLab/AntibodyDesignBFN`） | 112,993,226 B | — | 仅可在架构/state_dict 兼容检查后使用 | `DATABASE_SOURCES.md:50-52`（sha256 `f3e25477…`） |

补充事实：
- 主力指向已核对：`app_config.yaml:151` 与 `configs/demo_design.yml:4` 都指向 C1，与 `configs/CHECKPOINT_TRUTH.md:10` 一致；`configs/demo_design.yml:3` 保留了原值注释（符合该表 §规则1）。
- 但 **CHECKPOINT_TRUTH 这张"唯一权威索引"漏掉了整条 C3（v5.1 四级模型）与阶段 checkpoint**（`configs/CHECKPOINT_TRUTH.md` 全文 32 行，只列 C1、C2、C4 与"其余 bfn_* 时间戳目录"）。这是一个文档—资产不一致。
- `logs/` 下共有 96 个 `bfn_*` 时间戳目录（本次 `ls -1d logs/bfn_*` 计数），无统一 registry。

### 1.2 生成器

| 生成器 | 代码位置 | 版本/参数（有据可查的） | 已知表现 | 边界 |
|---|---|---|---|---|
| **BFN（自家，Bayesian Flow Network / Geometric Transformer）** | `modules/bfn_loader.py:301`（`run_bfn_design`，文件 449 行）；模型定义 `disorderflow/models/bfn_model.py`（191 行） | 固定骨架条件序列生成；disorder 条件开关见 `idp_platform_capability_v1.json`（`disorder_guided / epitope_disorder_profile`） | 4HIX 匹配诊断：native recovery **0.080**（disorder on）/ **0.153**（off）/ 0.063（parent BFN），mean candidate-median ipTM 0.456 / 0.445 / 0.452 | 自评 `deployment_grade_for_fixed_backbone_design`，但同一条目写"Multiscaffold antibody de novo design was a terminal negative internal benchmark"；"更高多样性/更低困惑度"未独立验证（`idp_platform_capability_v1.json` `vision_components[0]`） |
| **ProteinMPNN（第三方，本地 vendored）** | `ProteinMPNN/protein_mpnn_run.py`；vanilla 权重 `ProteinMPNN/vanilla_model_weights`；soluble 权重 `ProteinMPNN/soluble_model_weights` | **v_48_020**（`configs/benchmarks/peptide_h3_ecls_temporal_final_v1.yml:9-14`）；温度随任务变：C1 = 0.2（`publication/flex_boundary_manuscript_draft_v1.md:116-118`）、PAE 类似物 = 0.3（`publication/pae_analog_manuscript_draft_v1.md:61`）、旧合成规格 = 0.5（`SYNTHESIS_SPECS.md:68`） | 4HIX 匹配诊断 native recovery **0.326**、developability pass **1.000**；C1 里 1,300 个设计 / **789** 个不同 / 0 个等于 native（`results/flex_c1_v1/generation_summary.json:2-5`） | 它是当前 FLEX 线外包生成器（`docs/FLEX_BINDER_CAMPAIGN_V1.md:22` M8 条款），生成≠结合 |
| 其他第三方 | `diffab/`、`HDOCKlite-v1.1/`、`esm_if`（`modules/esmif_compat.py`） | — | 仓库内无独立验证记录（本次未深入） | — |

### 1.3 打分器（"裁判"）

**关键发现：仓库里叫 "ECLS" 的东西有两套不同底模，不是同一件事。**

| 打分器 | 底模 | 定义 | 实现位置 | 已知表现 |
|---|---|---|---|---|
| **ECLS v1（冻结主稿用）** | **ProteinMPNN v_48_020**（backbone-only 条件概率，seed 2309） | H3 NLL(complex backbone) − H3 NLL(peptide-stripped backbone)；advantage = mean(ECLS(组成匹配打乱)) − ECLS(native) | `scripts/benchmark_h3_epitope_delta.py:1-2`（docstring："peptide-conditioned ProteinMPNN H3 likelihood"）、`:102-120`（调 `protein_mpnn_run`）、`:241-300`（冻结契约校验）；统计 `scripts/analyze_h3_ecls_statistics.py`；配置 `configs/benchmarks/peptide_h3_ecls_temporal_final_v1.yml` | 31 结构 / 15 簇：mean 0.172281，median 0.118699，CI [0.059156, 0.291920]，12/15 簇为正（`results/publication/h3_ecls_temporal_final_v1/final_decision.json:14-21`） |
| **flex_ecls_marginal（FLEX J0/C1 用）** | **BFN s2032（C1）** | 同一公式，但 NLL 换成 BFN 的 `losses['seq']`，并对 4 个受限 OpenMM 构象做边际平均 | `scripts/flex_j0b_score_v1.py:66`（`marginal_ecls`）、`scripts/flex_c1_score_and_rank_v1.py:216-220`（score 复用 J0a/J0b 缓存） | J0a +0.124（CI [0.051, 0.189]，7/9 族）；J0b +0.141（CI [0.069, 0.204]，7/8 族）；C1 native 平均百分位 0.767（CI [0.676, 0.809]，Top-1 = 0） |
| PAE 单轴 | C2 三模型均值 | 归一化 candidate→antigen PAE | `publication/candidate_interface_pae_deployment_v1.json` | 见 §1.1 C2 |
| 组成基线（composition logistic / Ridge） | 无结构信息 | 20 维氨基酸组成（C1 用 LOFO 逻辑回归；PAE 用 Ridge α=10） | `scripts/flex_c1_score_and_rank_v1.py`（`composition_logistic` 分支）；`publication/pae_analog_manuscript_draft_v1.md:65` | C1：native 平均百分位 **0.544**（竟与自家裁判 0.767 同量级，且 5/8 族胜过裁判）；PAE v30：召回 50.0%，与全特征**完全一致** |
| disorder 头 | C1 同族 BFN 头 | `predict_disorder` + 校准 | `modules/idp_disorder_analysis.py`；`calibration_artifacts/disorder_v4_balanced_s2032.json` | CAID3 0.7551 < metapredict 0.7927；未外校准 |

> 结论：**"ECLS 已被验证"这句话只在 ProteinMPNN 底模 + native-vs-shuffle 任务上成立。** FLEX 复现的是"同一个似然差公式在 BFN 底模 + 单域骨架 + 新队列上的同类信号"，据 `docs/FLEX_BINDER_CAMPAIGN_V1.md:258-259` 自述"是 ECLS 信号在…新骨架类上的开发级复制，不是对 ECLS 冻结结果的确认"。二者不可互相引用为独立验证。

### 1.4 数据结构（规模、家族、阴性池、标签源）

| 资产 | 规模 | 家族/单元 | 标签源 | 依据 |
|---|---|---|---|---|
| ECLS 冻结时间面板 | **31 结构** | **15 个 official antigen cluster** | 沉积序列 + 组成匹配打乱（200 trials，bootstrap 10,000） | `results/publication/h3_ecls_temporal_final_v1/final_decision.json:14-21`；`configs/benchmarks/peptide_h3_ecls_temporal_final_v1.yml:14-22` |
| ECLS 适配集（开发出处） | **46 复合物 / 46 推理单元** | 46 | 同上 | `results/publication/h3_ecls_adaptation_v1/results.json`（`aggregate.n_complexes=46`、`mean_ecls_advantage=0.217481`、`fraction_positive=0.804348`） |
| FLEX 候选池 | SAbDab 快照 **21,142 行** → 无轻链×肽 **132 行 / 57 唯一 PDB**（上界） | — | 摘要行级粗筛（未核链身份） | `results/flex_campaign_v1_scaffold_scan.json:5-12`；`docs/FLEX_BINDER_CAMPAIGN_V1.md:33-48` |
| FLEX A1 合格集 | 57 PDB → 340 组件 → **107 组件通过** → **36 PDB**；32 条互异重链，跨 PDB 零重复；101 X-ray / 6 EM | — | 坐标/身份/接触门（ANARCII Chothia，CDR-H3 93–102，4.5 Å） | `results/flex_a1_scaffold_audit_v1/summary.json:2-7`；`docs/FLEX_BINDER_CAMPAIGN_V1.md:98-109` |
| FLEX A1b 开发队列 | 36 → 同 PDB 排除 6 → 隔离干净 **30** → 暴露洁净 **14 PDB** | 18 个肽家族（30 PDB 上）/ **14-PDB 队列跨 9 个抗原家族**；27 条重链谱系 | 家族门：肽 ≥50% 同一性/80% 覆盖；重链克隆级 ≥0.90 | `results/flex_a1b_isolation_families_v2/summary.json`（`clean_pdbs=30`、`peptide_families=18`、`heavy_lineages=27`）；`docs/FLEX_BINDER_CAMPAIGN_V1.md:191-204` |
| FLEX J0 队列 | **28 组件 / 14 PDB**；每组件 **50 个组成匹配打乱** | 9 族 | 沉积重链 + 肽（单姿态） | `results/flex_j0_v1/j0a_summary.json:2-5`；`results/flex_j0_v1/preregistration.json`（`judge_primary`/`shuffle_protocol`） |
| FLEX J0b 队列 | **26 个有效 ensemble 组件**（2 个按预注册剔除） | 8 族 | T1 协议 4 构象，≥2 接受构象且平均两两 RMSD ≥0.25 Å | `results/flex_j0b_v1/j0b_summary.json:3-7` |
| FLEX C1 任务 | **26 组件 × 101 候选池**（native + 50 MPNN + 50 打乱）；实生成 **1,300**，去重 **789** | 8 族 | — | `results/flex_c1_v1/generation_summary.json:2-5` |
| Phase D 阴性池 | IEDB B-cell 阴性线性 5–50aa：**1,234,987 条记录 → 709,890 独立序列 → 672,336 个不同组成**（平均每组成 1.056 条） | 30,388 个不同抗原带阴性测量；200/200 结构桥抗原带同抗原阴性（合计 20,799 条） | 公开 IEDB（元数据级，仅计数，未落表位序列） | `results/flex_phase_d_v1/negative_pool_summary.json:4-6`；`results/flex_phase_d_v1/feasibility_addendum_v3.json:12-14` |
| Phase D binder 侧 | SAbDab 肽抗原×亲和力注释 **249 行 / 138 PDB**；有方法标注 69（SPR 51、ITC 9、Other 9、Unknown 180）；与 A1 合格集交集 **1 个 PDB** | 106 个不同抗原名 | **未核验元数据**（不得直接引用数值） | `results/flex_a4_affinity_inventory_v1/summary.json:2-11`；`docs/FLEX_BINDER_CAMPAIGN_V1.md:147-160` |
| v5.1 构象库 | **1,289** 条（867/217/205）；源 LMDB 20 GB；构建停在 1,289（原目标 1,301） | — | `tanh(RMSF/2)`，AF2 5 种子 Cα RMSF（代理标签） | `data/confidence_conformation_v5/frozen_manifest_v5_1.json:4,10`；`HANDOFF_V5.md:10-15` |
| v5.1 NMR 校准 | 159 PDB → **129 个多模型 ensemble**（中位 20 模型、135 残基）；RMSF 均值 3.1949 Å | — | 沉积 NMR 模型（**不是 Boltzmann 加权样本**） | `results/flex_a2_nmr_rmsf_v1/summary.json`；`data/confidence_conformation_v5/rmsf_calibration.json:10-12` |
| Phase3 抗体-抗原对 | 2,952 / 328（train/val） | — | 接触/距离/FAPE/扭转 | `logs/v5_1_pipeline/state/readiness.json`（`phase3_counts`） |
| CAID2 外部集 | 348 靶，307 与现行 DisProt 重叠；剔除后 **3,009** 训练候选 | — | CAID2 盲外部（**不得并入训练**） | `DATABASE_SOURCES.md:7-11,28-31` |
| CAID3 外部集 | 204 靶 → 83 靶被评分（17 排除 + 3 无 UniRef50 归属），71 个 mixed-label 簇 | 71 | CAID3（一次性） | `docs/DISORDER_SUPERVISION_PROTOCOL.md:122-125`；`INTEGRATIVE_ANALYSIS.md:216-222` |
| SAbDab2 ML 归档 | `data/sabdab2_current/splits.tar.gz` **876,381,859 B**，MD5 `0dbb4cc499e9eb77f14008b232f2c38c` | — | Zenodo 20083995 / 0.1.0 / CC BY 4.0；用 `abag_split.csv` | `DATABASE_SOURCES.md:35-46`。**本次审计复算 MD5 一致**；`data/sabdab2_abag_project_split_v2/` 下已有 dev/adaptation LMDB |
| TCR-pMHC 队列（PAE v30） | 266/272 → 265 → 170 确认性组件；评估 **6 组件/6 族**，训练 12 组件/10 族 | 6 | AF2-Multimer v3 PAE 教师（模型 1+2，种子 7103/7111/7121，3 recycle，单序列 MSA） | `docs/PAE_ANALOG_TRANSFER_V30.md:11,17` |

---

## 2. 每条线：已冻结结论 / 边界 / 失败原因

> 只列能追到冻结文件或终局决定的结论。"失败原因"写仓库自己给出的原因；仓库未给出的，标"仓库未定位原因"。

### 2.1 ECLS v1（主稿，已冻结）

| 项 | 内容 | 依据 |
|---|---|---|
| 冻结结论 | 沉积 native H3 在 31 结构/15 抗原簇上，ECLS 优势均值 **0.172281**，median 0.118699，簇 bootstrap 95% CI **[0.059156, 0.291920]**，**12/15** 簇为正 | `publication/ECLS_SCOPE_FREEZE.yml:7-11`；`results/publication/h3_ecls_temporal_final_v1/final_decision.json:14-21` |
| 冻结门 | median ≥0.1 / 正簇比例 ≥0.7 / CI 下界 >0，三项全 pass | 同上，`final_decision.json:23-32` |
| 终局性 | `attempt=1`，`terminal=true`，`rerun_permitted=false` | `final_decision.json:5-7` |
| 边界 | 不是结合/亲和力预测，不是通用候选重排；打乱是对照反事实，**不是实验证实的非结合体** | `PUBLICATION_PROTOCOL.md:21,36`；`README.md:19-22` |
| 已知缺陷（仓库自述） | 七簇通用重排为负门（bfn mean gain −0.1548，CI [−0.4286, −0.0119]）；生成器校准仅探索性；时间独立性不等于训练独立性 | `results/publication/h3_candidate_reranking_dev_v1/results.json`（`aggregate.generators.bfn`）；`PUBLICATION_PROTOCOL.md:29-36` |

### 2.2 ECLS + PAE 等权融合（已判负）

| 项 | 内容 | 依据 |
|---|---|---|
| 结论 | 等权秩融合未过开发验证 | `docs/ECLS_PAE_JOINT_DEVELOPMENT.md:28` |
| 数字 | 6 组件 / 84 序列（主分析 72 条）：PAE 77.8%、Ridge 72.2%、MPNN-NLL **0.0%**、ECLS **11.1%**、融合 **40.3%**（主终点 3/12） | 同上 `:11,18-24` |
| 失败原因 | 融合比 PAE 低 **37.5 pp**（CI [−59.7, −16.7]）；ECLS 的 native/打乱区分力不能自动转为 AF2 代理排序力 | 同上 `:26,28` |
| 边界 | 该负结果同样不能推断真实结合能力（终点只是 AF2 教师） | 同上 `:28` |

### 2.3 PAE 回溯筛选 v20（独立分支）

| 项 | 内容 | 依据 |
|---|---|---|
| 结论 | 冻结 Ridge 在 20% 预算下类别等权召回 **60.4%**（随机 20.0%）；组成单独 58.3%；同池 ProteinMPNN H3 8.3%、骨架 6.2% | `publication/pae_analog_manuscript_draft_v1.md:25`；`docs/PAE_EVIDENCE_AUDIT_V20.md:16-18` |
| 边界 | 回溯池与开发数据重叠；机制消融显示"同骨架候选排序由 H3 组成驱动"，不得宣称几何增强 | `docs/PAE_EVIDENCE_AUDIT_V20.md:18` |
| 数据墙 | v21：两轮补查后**新增合格抗体候选数 = 0**；公开配对抗体-短肽空间已耗尽（VHH 口径只剩 3 个暴露干净条目 / 1 个可确认组件） | `docs/PAE_NEW_FAMILY_FEASIBILITY_V21.md`（"确认新增合格抗体候选数为0"节）；`publication/pae_analog_manuscript_draft_v1.md:29` |
| 失败/风险原因 | 版本链在 17 个版本后才撞上数据墙（M1 错误模式）；组成主导使序列顺序不可辨 | `docs/FLEX_BINDER_CAMPAIGN_V1.md:15`；`docs/PAE_ANALOG_TRANSFER_V30.md:33` |

### 2.4 PAE 类似物迁移 v30（独立分支，正向但温和）

| 项 | 内容 | 依据 |
|---|---|---|
| 冻结结论 | 6 族上教师 top-20% 召回 **50.0%**（随机精确平局期望 21.4%），配对差 **+28.6 pp**，CI [17.5, 39.7] pp，6/6 族不低于随机 | `docs/PAE_ANALOG_TRANSFER_V30.md:7,29` |
| 组成消融 | 仅组成 = 全特征 **50.0%**（逐族一致）；MPNN 生成时似然 33.3% | 同上 `:7,25` |
| 边界 | **仅组成驱动**；native 与打乱在教师层不可分（臂差 0.0004–0.013）；队列 6 组件；不是抗体验证 | 同上 `:33-37` |
| 路由 | 已改投 PLOS Computational Biology（2026-09-30） | `docs/PUBLICATION_MAP.md:26` |

### 2.5 T2.1 接触特异性恢复（已判负，终局）

| 项 | 内容 | 依据 |
|---|---|---|
| 结论 | 供体接触约束**没有**优于计数/角色/距离匹配的随机约束 | `publication/t21_mechanism_terminal_decision.json`（`mechanism_supported=false`，`terminal_for_current_mechanism=true`） |
| 数字 | 主对比 supplied−random = **−0.0366**（CI [−0.0638, −0.0108]），13 个推理单元；31 结构中 26 达扭转层、25 有效（96.2%）；supplied 0.581 / all 0.611 / random 0.619 / null 0.533 | `results_t2.1_v2_final.json`（`aggregate`）；`INTEGRATIVE_ANALYSIS.md:74-79` |
| 失败原因 | 随机约束反而更好 → 阳性臂内恢复不构成机制证据 | `INTEGRATIVE_ANALYSIS.md:81-85` |
| 边界 | 不得重跑现有 final；只允许"机制上不同的后继 + 新预注册 + 未暴露基准" | `publication/t21_mechanism_terminal_decision.json`（`decision`） |

### 2.6 multiscaffold 抗体从头设计（已判负）

| 项 | 内容 | 依据 |
|---|---|---|
| 结论 | **2400 次尝试，全部过门失败**，记为终局负基准 | `publication/idp_platform_capability_v1.json`（`key_negative_results[0]`） |
| 在 ECLS 合同中的地位 | "multiscaffold-v2 terminal negative design result" 被列入"仅作负结果/补充"，并明确 `antibody design success` 排除在主主张外 | `publication/ECLS_SCOPE_FREEZE.yml:22-32` |

### 2.7 disorder 头（独立预测任务）

| 项 | 内容 | 依据 |
|---|---|---|
| 历史失败 | charge-hydropathy 2,950 条 → AUC **0.509**；DisProt/MobiDB 143 条 → **0.500**（随机水平） | `INTEGRATIVE_ANALYSIS.md:205-212` |
| 后续开发结果 | balanced-v4 三种子 dev ROC-AUC 0.8076/0.8075/0.8115（300 步）与 0.8974/0.8929/0.8975（2,000 步） | `docs/DISORDER_SUPERVISION_PROTOCOL.md:71-75,83-88` |
| 冻结外部结果 | CAID3 macro ROC-AUC **0.7551**（CI [0.6945, 0.8119]）；metapredict **0.7927**；配对差 −0.0376（CI [−0.0786, 0.0056]）；Brier 0.2991 / ECE 0.3794 | 同上 `:122-127` |
| 边界 | 外部可判别，但**不优于 metapredict、未外校准**；"不产生任何 disorder-aware 设计主张" | `INTEGRATIVE_ANALYSIS.md:224-226` |
| 关键阻塞 | v5.1 的物理标签门**未通过**：全体中位 Spearman 0.217 < 0.30 门；中位偏 Spearman 0.14852 < 0.15 门；判定 **`rank_only`**（三子组也都是 rank_only） | `data/confidence_conformation_v5/rmsf_calibration.json:18-20,24-28,45` |

### 2.8 FLEX-BINDER Phase A（数据门，通过但含硬约束）

| 项 | 内容 | 依据 |
|---|---|---|
| A1 | 57 候选 → 36 PDB / 107 组件（≥12 停止规则**达标**） | `results/flex_a1_scaffold_audit_v1/summary.json:7`；`docs/FLEX_BINDER_CAMPAIGN_V1.md:93-111` |
| A1b | 逐级收紧后 **14 PDB / 9 抗原家族**（18 肽家族 / 27 重链谱系在 30 PDB 上） | `results/flex_a1b_isolation_families_v2/summary.json`；`docs/FLEX_BINDER_CAMPAIGN_V1.md:189-204` |
| A2 NMR 腿 | 129 ensemble，但 ATLAS 配对仅 **2/129**；改道 Chameleon/DPF 得 **0 对** → "MD vs NMR 配对路线的天花板即 2 对" | `docs/FLEX_BINDER_CAMPAIGN_V1.md:118-134,169-171` |
| A2 pLDDT 腿 | ATLAS 1,938/1,938 全审：中位 Spearman(原始 MD RMSF, −pLDDT) = **0.730**（IQR 0.615–0.808）；96.9% ≥0.30、87.5% ≥0.50 | `results/flex_a2_atlas_plddt_audit_v2/summary.json:4,13-14` |
| A4 | 249 行亲和力元数据 / 138 PDB；与 A1 合格集交集 **1** | `results/flex_a4_affinity_inventory_v1/summary.json:2-7` |
| 硬约束 | 9 个全新家族"不足以支撑独立确认终点"；确认性必须扩宇宙或降为描述性 | `docs/FLEX_BINDER_CAMPAIGN_V1.md:162-168,225-226` |
| 披露的过程 bug | A1 首跑全零假阴性（三字母残基名直比单字母集合），被 fail-closed 守卫拦下后修复 | 同上 `:114-116`；契约测试 `tests/test_flex_a1_scaffold_audit_contract.py:74-80` |

### 2.9 FLEX Phase B（J0a/J0b 裁判门，**通过但只是开发门**）

| 门 | 结论 | 数字 | 依据 |
|---|---|---|---|
| J0a 单姿态 | pass | 28 组件 / 9 族，7/9 族正（77.8%），均值 **+0.12419**，CI **[0.05132, 0.18949]**，25/28 组件正 | `results/flex_j0_v1/j0a_summary.json:2-9,51` |
| J0b ensemble | pass | 26 组件 / 8 族，7/8 族正（87.5%），均值 **+0.14094**，CI **[0.06924, 0.20386]**，24/26 组件正；两种边际化顺序 +0.141 / +0.138 | `results/flex_j0b_v1/j0b_summary.json:3-14,31` |
| 衰减预言 | **未复现**：J0b−J0a = **−0.0167**（T1 曾测得 ~0.20 衰减） | 同上 `:14`；`docs/FLEX_BINDER_CAMPAIGN_V1.md:279-283` |
| 边界（预注册保留） | 符号检验单独 p≈0.090（7/9），预注册功效只覆盖"≥8/9 强裁判"；门通过靠复合规则（70% + CI 排零） | `docs/FLEX_BINDER_CAMPAIGN_V1.md:252-257` |
| NLL 字段披露 | checkpoint 无 mismatch 权重 → `factual_nll_per_sample` 缺席 → 用 `losses['seq']`（单样本等价），已记录在 summary | `results/flex_j0_v1/j0a_summary.json`（`nll_field`）；`scripts/flex_j0_score_v1.py:9-12` |

### 2.10 FLEX Phase C1（排名门，**失败**——本线的转折点）

| 项 | 内容 | 依据 |
|---|---|---|
| 任务 | 26 组件 × 101 候选池里定位 deposited native（native + 50 MPNN + 50 打乱） | `results/flex_c1_v1/preregistration.json`（`task`）；`publication/flex_boundary_manuscript_draft_v1.md:113-118` |
| 冻结结果 | native 平均百分位：自家裁判 **0.767**（CI [0.676, 0.809]）、MPNN-NLL 0.637（CI [0.587, 0.668]）、组成逻辑回归 0.544（CI [0.545, 0.630]）；三法 Top-1 均 0/26 | `results/flex_c1_v1/c1_summary.json:5-29,67` |
| 门判定 | **fail**（`gate_parts` 三项全 false） | 同上 `:63-67` |
| 方向修正后 | 赢 MPNN-NLL **6/8**（条款通过）；赢组成基线 **5/8**（<70%，仍失败）；`verdict_robust_to_direction=true` | `results/flex_c1_v1/direction_correction.json:11-14,28-31,45-46` |
| 失败原因（仓库结论） | **连 20 维组成都赢不了 → "超出组成的信息"主张不成立**；这是 M8 失败的准确位置 | `docs/FLEX_BINDER_CAMPAIGN_V1.md:329-341` |
| 分解证据 | 对打乱族均值可达 1.00/0.99/0.97（seq_7/6/0），对设计可跌到 **0.00**（seq_4）与 0.33（seq_0）→ 裁判识别"非 native 组成"，不是"近 native 竞争者中的 native" | `results/flex_c1_v1/boundary_summary.json`（`note_shuffles_vs_designs`）；`docs/FLEX_BINDER_CAMPAIGN_V1.md:322-328` |
| 信息重叠 | 裁判与 MPNN-NLL 每池 Spearman 家族均值多为正（0.21–0.77），seq_3 为 −0.16 | `results/flex_c1_v1/boundary_summary.json`（`judge_vs_mpnn_spearman_family_means`） |
| 后果 | 按 M8 停止"裁判优于基线"的开发主张，重定位为基准/边界研究 | `docs/FLEX_BINDER_CAMPAIGN_V1.md:339-341` |

### 2.11 FLEX Phase D（仅草案，**被阻塞**）

| 项 | 内容 | 依据 |
|---|---|---|
| 状态 | 草案，**未冻结**；可行性门"原料 PASS、组装 OPEN" | `docs/FLEX_PHASE_D_BINDER_TASK_PREREG_DRAFT_V1.md:1-15` |
| 原料 | 30,388 抗原带阴性；阴性 1,234,987 条；200/200 结构桥抗原有同抗原阴性 | `results/flex_phase_d_v1/feasibility_report_v1.json`（`iedb_counts`）；`feasibility_addendum_v3.json:12-14` |
| 组装试产 | v3：200 家族仅 **1 个通过**，平均产率 0.96%；v3b（全阴性池）：产率 **1.63%**，仍 1/200，339 binder 仅 8 个匹配 | `results/flex_phase_d_v1/pilot_v3_summary.json:2-6`；`pilot_v3b_summary.json:2-10` |
| 失败原因（数学事实） | 12–15 mer 精确组成多重集碰撞概率天然极低 → **精确组成匹配在任何可达池规模下不可行** | `docs/FLEX_PHASE_D_BINDER_TASK_PREREG_DRAFT_V1.md:76-96` |
| 受体级单位被证伪 | IEDB B-cell 受体注释 4,734 条：阳性 4,160 vs 阴性 **12** → "同一抗体双标签"在公共数据里不存在 | `feasibility_addendum_v3.json:3-8` |
| 功效约束 | 24 家族检出 75% 家族阳性功效 0.77；30 家族 0.89；当前 8–9 家族同类功效 **~0.24**（"不设门、直接判负"） | `results/flex_phase_d_v1/power_table.json`；`docs/FLEX_PHASE_D_BINDER_TASK_PREREG_DRAFT_V1.md:45-47` |
| 倾向 | A'（扩池）已证伪 → 文内倾向 **B（终止，记录为"公共数据不足以支撑诚实严格度"）** | 同上 `:84-96` |

### 2.12 AF2 打分边界（负结论链，冻结）

| 项 | 内容 | 依据 |
|---|---|---|
| 结论链 | 单序列 AF2 不能折叠抗体（MSA 缺失伪影）；全复合物 ipTM 在 recycle 0/1 都无法区分 native vs scrambled 短肽（抗体 435 残基主导）；界面 PAE 的判别力**依赖 recycle 深度** | `publication/af2_scoring_boundary_final_decision_v1.json`（`findings_chain`） |
| 关键数字 | 6YXM native：单序列 ipTM 0.22；MSA r=0：native 0.753 vs scrambled 0.755；r=1 五模型 interface PAE 差 −2.03 ~ −4.94（5/5 同向） | 同上 |
| 边界 | 界面 PAE 部署角色仍是**跨骨架相对排序**（Spearman 0.70–0.87）；绝对判别为"SUGGESTIVE-PENDING-REPLICATION"（n=1 骨架） | 同上（`honest_end_state`） |
| 结论 | "去从头设计抗体是否真的结合无序靶点"**在现有证据基础上无法计算性认证**，SPR/BLI 仍是决定性读数 | 同上（`honest_end_state` / `recommended_next_step`） |

### 2.13 4HIX 匹配生成器诊断（开发级、小样本）

| 臂 | 候选 | native recovery | 平均候选-median ipTM | developability pass | 依据 |
|---|---:|---:|---:|---:|---|
| 现 BFN（disorder on） | 24 | 0.080 | 0.456 | 0.708 | `INTEGRATIVE_ANALYSIS.md:155-162` |
| 现 BFN（disorder off） | 24 | 0.153 | 0.445 | 0.750 | 同上 |
| parent BFN | 24 | 0.063 | 0.452 | 0.792 | 同上 |
| ProteinMPNN | 24 | **0.326** | 0.456 | 1.000 | 同上 |
| native / shuffle | 1 / 1 | 1.000 / 0.250 | 0.462 / 0.464 | — | 同上 |

边界：表位仅 6 残基（DAEFRH）；AF2 未分开 native 与 shuffle（0.462 vs 0.464）；PRODIGY 种子方差大；4HIX Fab 在 AF2 训练集内；单一骨架（`INTEGRATIVE_ANALYSIS.md:164-190`）。**所有数字都是 AF2 端点，不是结合数据。**

### 2.14 AAYL 亲和力排序（唯一带"实验端点"的线，负结果）

| 项 | 内容 | 依据 |
|---|---|---|
| 结论 | H3 ECLS 在两个亲本谱系中**没有正向亲和力排序增益，且弱于负 complex NLL 基线** | `docs/AAYL_FUNCTIONAL_RETROSPECTIVE_RESULTS.md:3` |
| 数字 | AAYL49/Ab-14：ECLS Spearman **−0.121**，负 complex NLL **+0.063**；AAYL51/Ab-91：**−0.071** vs **+0.132**（508 条变体） | 同上 `:9-12` |
| 边界 | 端点来自**公开 AlphaSeq 校准估计（非本项目 SPR/BLI）**，AF3 结构，同一抗原两个谱系；流程**非盲测**（评分前已见公开汇总与端点核对值） | 同上 `:14-16,49` |
| 后续 | 不足以支持"有效亲和力排序器"；不得用同批数据调参找正结果 | 同上 `:54` |

---

## 3. 契约测试与工程风险

### 3.1 tests/ 实际覆盖了什么

规模：`tests/` **181 个 `*.py`**、**630 个 `test_` 函数**（本次 `ls tests/*.py | wc -l` 与 `grep -c '^def test_'` 计数）。默认面是 **CPU-only、无 checkpoint、无 GPU**：

- `tests/conftest.py:5`（"default: CPU-only, no checkpoints, no GPU"）、`:24-34`（gpu/slow 默认 skip）
- `pyproject.toml:65-72`（`testpaths=["tests"]`，markers `gpu`/`slow`）
- 全库仅 1 个文件带 `mark.gpu`、1 个带 `mark.slow`（本次 grep 计数）

FLEX 线新增 3 个契约文件（**均为 untracked**，见 §3.2）：

| 契约文件 | 覆盖内容 | 数量 |
|---|---|---|
| `tests/test_flex_a1_scaffold_audit_contract.py` | AF2 multimer 链分隔符语义（`VH:VL` 拆链，历史事故致 ipTM 低估 4.5 倍）；三字母→单字母残基计数（A1 全零假阴性同款 bug） | 6 条（`:35-80`） |
| `tests/test_flex_j0_contracts.py` | 打乱组成保持、RNG 可复现（预注册种子）、组件 PDB 链与区域、region batch 的上下文语义 | 4 条（`:31-66`） |
| `tests/test_flex_j0b_contracts.py` | 按候选平均的边际化、有效性规则真/反例、conformer PDB 序列保持 | 3 条（`:24-57`） |

其他与本文相关的既有契约（仓库自述在位）：`tests/test_scientific_contracts.py`（343 行）、`test_bfn_loader_antigen_roles.py`（antigen 枚举掩码）、`test_freeze_candidate_interface_pae_deployment.py`、`test_abeta_4hix_af2_gate.py`、`test_ecls_submission.py`、`test_contact_v2_confirmatory_contract.py`。

### 3.2 工程风险清单（按严重度）

| # | 风险 | 事实依据 | 影响 |
|---|---|---|---|
| R1 | **FLEX 整条线的代码与其产物不在 git 里** | `git status --porcelain` 34 条 untracked：28 个 `scripts/flex_*.py` + 3 个 flex 测试 + 2 个 FLEX docs + `scripts/oneoff/flex_campaign_v1/`；`results/` 在 `.gitignore:78`（`git ls-files results \| grep -c flex` = **0**） | 新线的可复现性只靠 JSON 内 sha256，无版本历史；任何人 checkout 仓库都拿不到这条线 |
| R2 | **C1 判定逻辑没有契约测试**，且确实带两处方向错误进冻结产物 | 只有 a1/j0/j0b 三个 flex 契约文件，**无 `test_flex_c1*`**；`scripts/flex_c1_score_and_rank_v1.py:169-179`（docstring "0 = native best"，实现 1=best）、`:341`（`bootstrap_ci[1] < 0.5`）；错误由 `results/flex_c1_v1/direction_correction.json` 事后披露 | 两处方向 bug 靠人眼事后发现；同一模式可再发生 |
| R3 | **`top1_rate`/`top5_rate` 字段统计的是"垫底"而不是"夺冠"** | `scripts/flex_c1_score_and_rank_v1.py:311-314`：`top1 = percentile == 0`、`top5 = percentile <= 5/(pool_size-1)`，而 `percentile_of_native` 在同一文件 `:169-179` 返回 1=best | 见 §6-2：手稿"never top-5"与冻结数据冲突 |
| R4 | `results.json` 里的 `status` 与 `final_decision.json` 的结论不一致（历史字段） | `results/publication/h3_ecls_temporal_final_v1/results.json`（`status="development_only; publication final not evaluated"`）vs `final_decision.json:3`（`decision="accepted"`）；`final_decision.json:33` 自认这是 legacy 字段名 | 读者可能据 `results.json` 低估或据 `final_decision.json` 高估这批证据的地位 |
| R5 | v5.1 LMDB 指纹自相矛盾 | `data/confidence_conformation_v5/frozen_manifest_v5_1.json:8-9`：`canonical_fingerprint=1184e7…` vs `previously_recorded_fingerprint=ece2c9…` | 无法用单一指纹证明数据未变；只能靠 count + 单次重算 |
| R6 | v5.1 checkpoint 谱系不在"唯一权威索引"里 | `configs/CHECKPOINT_TRUTH.md`（32 行）无 C3；`grep -rn "v5_1" configs/*.yml` 为空 | 124 MB × 4 阶段资产无人认领，易被误当"当前模型"或永久遗忘 |
| R7 | WSL venv 内第三方补丁无法被测试覆盖 | `HANDOFF_V5.md:51-59`（`L_modules_multimer.py` 的 `L = num_residues` → `L = num_res`）；`docs/FLEX_BINDER_CAMPAIGN_V1.md:144-145` 明确"契约测试无法触及，记为环境风险" | 重装 alphafold 后静默回归，只有跑起来才报错 |
| R8 | 脚本面巨大且平铺 | `scripts/` 共 **562** 个 py，其中顶层 **339** 个（本次计数）；`docs/REPOSITORY_BOUNDARIES.md:5-12` 自己也把 `scripts/` 标为"active workflows" | 入口难辨；本次已见同一功能多版本并存（`flex_a1b_..._v1/_v2`、`flex_phase_d_assembly_pilot_v1..v3b`） |
| R9 | 文档与代码的轻微漂移示例 | `README.md:81` 称 `app.py` 是"65-line launcher"，实测 `wc -l app.py` = **73**；`publication/idp_platform_capability_v1.json` 仍把 V6 Phase 2 列为 BFN 资产，而 `configs/CHECKPOINT_TRUTH.md:25` 已把它降级为历史 | 说明"文档=事实"不成立，需以文件+哈希为准 |
| R10 | 依赖外部服务与服务器稳定性 | `docs/FLEX_BINDER_CAMPAIGN_V1.md:132-134`（ATLAS 曾 WinError 10060，断点续跑）；`results/flex_phase_d_v1/feasibility_report_v1.json`（`protabank_status`：网络不可达，SSL 失败） | 复现依赖缓存；ProtaBank 侧缺失 |
| R11 | 环境三分为 Windows venv / WSL / `D:\DisorderFlowRuntime` | `README.md:110-112`；`HANDOFF_V5.md:20-45`（jax GPU 在 Windows 无 wheel） | 计算复现必须写明发行版与路径，否则不可重放 |

### 3.3 本次审计未覆盖/未执行的（诚实边界）

- **未运行 `pytest`**：因此"6 条全绿 / 4 条全绿 / 3 条全绿"等均为 `docs/FLEX_BINDER_CAMPAIGN_V1.md:138,232,268` 的仓库自述，不是本次观察。
- 未审 `docs/` 全部 markdown（该目录约 50 个 md）（只精读任务指定 8 篇 + 按需 4 篇：`PAE_NEW_FAMILY_FEASIBILITY_V21.md`、`REPOSITORY_BOUNDARIES.md`、`AAYL_FUNCTIONAL_RETROSPECTIVE_RESULTS.md`、`AAYL_FAILURE_DIAGNOSIS_AND_REPAIR.md` 开头）。AAYL 另 3 篇（learnability / low-order / remeasurement）未读。
- `_archive_local/`、`outputs/`、`reviewer_outputs/`、`dist/`、`casp17/`、`iBEC_Submission*` 未系统盘点。
- 未对 96 个 `logs/bfn_*` 目录逐一核验 checkpoint 完整性。

---

## 4. 湿实验状态：有纸面计划，没有自家实测数据

**结论：仓库里有 3 份完整的湿实验计划/物料包，有 1 处"计划即状态"的机器可读记录明确写着 `not_executed`，有 1 套盲法 BLI 布局与结果模板（空表），但没有任何本项目自己产生的结合测量数据。**

### 4.1 有什么（纸面）

| 资产 | 内容 | 依据 |
|---|---|---|
| `WETLAB_VALIDATION_PLAN_V15.docx` | 56,293 B，2026-07-05；V15，25 个候选，SPR/BLI 执行级细节、预算、优先级（本次解包统计 2,422 段，SPR 命中 19 处、BLI 21 处、KD 13 处） | 文件本身；`SYNTHESIS_SPECS.md` 同源 |
| `QLU_NATBIOTECH_ROADMAP_V1.md` | 本科在读 + 25 个 scFv 候选 + 4 阶段（校内表达→BLI 初筛→SPR 定量→写论文），预算 ¥20k–50k，时间表 2026-07 至 2027 | `QLU_NATBIOTECH_ROADMAP_V1.md:3,47-54,60-61,199-221` |
| 合成规格 | 4HIX-D1..D5 的完整 VH 序列、H3、AF2 ipTM（0.4491 native → 0.4615 最好）、轻链、载体/纯化建议、对照定义 | `SYNTHESIS_SPECS.md:6-62` |
| 候选 DNA/AA 物料 | 25 条 scFv（VH-(G4S)3-VL-His6），`candidates_dna.fa`（20,566 B）等 6 个文件；**S0 完成** | `idp_design_results/s5_wetlab_synthesis_20260705/synthesis_summary.json`（`total_candidates=25`、`step=S0 sequence completion`、`date=20260705_223632`） |
| Aβ/4HIX 前瞻物料包 | 盲法 BLI 布局（`bli_blinded_layout.csv` 13,847 B）、盲法键、**空的**结果模板（145 行，数据列全空）、实验验收判据（预注册于测量前） | `results/prospective/abeta_4hix_expression_prep_v1/`；`experimental_acceptance_criteria.json`（`status="preregistered_before_measurements"`） |
| 验收判据 | 候选通过：信号 ≥ 外部阴性 3.0 倍、打乱/BSA ≤ 目标的 20%、需两个独立表达批次；且明确"单浓度筛选**不能**确立 KD"、"3D6-H3-shuffle 反事实**不是**实验证实的非结合体，不能替代外部无关 scFv 阴性" | 同上 `experimental_acceptance_criteria.json` |
| 湿实验状态机 | 6 个构建体（5MP3×3、8B9V×3）× 9 个阶段（表达/纯化/身份/SEC/DSF/BLI/SPR/状态特异性/正交确认）**全部 `not_executed`**；12 个抗原状态全部 `not_executed`；8 个 gate 全部 `not_executed`；`results_present=false` | `outputs/non_abeta_idp_wetlab_handoff_v1/wetlab_status.json`（`overall_status="not_executed"`、`notes[0]="No experimental result is encoded in this file."`） |
| Aβ 交接报告 | "Candidates: 2; status: `ready_for_protocol_review_and_execution`"；`Claim Boundary`：无实验结合/亲和力/状态特异性声明 | `outputs/abeta_wetlab_handoff_v1/wetlab_handoff_report.md`（前 40 行） |

### 4.2 没有什么（实测）

- **没有本项目产生的 SPR/BLI 传感图、KD、RU、复孔 CV**：`bli_results_template.csv` 只有表头与空列；没有 `sensorgram`/`SPR/*.csv` 类数据文件（除第三方 PDB 文件名的巧合命中）。
- **没有合成订单回执、没有表达产量、没有 SEC/DLS 数据**：`expression_batch_tracking.csv`、`construct_manifest.csv` 是跟踪模板。
- **仓库自述一致**：`README.md:73-74`（"No wet-lab binding measurements are claimed"）、`INTEGRATIVE_ANALYSIS.md:102-107`（"No wet-lab data / Generated H3 sequences were not tested for binding"）、`publication/idp_platform_capability_v1.json`（`claim_boundary`："No wet-lab validation has been performed"）、`publication/ECLS_SCOPE_FREEZE.yml:33,37`（排除 "planned SPR/BLI experiments"；`wet_lab_language_before_measurements: forbidden`）。
- **唯一的"实验数据"是借来的**：AAYL 线用的是 **公开 AlphaSeq 的估计 Kd 端点**（508 变体，同一抗原两谱系），不是本项目测量；且该线结论为负（`docs/AAYL_FUNCTIONAL_RETROSPECTIVE_RESULTS.md:3,14-16`）。
- **时间线事实**：路线图计划 2026-09 做 BLI 初筛（`QLU_NATBIOTECH_ROADMAP_V1.md:208-214`）；今天是 2026-11-01，`wetlab_status.json` 的最后更新是 **2026-07-28**，所有阶段仍是 `not_executed`。即：**计划里的时间点已过，状态未被更新为已执行**（不能据此断言"做失败了"，只能断言"仓库里没有结果"）。

---

## 5. 缺口清单：要实现「针对无序/柔性表位的结合体设计并实验验证」

> 每条给依据。这五类缺口是**并列**的：补齐任何一类都不足以宣称目标达成。

### 5.1 数据缺口

| # | 缺什么 | 依据 |
|---|---|---|
| D1 | **独立抗原家族的确认性队列**（≥24 家族，目标 30）。现有 FLEX 池只有 14 PDB / 9 新家族；9 家族在同类效应下功效仅 ~0.24 | `docs/FLEX_BINDER_CAMPAIGN_V1.md:162-168`；`results/flex_phase_d_v1/power_table.json` |
| D2 | **实验证实的 non-binder（阴性）**，且与 binder 组成匹配。"不在 PDB 里"不构成阴性 | `docs/FLEX_PHASE_D_BINDER_TASK_PREREG_DRAFT_V1.md:34-40` |
| D3 | **精确/近似的组成匹配对照在真实序列空间不可得**：709,890 独立阴性序列、672,336 组成，命中率 1.63%，200 家族过 1 个 | `results/flex_phase_d_v1/pilot_v3b_summary.json:2-10` |
| D4 | **配对 H/L 抗体–短肽公共空间已耗尽**：新增合格家族 = 0；VHH 口径 3 个干净条目 / 1 个可确认组件 | `docs/PAE_NEW_FAMILY_FEASIBILITY_V21.md`；`publication/pae_analog_manuscript_draft_v1.md:29` |
| D5 | **物理可信的柔性标签**：MD-vs-NMR 配对天花板 2 对（129 中）；v5 的 AF2 5 种子 RMSF 是"代理的代理"（种子间 per-res std ≈ 0.09） | `docs/FLEX_BINDER_CAMPAIGN_V1.md:55-60,118-134,169-171` |
| D6 | **物理标签门未过**：中位 Spearman 0.217 < 0.30、偏 Spearman 0.1485 < 0.15 → 只能 `rank_only`，不能做连续 RMSF 回归 | `data/confidence_conformation_v5/rmsf_calibration.json:18-28,45` |
| D7 | 亲和力注释未核验：249 行里 180 行方法 Unknown，与 A1 合格集交集仅 1 个 PDB | `results/flex_a4_affinity_inventory_v1/summary.json:2-11` |

### 5.2 模型缺口

| # | 缺什么 | 依据 |
|---|---|---|
| M1 | **底模本身未在"无序/柔性表位"上做过结合相关的训练**：主稿 ECLS 用 ProteinMPNN，FLEX 裁判用 BFN，两者都只是似然；没有任何模型以"结合/非结合"为监督目标训练过 | `scripts/benchmark_h3_epitope_delta.py:1-2`；`results/flex_j0_v1/j0a_summary.json`（`checkpoint_path`）；`docs/FLEX_BINDER_CAMPAIGN_V1.md:72-76` |
| M2 | **v5.1 四级模型无任何评测**：124 MB checkpoint、无指标、无下游任务、无 config 引用、不在 CHECKPOINT_TRUTH | `logs/v5_1_pipeline/state/final_checkpoint.txt`；`configs/CHECKPOINT_TRUTH.md` |
| M3 | **disorder 头劣于 metapredict 且未外校准**（0.7551 vs 0.7927；Brier 0.2991 / ECE 0.3794） | `docs/DISORDER_SUPERVISION_PROTOCOL.md:122-127` |
| M4 | **生成能力与裁判能力错配的历史根源**：BFN native recovery 0.080 vs MPNN 0.326，而早期 BFN 同时承担生成+排序。FLEX 已改为"生成外包、BFN 只当裁判"，但**裁判在 C1 仍失败**——说明换分工没有解决能力问题 | `INTEGRATIVE_ANALYSIS.md:157-160`；`docs/FLEX_BINDER_CAMPAIGN_V1.md:22`（M8）；`results/flex_c1_v1/c1_summary.json:63-67` |
| M5 | 无 binder/non-binder 判别器（Phase D 只到草案，且倾向终止） | `docs/FLEX_PHASE_D_BINDER_TASK_PREREG_DRAFT_V1.md:84-96` |

### 5.3 裁判（评价函数）缺口

| # | 缺什么 | 依据 |
|---|---|---|
| J1 | **ECLS 类似然分已经被证明"组成可替代"**：5/8 族输给 20 维组成基线 | `results/flex_c1_v1/direction_correction.json:11-14` |
| J2 | AF2 单姿态 ipTM 对 native/shuffle 臂差仅 0.0004–0.013；PRODIGY 种子方差大 | `INTEGRATIVE_ANALYSIS.md:164-168`；`docs/PAE_ANALOG_TRANSFER_V30.md:33` |
| J3 | AF2-Multimer PAE 目标是**组成主导**的（native vs 组成打乱不可分），不能当序列特异性裁判 | `publication/pae_analog_manuscript_draft_v1.md:41` |
| J4 | 界面 PAE 的绝对判别只在 n=1 骨架上"待复现"；pLDDT 跨骨架迁移被五次消融证伪；ipTM 可靠对不足（2/6 骨架） | `publication/candidate_interface_pae_deployment_v1.json`（`abstentions`）；`publication/af2_scoring_boundary_final_decision_v1.json` |
| J5 | 没有任何"结合测量"级别的裁判；AF2 端点 ≠ 结合 | `INTEGRATIVE_ANALYSIS.md:104-107`；`publication/af2_scoring_boundary_final_decision_v1.json`（`honest_end_state`） |

### 5.4 生成缺口

| # | 缺什么 | 依据 |
|---|---|---|
| G1 | 只有一种外包生成器（ProteinMPNN vanilla v_48_020）与一种自家生成器（BFN 固定骨架）；无多生成器交叉 | `publication/flex_boundary_manuscript_draft_v1.md:174-183` |
| G2 | 生成多样性受限：1,300 designs → 789 不同（最差组件 50 个设计只 7 个不同，`8opi_A_B`） | `results/flex_c1_v1/generation_summary.json`（`rows[*].n_distinct`） |
| G3 | 没有"结合体格式"的生成：C1/合成规格都是固定骨架 CDR-H3 替换，不是 de novo binder 生成 | `SYNTHESIS_SPECS.md:1-6`；`docs/FLEX_BINDER_CAMPAIGN_V1.md:5` |
| G4 | 骨架类别单一：单域 VHH/nanobody（14 PDB），未覆盖 monobody/FN3 等（文内自己列为扩宇宙方向） | `docs/FLEX_BINDER_CAMPAIGN_V1.md:167-168,216-219` |

### 5.5 实验缺口

| # | 缺什么 | 依据 |
|---|---|---|
| E1 | **零条自家结合测量**：所有湿实验阶段 `not_executed`，`results_present=false` | `outputs/non_abeta_idp_wetlab_handoff_v1/wetlab_status.json` |
| E2 | 没有合成/表达的落地记录：物料包是"ready_for_protocol_review_and_execution"，不是"已下单/已表达" | `outputs/abeta_wetlab_handoff_v1/wetlab_handoff_report.md:3`；`idp_design_results/s5_wetlab_synthesis_20260705/synthesis_summary.json`（`step=S0`） |
| E3 | 没有仪器/机时事实：路线图还停在"去共享平台搜 Biacore/Octet" | `QLU_NATBIOTECH_ROADMAP_V1.md:19-31,106-136` |
| E4 | 没有阴性对照的实物：唯一现成的阴性只有外部无关 scFv（计划中），shuffle 不被承认为阴性 | `results/prospective/abeta_4hix_expression_prep_v1/experimental_acceptance_criteria.json`（`counterfactual_control_boundary`） |
| E5 | 计划书里明确要求的判定链（表达→SEC 单体→BLI→SPR 定量→反向筛选）**一步都没执行** | `QLU_NATBIOTECH_ROADMAP_V1.md:65-158`；`WETLAB_VALIDATION_PLAN_V15.docx` |

---

## 6. 容易被夸大的地方（重点）

> 本节是本次审计最需要下游注意的部分。每条都给可核依据。

1. **"ECLS 被复现/被验证"是跨了底模的**。主稿 ECLS 用 **ProteinMPNN v_48_020**（`scripts/benchmark_h3_epitope_delta.py:1-2,102-120`；`configs/benchmarks/peptide_h3_ecls_temporal_final_v1.yml:9-14`），FLEX 的 `flex_ecls_marginal` 用 **BFN s2032**（`results/flex_j0_v1/j0a_summary.json` `checkpoint_path/sha256`）。两者共享公式、不共享模型。`docs/FLEX_BINDER_CAMPAIGN_V1.md:258-259` 已自我限定为"开发级复制，不是确认"，但 `publication/flex_boundary_manuscript_draft_v1.md:156-158` 的"replicates across a scaffold class and cohort with no homology to the original panel"读起来像同一指标的独立复现。**引用时必须写明底模。**

2. **手稿里"The native is never ranked first…and never top-5"第二句与冻结数据冲突。**【本审计重算】我用 `results/flex_c1_v1/scoring/*.json` + `results/flex_j0_v1/scoring/*.json` 的冻结缓存，按 `scripts/flex_c1_score_and_rank_v1.py:216-220` 的公式逐候选重建，得到：**native 从未排第 1（0/26，最好排名第 2）**，但**在 9/26 个池中排进前 5**（如 6zwk_D_J、6zwk_E_K、8op0_A_B、8opi_A_B、8pii_A_B 均为第 2 名；6i2g_A_B、6zwk_B_H 第 3；8b8i_E_M 第 2；9eiu_B_A 第 5）。手稿的"never top-5"与 `c1_summary.json:10-11` 的 `top1_rate/top5_rate = 0.0` 完全一致（很可能是其来源），而这两个字段按 `scripts/flex_c1_score_and_rank_v1.py:311-314` 统计的是 `percentile == 0` 与 `percentile ≤ 5/(pool_size-1)`，在 1=best 的约定下等于"**从不垫底**"。正确表述应是：**从不是第 1（最好第 2），9/26 池进前 5；从不垫底。** 这条会直接改变边界手稿的措辞强度。

3. **`composition_logistic` 的置信区间下界高于它自己的点估计**（mean 0.54399 vs CI [0.54496, 0.63025]，`results/flex_c1_v1/c1_summary.json:23-29`），手稿照抄为 "0.544 ([0.545, 0.630])"（`publication/flex_boundary_manuscript_draft_v1.md:122-123`）。原因不是随机噪声：`pooled_stats` 的点估计是**组件等权**均值（`:304`），bootstrap 却是**家族聚类**（每族抽一个组件后再平均，`:291-306`），两者估计量不同。家族等权均值 【本审计重算】为 0.5844。后果：读者会以为"CI 排除了自己的点估计 = 有问题"或反过来误读为"组成基线其实 0.58"。**报告时应同时给两种口径。**

4. **冻结结果文件自己写着 `development_only`**：`results/publication/h3_ecls_temporal_final_v1/results.json` 的 `status` 是 `"development_only; publication final not evaluated"`，而 `final_decision.json:3-7` 判定 `accepted` / `terminal` / `rerun_permitted=false`，并在 `:33` 解释"legacy 字段名 + 单独 decision 文件确立 final 地位"。**两个文件都不假，但不能只引一个。**

5. **"15 clusters" 是 15 个推理单元，31 个结构**；预注册功效只够检出很强的效应（J0 的对应数字：9 族单侧符号检验临界数 8，实测 7/9 的 p≈0.090，`docs/FLEX_BINDER_CAMPAIGN_V1.md:252-257`）。把 12/15 当作"80% 阳性率"的一般化证据会高估。

6. **Phase D 的"原料富余三个数量级"与"200 家族过 1 个"必须一起读**。前者在 `docs/FLEX_PHASE_D_BINDER_TASK_PREREG_DRAFT_V1.md:6-11`，后者在同文件 `:76-82`。单引前者会得出"数据够了"的错误结论；仓库自己的结论是**精确组成匹配是数学不可行**（`:88-90`）。

7. **同一个数字不一定是同一个东西**："1289" 是 v5.1 冻结条数（`data/confidence_conformation_v5/frozen_manifest_v5_1.json:4`），构建目标原本是 **1301**（`HANDOFF_V5.md:11`），而 `publication/idp_platform_capability_v1.json` 写成 "1289 conformations, **863 IDPs** × 5 seeds"。三个说法指同一批数据的不同切面，不能互换。

8. **"57 PDB" 是候选池上界，不是合格数**。仓库自己写了这一点（`docs/FLEX_BINDER_CAMPAIGN_V1.md:46-48`、`results/flex_campaign_v1_scaffold_scan.json:3`），但"57 → 36 → 14"这条链在摘要里被引用时极易丢掉中间的门。

9. **A4 的 249 行亲和力是未核验元数据**（`docs/FLEX_BINDER_CAMPAIGN_V1.md:159-160`），不能当"我们有 249 条带 KD 的复合物"。其中 180/249 行方法未知（`results/flex_a4_affinity_inventory_v1/summary.json:9-11`）。

10. **"native recovery 0.326" 是 AF2 端点，不是结合**。它来自 4HIX 匹配诊断的 AF2 ipTM/PRODIGY 打分（`INTEGRATIVE_ANALYSIS.md:153-168`），且 AF2 **未分开** native 与 shuffle（0.462 vs 0.464）。任何"ProteinMPNN 恢复率 33%"的表述都必须带"AF2 端点"。

11. **"all_authors_approve_submission: true" 不等于可投稿**。同一文件里 `author_contributions=null`、`funding=null`、`conflicts_of_interest=null`、`related_manuscripts_status_confirmed=false`、`manuscript_data_license_confirmed=false`，`pending` 列表非空，`journal_submission_confirmed=false`、`remote_publication_confirmed=false`（`release/ecls_v1/submission_metadata.json`，键 `author_contributions` / `pending` / `journal_submission_confirmed` 等；该文件含非 UTF-8 字节，需 `encoding='utf-8'` 读）。`docs/PUBLICATION_MAP.md:16` 也写了"Planned DOI … is not evidence of a published archive"。

12. **"本地稿"≠"已投稿"**：三份稿件全部是本地文件，且 `publication/*.md` 被 `.gitignore:119` 忽略 → 连版本控制都没有（`publication/MANUSCRIPT_DRAFT.md`、`publication/flex_boundary_manuscript_draft_v1.md` 均为 ignored；`git status` 里连 untracked 都不显示）。

13. **C1 的 pool_size 记录是名义值 101**（`c1_results.json` 每行 `pool_size: 101`），但实际去重后的竞争者是 **57–100**（例如 8opi_A_B 只有 57 个 others）——因为 `judge_scores` 是 dict，重复设计会塌缩（`scripts/flex_c1_score_and_rank_v1.py:216-220`）。百分位的分母因此不是 100。这会让"101-candidate pool"的表述偏乐观。

14. **两个方向错误都发生在"看起来最不需要测试"的地方**（百分位极性、CI 不等式方向），并且**进了冻结产物**（`c1_summary.json:63-65` 记 `judge_ci_beats_random=false`，而实际 0.676 > 0.5 应通过）。这不是"已经修正所以没事"——因为冻结文件保持原样（这是对的），后续引用者仍会读到错字段。

---

## 附录 A：本次审计用到的关键命令（只读，可复现）

```bash
# git 状态（release 分支，HEAD 13160681）
git -C D:/biological/DisorderFlow status --porcelain
git -C D:/biological/DisorderFlow ls-files results | grep -c flex     # -> 0

# checkpoint 哈希（本次复算，与记录一致）
python -c "
import hashlib
p='logs/bfn_disorder_v4_afdb_supervised_2026_08_05__17_22_45_disorder_v4_balanced_long_s2032/checkpoints/best.pt'
h=hashlib.sha256()
with open(p,'rb') as f:
    for c in iter(lambda: f.read(1<<20), b''): h.update(c)
print(h.hexdigest())   # -> ed9984504d799187b7660c1d40c3dc68c01a0ce9b3cf92e25af1273195931252
"

# ECLS 冻结件哈希（与 ECLS_SCOPE_FREEZE.yml:14,17 一致）
# final_decision.json -> 0e32c22c…cb0
# results.json        -> d4c47461…c69

# SAbDab2 归档 MD5（与 DATABASE_SOURCES.md:39 一致）
# data/sabdab2_current/splits.tar.gz -> 0dbb4cc499e9eb77f14008b232f2c38c

# C1 排名重建（只读冻结 JSON，无模型推理）：见 §6-2
```

## 附录 B：文件读取受限记录

- 无 ACL/权限失败。所有指定文件均可读。
- `WETLAB_VALIDATION_PLAN_V15.docx` 需按 UTF-8 解包 `word/document.xml` 才能正确显示中文（控制台默认 GBK 会乱码）；已确认其为**计划书**而非结果报告。
- `release/ecls_v1/submission_metadata.json` 含非 UTF-8 字节，需显式 `encoding='utf-8'` 读取（默认 GBK 报 `UnicodeDecodeError`）。
