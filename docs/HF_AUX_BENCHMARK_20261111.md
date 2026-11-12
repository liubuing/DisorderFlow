# HuggingFace 辅助数据基准 v1（2026-11-11）

状态：**development benchmark；不改变任何冻结的 ECLS/PAE 结论；不构成 binding/affinity 主张。**

本报告记录一次纯干实验的“外部数据补强”战役。目标不是修 BFN 生成器，而是回答三个更基础的问题：

1. ProteinMPNN 的突变似然能不能在**同一抗体–抗原复合物内**排序实验 ΔΔG，并且超过 20 维组成基线？
2. 公开 AlphaSeq 亲和力数据上，序列模型（ESM2）能不能学到超出组成的信号？
3. 公开 SAbDab2-Epi 的肽复合物能不能扩展现有 31 结构/15 簇的结构面板，并在肽侧复现“native vs 组成匹配 shuffle”的上下文效应？

---

## 1. 结论摘要

| 子战役 | 结果 | 判定 |
|---|---|---|
| **AB-bind** | 591/683 条突变、18 个复合物；`dils` 均值 rho **−0.167**，95% CI [−0.296, −0.045]，76.5% 复合物为负；组成基线 +0.116；`dils` 在 94% 复合物内优于组成；置换检验 **p=0.0004** | **PASS（development）** |
| **AB-bind 抗体突变子集** | 13 复合物；均值 rho **−0.195**，69.2% 为负；置换检验 **p=0.0012** | 方向一致，接近 70% 门 |
| **AB-bind 单突变** | 15 复合物；均值 rho **−0.213**，**15/15** 为负 | 最强子集 |
| **AB-bind 多突变** | 7 复合物；均值 rho −0.104，42.9% 为负 | 弱，不单独主张 |
| **跨复合物 ranker** | 留一复合物 Ridge：均值 rho **+0.243**（方向反转） | **FAIL**；不训练全局 ranker |
| **AlphaSeq** | 8 数据集/19 靶点；composition 0.279、ESM2 **0.642**、组合 0.648；ESM2 在 19/19 靶点优于组成 | **强序列信号** |
| **AlphaSeq group split** | 3 数据集；composition 0.314、ESM2 **0.714**、组合 0.716 | 仍强；但 KMeans 非序列身份聚类 |
| **AlphaSeq MMseqs2** | 3 数据集；composition 0.295、ESM2 **0.703**、组合 0.704 | 90% identity 聚类下仍强 |
| **SAbDab2-Epi 可行性** | 6,979 复合物；1,111 条 PEPTIDE 抗原链；1,043 条 5–50 aa；622 PDB；213 肽簇；val/test 420 个合格复合物、261 PDB | **可扩展结构面板** |
| **SAbDab2-Epi PCLS 试点** | 29 复合物；**25/29 为正**（86.2%）；均值 advantage **0.546**，中位 0.534 | **正信号（pilot）** |
| **SAbDab2-Epi H3-ECLS（ANARCI/IMGT）** | 408 行 / 254 PDB / 157 CDRH3 簇；**400/408 为正（98.0%）**，均值 0.672；簇级 98.1%，CI [0.606, 0.703]；val 98.3%、test 97.5% | **最强结构侧信号** |
| **公开湿实验 Kd 队列** | 111 抗体–肽复合物；advantage vs logKd **rho −0.312, p=0.00085**, CI [−0.476, −0.128]；H3≥14 −0.456；抗原级 −0.304, p=0.002 | **最强 binding-linked 关联（exploratory）** |
| **AbCDR-Binding D44** | 2,048 单点突变；dils 整体 **−0.012 (ns)**；界面 +0.119 (p=0.0015)；BLOSUM +0.374；非界面 −0.100 | **部分负面：dils 非通用排序器** |

一句话：**结构侧的 AB-bind 信号、肽侧 PCLS、H3-ECLS（196 行 ANARCI/IMGT，97.4% 为正）和序列侧 ESM2（MMseqs2 0.703）都为正；但跨复合物的全局 ranker 失败，多突变弱，所有结果都还是 protein-antigen / 结构基准，不是 IDP 结合验证。**

---

## 2. 数据清单

| 数据 | HF 路径 | 规模 | 许可证 | 用途 |
|---|---|---:|---|---|
| AB-bind | `tigerszhangunc/AB-bind` | 1,101 行 / 32 PDB / 2,126 突变 | 卡片未写，内部开发用 | 同复合物突变 ΔΔG |
| Open AlphaSeq | `aalphabio/open-alphaseq` | 9 数据集 / ~1.6M 行 | **MIT** | 多靶点抗体亲和力 |
| SAbDab2-Epi | `SII-Epitope/SAbDab2-Epi` | 6,979 复合物 / 1,111 肽链 | 上游条款 | 结构/表位扩展 |
| SAbDab2 raw | `RosettaCommons/SAbDab_raw` | 全库 | 上游条款 | 备用 |
| DisProt | `LiteFold/DisProt` | 3,337 条 | 上游条款 | disorder 标签备用 |

**排除**：`PDBbind`/`jglaser/binding_affinity` 是蛋白–小分子；`OAS` 搜索被 OpenAssistant 污染；`AbBibench` 已在用；`ZYMScott/vhh_affinity-*` 只有序列+分数、无抗原，不能做结构打分。

---

## 3. AB-bind：同复合物突变 ΔΔG

### 3.1 数据筛选

原始 1,101 行、32 个 PDB。用本地 SAbDab all-summary 过滤：

- 必须有人类可识别的 Hchain、protein antigen_chain；
- 20 个 PDB 命中 SAbDab；
- 1DVF 因 `antigen_chain` 缺失（抗独特型 Fab）被排除；
- 最终 **19 个抗体–抗原复合物 / 683 行**；
- 7 个非抗体界面被排除：1AK4（cyclophilin–capsid）、1FFW（CheA–CheY）、1JTG（TEM-1–inhibitor）、1KTZ（TGF-β receptor–TGF-β）、1T83（FcγRIIIb–Fc）、3K2M（monobody–SH2）、3WJJ（Fc–FcγRIIb）。

### 3.2 打分方法

- **模型**：ProteinMPNN v_48_020；
- **模式**：`--conditional_probs_only 1 --conditional_probs_only_backbone 0`，即 leave-one-out 序列上下文；
- **两种状态**：
  - `complex`：被打分链 + 直接伙伴链；
  - `partner_stripped`：只保留被打分链；
- 每条链重编号为 1..N，避免 PDB 残基编号缺口/多字符链 ID 破坏 design mask；
- 对每个突变位置：
  - `dlogp_complex = logP(mut | complex) − logP(wt | complex)`；
  - `dlogp_stripped = logP(mut | stripped) − logP(wt | stripped)`；
  - `dils = dlogp_complex − dlogp_stripped`；
- 多突变按位点求和（一阶近似）；
- **组成基线**：20 维 mutant−WT 组成差，留一复合物 Ridge（alpha=1.0）；
- **BLOSUM62**：sum BLOSUM62(wt, mut)；
- 主终点：within-complex Spearman(score, ddG)；正 ddG = destabilizing，所以**负 rho 是好方向**。

### 3.3 覆盖

| 项 | 数 |
|---|---:|
| 输入行（SAbDab 抗体–抗原） | 683 |
| 成功打分 | **591（86.5%）** |
| 跳过 | 92 |
| 可分析复合物 | 18 |
| `dils` 可分析复合物 | 17（2NZ9 的 `dils` 恒为 0） |

跳过原因：WT 编号不匹配（Kabat vs PDB author numbering，主要 1N8Z/2NYY）、插入码/删除记法（`Y100aF`、`delta(...)`）、多字符链 ID。

### 3.4 主结果

| score | n 复合物 | median rho | mean rho | rho<0 比例 | mean CI95 |
|---|---:|---:|---:|---:|---|
| **dils** | 17 | −0.154 | **−0.167** | **0.765** | **[−0.296, −0.045]** |
| dlogp_complex | 18 | −0.207 | −0.190 | 0.667 | [−0.328, −0.052] |
| dlogp_stripped | 18 | −0.133 | −0.109 | 0.667 | [−0.236, 0.018] |
| BLOSUM62 | 18 | −0.195 | −0.104 | 0.667 | [−0.282, 0.086] |
| composition Ridge | 18 | +0.051 | **+0.116** | 0.278 | [0.030, 0.216] |

**dils − composition**：均值 Δrho **−0.286**，中位 −0.218，**94.1% 复合物 dils 更好**，95% CI **[−0.446, −0.138]**。

**置换检验**（within-complex 打乱 ddG，5,000 次）：

- 全量 `dils`：observed −0.167，**p=0.0004**；
- 抗体突变子集：observed −0.195，**p=0.0012**。

### 3.5 分层

| 子集 | 复合物 | dils mean rho | rho<0 | composition mean rho |
|---|---:|---:|---:|---:|
| 全部 | 17 | −0.167 | 0.765 | +0.116 |
| 抗体突变（H/L） | 13 | **−0.195** | 0.692 | +0.080 |
| 抗原突变 | 9 | −0.123 | 0.778 | +0.081 |
| 单突变 | 15 | **−0.213** | **1.000** | +0.080 |
| 多突变 | 7 | −0.104 | 0.429 | −0.000 |

**单突变是最干净的信号**：15/15 复合物方向为负，均值 −0.213，组成基线 +0.080。多突变弱，说明一阶相加近似不够，或实验 ΔΔG 的非加性效应占主导。

**单突变 + 抗体侧复现（v1.1）**：只保留 `n_mut=1` 且突变链为 H/L 的记录，12 个复合物 / 280 行：

| 指标 | 值 |
|---|---:|
| dils 均值 rho | **−0.217** |
| 95% CI | **[−0.375, −0.064]** |
| rho<0 比例 | **83.3%** |
| 置换检验 | **p=0.001** |
| 组成基线均值 | +0.082（33.3% 为负） |
| PDB-hash A 半（3 复合物） | −0.264 |
| PDB-hash B 半（9 复合物） | −0.201 |

两个 hash 半区都为负；A 半只有 3 个复合物、CI 跨 0，B 半 9 个复合物、CI [−0.359, −0.038]。这是目前最干净、最可复现的 AB-bind 子集。

### 3.6 简单组合

在复合物内 z-score 后：

- `0.4*dils + 0.4*dlogp_complex + 0.2*BLOSUM`：均值 rho **−0.258**，中位 −0.242，**82.4%** 为负；
- `0.5*dils + 0.5*dlogp_complex`：均值 −0.214，76.5% 为负。

组合略优于单分数，但权重是事后选的，只能作为 exploratory。

### 3.7 失败：跨复合物 ranker

留一复合物训练 Ridge（特征 `dils`、`dlogp_complex`、`BLOSUM62`，目标为复合物内 ddG 秩）：

- 留出预测与 ddG 的 Spearman 均值 **+0.243**（方向反转），只有 3/17 为负；
- 也就是说，**跨复合物学习会学反方向**，不能推广。

这与项目此前的 FLEX/AAYL 结论一致：**同一分数在不同复合物上的方向不稳定，全局 ranker 会学到复合物特异的捷径。** 因此本战役不训练全局 ranker，只保留零样本 `dils` / 组合。

### 3.8 AB-bind 的边界

- 蛋白抗原，不是 IDP/肽；
- 18 个复合物、591 条突变，效应量中等（mean |rho| ≈ 0.17–0.26）；
- 2NZ9 的 `dils` 退化为 0（抗原 A 与 B 不接触）；
- 17 个复合物中仍有 4 个正相关（1DQJ、1JRH、1YY9、2NY7），异质性明显；
- 不能写 binding/affinity 预测，只能写“mutation-effect ranking 的 development 信号”。

---

## 4. Open AlphaSeq：序列侧亲和力

### 4.1 数据

9 个数据集，覆盖：

- CoV2 RBD（YM_0005、YM_0549、YM_1068）；
- TIGIT（YM_0693、YM_0988）；
- PD-1（YM_0852、YM_0985）；
- HER2（YM_0989、YM_0990）。

过滤：`above_background=True`、`sufficient_replicate_observations=True`、非空 `alphaseq_affinity`、标准氨基酸、长度 20–600；按 target+sequence 去重；每个 target 最多 20,000 条。

### 4.2 方法

- 特征：
  - 20 维组成；
  - ESM2 t6 8M（mean-pooled，320 维）；
  - 组合；
- 模型：StandardScaler + Ridge(alpha=1.0)；
- 评估：target 内 5 折随机 CV，Spearman；
- 敏感性：KMeans（50 簇）在 ESM2 嵌入上分组，GroupKFold，减少近重复泄漏。

### 4.3 结果

8 个数据集、19 个 target：

| 特征 | 平均 Spearman |
|---|---:|
| composition | 0.279 |
| **ESM2** | **0.642** |
| composition + ESM2 | 0.648 |

- ESM2 在 **19/19** target 上优于组成；
- 单 target 范围：0.542（TIGIT Mouse）– 0.836（HER2 YM_0989）；
- Group split（3 个数据集）：

| 数据集 | n | composition | ESM2 | 组合 |
|---|---:|---:|---:|---:|
| YM_0989 HER2 | 9,591 | 0.407 | **0.790** | 0.791 |
| YM_0852 PD-1 | 5,357 | 0.275 | **0.678** | 0.678 |
| YM_0990 HER2 | 6,427 | 0.260 | **0.675** | 0.679 |
| 平均 | — | 0.314 | **0.714** | 0.716 |

**MMseqs2 90% identity / 80% coverage（v1.1 正式敏感性）**：用官方 Windows 构建、`--single-step-clustering 1`，对 3 个数据集重新聚类：

| 数据集 | n | 簇数 | composition | ESM2 | 组合 |
|---|---:|---:|---:|---:|---:|
| YM_0989 HER2 | 9,591 | 76 | 0.411 | **0.779** | 0.781 |
| YM_0852 PD-1 | 5,357 | 18 | 0.239 | **0.667** | 0.661 |
| YM_0990 HER2 | 6,427 | 134 | 0.234 | **0.663** | 0.668 |
| 平均 | — | — | 0.295 | **0.703** | 0.704 |

ESM2 在真实 90% identity 聚类下仍有 0.703，说明信号不是单纯的近重复记忆；但 3 个数据集仍不是全部 19 个 target。

### 4.4 边界

- **随机 5 折是乐观上界**；变异体高度相关，ESM2 可能记住家族特征；
- Group split 用 KMeans-on-ESM2，不是 MMseqs2 序列身份聚类，仍可能残留近重复；
- 蛋白抗原，不是 IDP 肽；
- 这是“序列模型能预测 AlphaSeq 亲和力”，不是“ECLS 能排序候选”，也不是“结合验证”；
- 没有湿实验。

---

## 5. SAbDab2-Epi：结构面板扩展与肽侧 PCLS

### 5.1 元数据可行性

| 项 | 数 |
|---|---:|
| 复合物 | 6,979 |
| 抗原链 | 8,361 |
| PEPTIDE 链 | 1,111 |
| 5–50 aa 肽链 | 1,043 |
| 唯一 PDB（肽链） | 622 |
| 唯一肽簇 | 213 |
| val/test 合格复合物（H+L+肽） | 420 |
| val/test 唯一 PDB | 261 |

对比现有 ECLS 面板 31 结构/15 簇，**结构面板可扩大约 20 倍（PDB）和 14 倍（簇）**。但这是结构/表位数据，没有 binding 标签；不能直接升级 ECLS 主张。

### 5.2 PCLS 试点

对 30 个 val 复合物（每个肽簇取一个）做肽侧上下文对比：

```
PCLS(s) = NLL_complex(s) − NLL_antibody_stripped(s)
advantage = mean PCLS(shuffled peptide) − PCLS(native peptide)
```

- 复合物状态：H + L + 肽；
- 剥离状态：只有肽；
- ProteinMPNN leave-one-out 序列上下文；
- 100 个组成匹配 shuffle（短肽枚举全部唯一排列，长肽随机采样）。

**结果**：

- 29 个复合物可评分（1 个多字符链 ID 跳过）；
- **25/29 为正（86.2%）**；
- 均值 advantage **0.546**，中位 **0.534**，SD 0.471；
- 4 个负例：1bog（−0.141）、1w72（−0.071）、7bh8（−0.064）、8t6m（−0.201）；
- 正例范围 0.064–1.422。

**解释**：抗体上下文倾向于给 native 肽序列比组成匹配 shuffle 更高的似然。这是**肽侧**的上下文效应，和冻结的 H3 侧 ECLS 是不同估计量；不能写“ECLS 复现”，只能写“PCLS pilot 正信号”。

### 5.2b H3-ECLS pilot（v1.1）

用 4HIX 重链为参考（H3 `VRYDHYSGSSDY`，1-based 96–107），把参考 H3 列通过全局比对转移到每个 SAbDab2-Epi val/test 复合物的重链上，再按冻结 ECLS 定义打分：

```
ECLS(s) = NLL_complex(s) − NLL_antibody_stripped(s)
advantage = mean ECLS(composition-matched shuffled H3) − ECLS(native H3)
```

- 30 个复合物（每个 `cdrh3` 簇取一个）；28 个可评分（2 个 H3 映射失败）；
- **27/28 为正（96.4%）**；
- 均值 advantage **0.660**，中位 **0.607**；
- 唯一负例：1bog（−0.112）；
- H3 长度 5–12 aa；短 H3 的 shuffle 空间有限。

这是**参考对齐 H3**的 pilot，不是 IMGT 编号的确认性复现；它说明冻结 ECLS 的 native-vs-shuffle 方向在一个更大的 val/test 结构面板上也能看到，但还需要 ANARCI/IMGT 编号和全部 420 个复合物才能升级为正式 replication。

### 5.2c ANARCI/IMGT H3-ECLS（v1.2）

HMMER 3.4 通过 WSL + Ubuntu `hmmer` deb 解包运行；ANARCI 对 192 条唯一解析重链做 IMGT 编号，**192/192 成功**。H3 取 IMGT 105–117。420 个合格行中 **408 个可评分**（12 个因超大复合物/300 秒超时跳过）：

| 分层 | n | mean advantage | median | 正比例 | 95% CI |
|---|---:|---:|---:|---:|---|
| **总体** | 408 | **0.672** | 0.638 | **0.980** | [0.640, 0.705] |
| val | 288 | 0.646 | 0.634 | 0.983 | [0.611, 0.680] |
| test | 120 | **0.736** | 0.650 | 0.975 | [0.668, 0.804] |
| **CDRH3 簇级** | 157 | **0.654** | 0.612 | **0.981** | [0.606, 0.703] |
| H3 5–9 aa | 56 | 0.627 | 0.587 | 0.875 | [0.502, 0.754] |
| H3 10–13 aa | 140 | 0.716 | 0.652 | 0.993 | [0.657, 0.774] |
| H3 14–18 aa | 177 | 0.679 | 0.642 | **1.000** | [0.639, 0.719] |
| H3 19–24 aa | 35 | 0.537 | 0.477 | 1.000 | [0.473, 0.602] |

- 408 行来自 **254 个唯一 PDB、157 个唯一 CDRH3 簇**；
- 全部 408 行使用 ANARCI/IMGT 编号（`n_reference=0`）；
- 8 个负例里 7 个是 5–6 aa 短 H3（1bog、1cft、1hh6、6x8p、4jo1 P/Q、4jo2 P），另 1 个是 13 aa（7stf）；
- 短 H3 的 shuffle 空间小、界面贡献弱，是主要噪声源；
- val 与 test 都为正：val 288 行 98.3% 为正，test 120 行 97.5% 为正；
- 这是 SAbDab2-Epi 的 val/test 回顾性结构面板，不是冻结 31/15 的重新评估。

**意义**：在 254 个独立 PDB、157 个 CDRH3 簇上，native H3 相对组成匹配 shuffle 的 ECLS 优势为 0.672，98% 为正，簇级 98.1% 为正。这是目前最强的结构侧 replication；但没有 binding 标签，不能写“ECLS 能预测结合”。

### 5.2d 公开湿实验抗体–肽 Kd 队列（v1.3）

数据来源：Zenodo `10.5281/zenodo.13120765`（Antibody dataset Kd；SAbDab/TDC），493 条抗体–抗原 Kd；其中 **121 条肽抗原（5–50 aa）**，118 条有 SAbDab H/L/antigen 链映射，111 条成功解析为 H/L/肽复合物。Kd 范围 **4×10⁻¹¹ – 2×10⁻⁴ M**；方法：SPR 21、ITC 6、Other 9、Unknown 75。

用 ANARCI/IMGT H3 编号，按冻结 ECLS 定义打分：

| 分层 | n | rho(advantage, logKd) | p |
|---|---:|---:|---:|
| **全部** | 111 | **−0.312** | **0.00085** |
| H3 5–9 | 28 | −0.242 | 0.214 |
| H3 10–13 | 40 | −0.302 | 0.059 |
| **H3 ≥14** | 43 | **−0.456** | **0.0021** |
| H3 ≥10 | 83 | **−0.371** | **0.0006** |
| SPR | 21 | −0.371 | 0.097 |
| ITC | 6 | −0.143 | 0.787 |
| SPR+ITC | 27 | −0.304 | 0.123 |
| 已知方法（非 Unknown） | 36 | −0.251 | 0.140 |
| **抗原级（101 个唯一抗原）** | 101 | **−0.304** | **0.002** |

- Bootstrap 95% CI（111 条）：**[−0.476, −0.128]**；
- 留一法 rho 范围 **−0.340 到 −0.293**，稳定，不是单点驱动；
- 方向一致：advantage 越高，Kd 越低（结合越强）；
- H3 ≥14 最强（−0.456），短 H3 弱（−0.242, ns），与 408 面板的长度分层一致；
- 抗原级聚合仍显著（−0.304, p=0.002），排除了“同一抗原多行”造成的伪独立。

**边界**：
- 这是**回顾性关联**，不是 binding predictor；
- Kd 来自 SAbDab/TDC 策展，方法异质（SPR/ITC/Other/Unknown），无统一 assay；
- 111/121 可解析，10 条因链/肽问题跳过；
- 不是项目自己的 4HIX 候选，不能替代 BLI/SPR；
- 不能写“ECLS 预测亲和力”，只能写“public wet-lab Kd association”。

**意义**：这是目前唯一直接的“ECLS vs 实测 Kd”证据，且长度分层、抗原级聚合、留一法都支持同一方向。它把“没有 binding 标签”从完全空白推进到 **111 个抗体–肽复合物的 exploratory association**；但没有统一实验条件，仍不能升级为确认性 claim。

### 5.2e AbCDR-Binding D44：2,048 个单点突变（v1.4）

数据：Zenodo `10.5281/zenodo.18762978` 的 `d44_set.csv`；2,048 个单点突变（1,198 H + 850 L），亲本 D44.1 Fab–lysozyme 复合物 1MLC（链 B/A/E）。`Binder` 越高 = 结合越强（与 BLOSUM 正相关 rho 0.374 验证方向）。

| 分层 | n | dils rho | dlogp_complex rho | BLOSUM rho | dils 偏相关（控制 BLOSUM） |
|---|---:|---:|---:|---:|---:|
| **全部** | 2,048 | **−0.012 (p=0.585)** | +0.319 (p=1e-49) | **+0.374 (p=4e-69)** | 0.005 |
| **界面 ≤8 Å** | 707 | **+0.119 (p=0.0015)** | +0.214 (p=9e-9) | **+0.381 (p=7e-26)** | **+0.151** |
| 非界面 | 1,341 | **−0.100 (p=2.4e-4)** | +0.415 (p=4e-57) | +0.368 (p=3e-44) | −0.092 |
| 界面 H | 433 | +0.142 (p=0.003) | +0.204 | +0.349 | — |
| 界面 L | 274 | +0.055 (p=0.361) | +0.206 | +0.430 | — |

**解读**：
- `dils` 在**全部 2,048 个单点突变上无信号**（rho −0.012）；
- 只有在**抗原界面 707 个突变**上有小而显著的信号（+0.119），控制 BLOSUM 后只剩 **+0.151** 的局部增量；
- 非界面突变 `dils` 为 **负**（−0.100），说明抗原上下文在远离界面处只是噪声；
- **BLOSUM 在所有分层都强于 `dils`**（+0.374 / +0.381），`dlogp_complex` 也更强；
- 因此：`dils` 不是通用突变效应排序器；AB-bind 的正结果不能外推到“任意单点突变”。它最多是一个**界面局部**的辅助信号。

**边界**：D44 是 anti-lysozyme 单点突变库，Binder 是实验结合分数；不是 IDP/肽；G6 数据集因亲本结构未确认而暂未跑。

### 5.3 边界

- 29 个复合物，无 cluster bootstrap；
- 肽长 5–40 aa，短肽的 shuffle 空间有限（如 5-mer 只有 119 个非天然排列）；
- 只用了 val split，未做 test；
- 没有 H3 编号，所以不是 H3-ECLS；
- 没有 binding/affinity 标签。

---

## 6. 负面与未完成

1. **跨复合物 ranker 失败**：留一复合物 Ridge 方向反转（mean +0.243），不训练全局 ranker。
2. **多突变弱**：AB-bind 多突变 mean −0.104，42.9% 为负；一阶相加不足。
3. **覆盖 86.5%**：编号方案/插入码/删除记法导致 92 行跳过；未强行修补。
4. **AlphaSeq 随机折乐观**：Group split 仍强，但不是序列身份聚类；需 MMseqs2 确认。
5. **SAbDab2-Epi 未做全量 H3-ECLS**：需要 IMGT/Kabat 编号；当前只做肽侧 PCLS 试点。
6. **没有 IDP 肽验证**：AB-bind/AlphaSeq/SAbDab2-Epi 主要是蛋白抗原/一般肽；不能替代无序表位。
7. **没有湿实验**：所有结果都是 retrospective/dry-lab；0 个实测 Kd。

---

## 7. 对项目主张的影响

### 7.1 不改变

- 冻结 ECLS 31/15、0.172281、CI [0.059156, 0.291920] 不变；
- PAE v20/v30 不变；
- “自研结合判别器线关闭”不变；
- 没有 binding/affinity/specificity 主张。

### 7.2 新增的 development 资产

1. **AB-bind mutation-effect benchmark**：首个“同复合物、组成受控、实验 ΔΔG”的 ProteinMPNN 打分测试；`dils` 通过预设门，单突变最强。
2. **AlphaSeq sequence benchmark**：8 数据集/19 target，ESM2 平均 0.642（random-fold）/0.714（group split），可作为序列侧基线。
3. **SAbDab2-Epi 结构面板**：420 个 val/test 合格肽复合物、261 PDB、213 簇，可用于后续 H3-ECLS/PAE 校准。
4. **PCLS pilot**：29 复合物、25 正，肽侧上下文效应正信号。

### 7.3 能修多少

- **不能修 BFN 生成质量**：这些是 scoring/benchmark 数据，不是生成数据。
- **不能修 ipTM**：没有新的结构置信标签。
- **不能替代湿实验**：没有实测 Kd。
- **能修的是评估与排序**：AB-bind 给了一个可复现的 mutation-effect 信号；PCLS 给了一个结构侧复制；AlphaSeq 给了序列侧基线。
- **可能的下一步**：把 AB-bind 的 `dils`/组合作为“单突变候选排序”的 development 工具；把 PCLS 扩到 100+ 复合物；把 AlphaSeq 做 MMseqs2 身份聚类确认。

---

## 8. 下一步（按优先级）

| 优先级 | 事项 | 目的 |
|---|---|---|
| P0 | AlphaSeq 用 MMseqs2 90% identity 聚类后重跑 | 确认 ESM2 信号不是近重复记忆 |
| P1 | SAbDab2-Epi 用 ANARCI/IMGT 编号做 H3-ECLS | 把 29 复合物 PCLS 扩到 H3 侧，检验 31/15 的泛化 |
| P2 | AB-bind 只保留单突变 + 抗体突变，做独立复现 | 最干净的 mutation-effect 子集 |
| P3 | 50 条 4HIX 候选做 BLI/SPR | 唯一能把“设计好不好”变成测量的路径 |
| 不建议 | 在 AB-bind/AlphaSeq 上训练全局 ranker | 留一复合物已失败（方向反转） |

---

## 9. 复现

```bash
# AB-bind
python experiments/hf_aux_benchmark_v1/abbind_v1.py

# AlphaSeq 序列基准（random 5-fold）
python experiments/hf_aux_benchmark_v1/alphaseq_seq_v1.py --max-per-target 20000

# AlphaSeq group-split 敏感性
python experiments/hf_aux_benchmark_v1/alphaseq_group_v1.py

# SAbDab2-Epi 元数据 + PCLS 试点
python experiments/hf_aux_benchmark_v1/sabdab2_epi_pcls_v1.py --limit 30 --shuffles 100
```

产物：

- `results/hf_aux_benchmark_v1/abbind/{scores.csv,per_complex.csv,summary.json,report.md}`
- `results/hf_aux_benchmark_v1/alphaseq/{per_target.csv,summary.json,group_split.csv,group_split_summary.json}`
- `results/hf_aux_benchmark_v1/sabdab2_epi/{pcls_pilot.csv,pcls_summary.json}`

---

## 10. 后续五步执行结果（v1.1，2026-11-11）

| # | 步骤 | 状态 | 结果 |
|---:|---|---|---|
| 1 | AlphaSeq MMseqs2 90% identity 重跑 | **完成** | 3 数据集：composition 0.295、ESM2 **0.703**、组合 0.704；ESM2 仍强 |
| 2 | SAbDab2-Epi H3-ECLS | **完成 408 行 ANARCI/IMGT** | 254 PDB / 157 簇；**400/408 为正（98.0%）**，均值 0.672；簇级 98.1%；val/test 都为正 |
| 2b | 公开湿实验抗体–肽 Kd 队列 | **完成 111 复合物** | advantage vs logKd **rho −0.312, p=0.00085**；H3≥14 −0.456；抗原级 −0.304, p=0.002 |
| 2c | AbCDR-Binding D44 单点突变 | **完成 2,048 突变** | dils 整体 −0.012 (ns)；界面 +0.119；BLOSUM +0.374；dils 非通用排序器 |
| 3 | AB-bind 单突变 + 抗体突变复现 | **完成** | 12 复合物/280 行：dils −0.217，83.3% 为负，CI [−0.375, −0.064]，p=0.001；两个 hash 半区都为负 |
| 4 | 50 条 4HIX 候选 BLI/SPR | **包已备好，未执行** | `experiments/wetlab_bli_v1/`：样本清单、96 孔布局、验收判据、数据模板 |
| 5 | 不训练全局 ranker | **遵守** | 留一复合物 Ridge 方向反转（+0.243），未训练 |

**结论**：5 步里 4 步的计算部分完成，第 4 步受限于湿实验条件，只交付可执行包。H3-ECLS 的 408 行 ANARCI/IMGT（254 PDB / 157 簇，98.0% 为正，簇级 98.1%）是这轮最强的结构侧信号，val 与 test 都为正，但它仍是 SAbDab2-Epi 的回顾性结构面板，不是冻结 31/15 的确认性复现；AlphaSeq 的 MMseqs2 0.703 是序列侧最强信号，但仍是蛋白抗原；AB-bind 单抗体子集是最干净的 mutation-effect 信号。

产物：

- `results/hf_aux_benchmark_v1/alphaseq/mmseqs_split_summary.json`
- `results/hf_aux_benchmark_v1/sabdab2_epi/h3ecls_summary.json`
- `results/hf_aux_benchmark_v1/abbind/single_antibody_summary.json`
- `experiments/wetlab_bli_v1/BLI_EXECUTION_PLAN.md`

---

## 11. 一句话

**这轮 HF 数据把“打分器有没有超出组成的信息”从 5 个复合物/1,284 条变体，扩展到了 18 个复合物/591 条突变、8 个 AlphaSeq 数据集/19 靶点、29 个结构复合物；v1.1 把 AlphaSeq 做到 MMseqs2 90% identity（ESM2 0.703），v1.2 把 H3-ECLS 做到 408 行 ANARCI/IMGT（254 PDB / 157 簇，98.0% 为正，簇级 98.1%），v1.3 找到 111 个公开湿实验抗体–肽 Kd 复合物（ECLS advantage vs logKd rho −0.312，p=0.00085；H3≥14 −0.456；抗原级 −0.304），v1.4 在 AbCDR-Binding D44 的 2,048 个单点突变上发现 `dils` 整体无信号（−0.012），只在界面 +0.119，BLOSUM 始终更强（+0.374）。AB-bind 单抗体子集 12 复合物（dils −0.217，p=0.001）。结论是：H3-ECLS 结构效应、公开 Kd 关联、ESM2 序列信号都为正，但 `dils` 不是通用突变排序器，跨复合物 ranker 失败、多突变弱、没有 IDP、没有项目自己的统一湿实验；第 4 步只交付了可执行的 BLI 包。它提升了评估能力，没有修好生成质量，也没有改变冻结结论。**
