# 论文草案：AlphaFold 派生置信度用于抗体-肽界面设计与排序的系统边界评估

> 状态：草案 v1，未经同行评审。定位：Q2–Q3 方法学/benchmark 论文。

## 一、标题（三个候选，按推荐排序）

1. **"Interface PAE, not ipTM or pLDDT, transfers across independent antibody-peptide scaffolds for candidate ranking"**（直接、可检验、点出核心结论）
2. "Systematic limits of AlphaFold-derived confidence for validating antibody-peptide interface design"（偏边界/警示）
3. "Disorder-aware BFN candidate generation with a transferable interface-PAE ranking signal"（偏方法）

## 二、目标期刊（现实定位）

| 期刊 | 区位 | 定位 |
|---|---|---|
| Briefings in Bioinformatics | Q1（冲） | 方法学综述/评估 |
| Bioinformatics | Q1–Q2（冲） | 方法 + benchmark |
| PLOS Computational Biology | Q1–Q2 | 方法 + 负结果友好 |
| **Protein Science / Proteins** | **Q2（稳）** | 结构 + 方法 |
| Scientific Reports | Q2–Q3（保） | 负结果 + 方法 |
| PLOS ONE | Q3（兜底） | 任何可复现研究 |

**主投建议**：`Bioinformatics` 或 `PLOS Computational Biology`，定位成「benchmark + 方法边界」论文；备选 `Protein Science`。

## 三、摘要（草稿）

> De novo antibody design against disordered targets is limited by validation: AlphaFold-derived confidence is the standard proxy, yet its reliability for antibody-peptide interfaces is rarely tested. We systematically evaluate three AF2 confidence signals (pLDDT, ipTM, interface PAE) across six independent, homology-isolated antibody-peptide scaffolds. We find (i) only interface PAE transfers across scaffolds (median Spearman 0.70–0.87), while pLDDT and ipTM do not; (ii) single-sequence AF2 fails to recapitulate a native crystal complex (ipTM 0.22), and (iii) MSA restores the antibody fold (ipTM 0.75) but cannot discriminate native from scrambled 8-residue peptides via whole-complex ipTM (0.753 vs 0.755). We conclude that interface PAE, after variance-matching calibration, is the only deployment-grade relative ranking signal, and that absolute binding validation requires wet-lab measurements. This defines a concrete, reproducible boundary for AF2-based scoring in antibody-peptide design.

## 四、正文结构

### 1. Introduction
- IDP 靶点的抗体设计困境；验证缺环；AF2 置信度被默认当作打分器。

### 2. Methods
- 2.1 外部 scaffold 校准：三轮预注册发现 + 五轴同源隔离 → 6 独立 scaffold
- 2.2 BFN 候选生成（无序感知，`disorder_guided` 因子）
- 2.3 AF2 协议：单序列（JAX）vs 带 MSA（ColabFold）
- 2.4 界面 PAE 的方差匹配校正

### 3. Results（四张图的核心）
- **R1**：PAE 跨 scaffold 可迁移（Spearman 0.70–0.87）；pLDDT/ipTM 不可（5 方向消融）
- **R2**：单序列 AF2 把真 binder（6YXM 晶体复合物）打成 ipTM 0.22
- **R3**：MSA 修抗体不修短肽（天然 0.753 vs 打乱 0.755）
- **R4**：PAE 校正后 3 seed 过 6 门控，成为部署级排序信号

### 4. Discussion
- AF2 打分的真实边界；对领域（谁在用单序列 AF2 打分）的警示；湿实验是唯一闭环。

## 五、图表计划

- **Fig 1**：6 scaffold 的五轴隔离 + 校准流程
- **Fig 2**：单序列 vs MSA 的真 binder 打分对比（0.22 → 0.75）
- **Fig 3**：天然 vs 打乱肽的 ipTM 无区分（0.753 vs 0.755）
- **Fig 4**：PAE 校正后的 3-seed 门控表
- **Table 1**：三个置信度信号的可迁移性矩阵

## 六、卖点与风险

**卖点**：领域里少见的「AF2 打分边界」系统性负结果 + 一个可复现的校准方法论。审稿人会说「有用」。

**风险**：无湿实验 → 只能停在「方法 + 边界」；审稿人可能问「你的方法到底设计出 binder 了吗」，标准答案只有一句——「这是本文明确划定的边界，湿实验是下一步」。

## 七、支撑材料（仓库内已有、可引用）

- `publication/candidate_interface_pae_deployment_v1.json` — PAE 单轴部署合同（3 seed 过 6 门控）
- `publication/af2_scoring_boundary_final_decision_v1.json` — AF2 打分边界结论
- `publication/final_honest_positioning_v1.json` — 诚实定位
- `publication/idp_platform_capability_v1.json` — 能力清单
- `publication/idp_rd_c1_preregistration.json` — C1 预注册
- `data/candidate_interface_external_calibration_v1/` — 6 独立 scaffold 校准数据
