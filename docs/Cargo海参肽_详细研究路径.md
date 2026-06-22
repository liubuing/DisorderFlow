# Cargo 海参肽：详细研究路径与准备清单

---

## 目录

1. [路线总览：从零到确定 Cargo 序列](#1-路线总览)
2. [Step 1：精读夏老师论文，提取关键信息](#2-step-1精读夏老师论文)
3. [Step 2：海参肽文献系统挖掘](#3-step-2海参肽文献系统挖掘)
4. [Step 3：活性肽序列收集与数据库构建](#4-step-3活性肽序列收集与数据库构建)
5. [Step 4：候选肽的理化性质与成药性评估](#5-step-4候选肽的理化性质与成药性评估)
6. [Step 5：多序列比对与活性基序提取](#6-step-5多序列比对与活性基序提取)
7. [Step 6：Cargo 与 VHH 融合的兼容性预评估](#7-step-6cargo-与-vhh-融合的兼容性预评估)
8. [Step 7：最终 2-3 条 Cargo 确定与参数记录](#8-step-7最终-cargo-确定)
9. [Step 8：Cargo 相关的体外验证设计](#9-step-8cargo-相关的体外验证设计)
10. [准备清单汇总（Checklist）](#10-准备清单汇总)
11. [附录：关键资源与工具](#11-附录关键资源与工具)

---

## 1. 路线总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Cargo 海参肽研究路径全景图                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Step 1          Step 2          Step 3          Step 4                 │
│  精读夏老师论文 → 文献系统挖掘 → 序列收集建库 → 理化性质评估              │
│  (1-3天)         (1-2周)         (3-5天)         (2-3天)                 │
│                                                                         │
│  Step 5          Step 6          Step 7          Step 8                 │
│  活性基序提取 → 融合兼容性评估 → 最终Cargo确定 → 验证方案设计             │
│  (2-3天)         (3-5天)         (1天)           (2-3天)                 │
│                                                                         │
│  ═══════════════════════════════════════════════════════════            │
│  总计：约 4-6 周（与 BFN 设计准备工作并行推进）                            │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.1 核心原则

- **体内活性证据优先**：没有体内数据的肽，序列再漂亮也只作为备选
- **夏老师数据优先**：她手上的未发表数据比你从文献里挖到的更有价值
- **可融合性决定生死**：一个活性再好但不能跟 VHH 融合的肽 = 没用
- **2-3 条平行推进**：不要把鸡蛋放在一个篮子里，至少保证 2 条有体内证据的候选

---

## 2. Step 1：精读夏老师论文（1-3 天）

### 2.1 论文信息

> Xia X, et al. "Sea Cucumber Egg Oligopeptides Ameliorate Cognitive Impairments and Pathology of Alzheimer's Disease Through Regulating HDAC3 and BDNF/NT3 via the Microbiota-Gut-Brain Axis." *Nutrients*, 2025, 17(14), 2312.

### 2.2 带着 15 个问题精读

以下 15 个问题是 cargo 研究路径的起点，每一个都直接决定后续设计：

#### A. 肽的制备与鉴定

| # | 问题 | 为什么重要 | 找不到怎么办 |
|---|------|-----------|-------------|
| 1 | 海参卵用什么酶解的？（胰蛋白酶？碱性蛋白酶？复合酶？） | 酶切位点决定肽段边界，帮你理解序列的 N/C 端特征 | 看 Methods 的 "Preparation of oligopeptides" 小节 |
| 2 | 酶解后有没有做膜分离/色谱分离？截留分子量多少？ | 知道分子量范围，缩小候选肽大小 | 通常 Methods 会写 ultrafiltration 条件 |
| 3 | 有没有做 LC-MS/MS 鉴定？用了什么数据库搜库？ | 这是获取具体肽序列的最直接途径 | 翻 Supplementary Materials，找肽段列表表格 |
| 4 | 鉴定了多少条肽？有没有报道丰度（intensity/spectral count）？ | 高丰度肽 > 低丰度肽作为候选 | 补充材料里的肽段鉴定表 |
| 5 | 鉴定的肽段序列有没有公开在论文/SI 里？ | 直接获取夏老师手上的活性肽序列 | 如果 SI 没有，直接问夏老师要 |

#### B. 活性数据

| # | 问题 | 为什么重要 | 找不到怎么办 |
|---|------|-----------|-------------|
| 6 | 灌胃剂量是多少？mg/kg/day？ | 反推脑内需要达到的有效浓度 | Results 的行为学/生化部分 |
| 7 | **有没有脑内肽浓度数据？** PK 还是脑组织匀浆？ | 这是最关键的问题——知道脑内暴露量才能估算体外实验需要的浓度范围 | 如果没有 PK 数据，这是后续体内实验要补的关键空白 |
| 8 | 海马皮层 Aβ₁₋₄₂ 降低了多少%？vs 模型组 | 量化 cargo 的 Aβ 清除能力 | Results 的 ELISA 部分 |
| 9 | 氧化应激指标（SOD、MDA、GSH）改善了多少？ | 量化 cargo 的抗氧化能力 | Results 的生化指标部分 |
| 10 | HDAC3 是怎么验证的？表达水平下降还是直接结合？ | 如果只是表达水平变化（间接）→ cargo 可能不直接靶向 HDAC3；如果做了 CETSA/SPR（直接结合）→ 多了个验证靶点 | Results 的 mechanism 部分；大概率是间接的（通过菌群-肠-脑轴调控） |

#### C. 机制与安全性

| # | 问题 | 为什么重要 |
|---|------|-----------|
| 11 | 菌群-肠-脑轴的证据链完整吗？（菌群 16S → 代谢物 → 脑） | 理解 cargo 是通过哪条路起效的，决定后续体外实验重点 |
| 12 | 有没有做剂量-效应关系？ED50？ | 确定有效浓度窗口 |
| 13 | 有没有检测血清/脑的肽降解产物？ | 肽在体内的半衰期是 crucial 信息 |
| 14 | 体重、脏器指数的安全性数据？ | Cargo 的安全性基线 |
| 15 | 讨论部分有没有指出具体活性肽段是哪个？或者提到需要进一步鉴定？ | 夏老师自己对"哪个肽是真正起作用的"有没有猜测 |

### 2.3 输出：论文关键信息提取表

读完论文后，输出下面这张表：

```markdown
| 维度 | 论文中报告的数据 | 对 Cargo 设计的含义 |
|------|-----------------|---------------------|
| 原料 | （海参卵/体壁？品种？） | 决定肽序列的来源物种 |
| 酶解条件 | （酶种类、温度、pH、时间） | 预测肽段的 N/C 端氨基酸偏好 |
| 分子量范围 | （<1 kDa / 1-3 kDa / 3-10 kDa？） | 确定 peptide cargo 的长度范围 |
| LC-MS/MS 鉴定条数 | （X 条肽段） | 候选序列池大小 |
| 灌胃剂量 | （mg/kg/day） | 反推脑内所需浓度 |
| 脑 Aβ₁₋₄₂ 降低% | （%） | Cargo 效力量化基线 |
| SOD/MDA/GSH 变化 | （倍数/%变化） | 抗氧化效力量化基线 |
| HDAC3 机制 | （直接/间接？） | 是否增加 HDAC3 作为体外验证靶点 |
| 体内安全性 | （有无异常？） | 安全性基线 |
| 是否公开肽序列 | （是/否，在正文/SI/未公开） | 决定你是直接获得序列还是需要自己推测 |
```

### 2.4 如果论文 SI 中没有肽序列——怎么办？

优先级从高到低：

1. **直接问夏老师要** LC-MS/MS 鉴定结果（最理想）
2. 根据酶解条件 + 海参蛋白数据库 **反推** 可能的肽段（下策，工作量大）
3. 用文献中其他人的海参活性肽序列作为 **替代模板**（最差但能推进）

---

## 3. Step 2：海参肽文献系统挖掘（1-2 周）

### 3.1 检索策略

#### 数据库覆盖

```
PubMed      → 英文主流期刊（Mar. Drugs, J. Funct. Foods, Food Chem. 等）
CNKI/万方   → 中文海参肽研究（国内是主阵地）
Web of Science → 补 PubMed 遗漏 + 查引用关系
Google Scholar → 灰色文献追踪
```

#### 检索式设计

**PubMed（英文）**：

```text
# 核心检索式（AND 组合）
("sea cucumber" OR "Stichopus japonicus" OR "Apostichopus japonicus" 
 OR "Holothuria" OR "Cucumaria" OR "sea cucumber collagen")

AND

("peptide" OR "oligopeptide" OR "hydrolysate" OR "protein hydrolysate")

AND

("Alzheimer" OR "cognitive" OR "neuroprotective" OR "neuroprotection" 
 OR "Aβ" OR "amyloid" OR "antioxidant" OR "oxidative stress" 
 OR "memory" OR "learning" OR "neuroinflammation")
```

```text
# 宽松检索式（扩大范围，然后手动筛选）
("sea cucumber peptide" OR "marine peptide" OR "collagen peptide")
AND
("brain" OR "BBB" OR "neuro*" OR "cognition")
```

```text
# 活性机制检索式
("sea cucumber peptide" OR "collagen peptide" OR "marine oligopeptide")
AND
("Aβ aggregation" OR "acetylcholinesterase" OR "HDAC" 
 OR "Nrf2" OR "NF-κB" OR "BDNF" OR "antioxidant enzyme")
```

**CNKI（中文）**：

```text
# 核心检索
(海参肽 OR 海参胶原肽 OR 海参寡肽 OR 海参酶解物)
AND
(阿尔茨海默 OR 认知 OR 神经保护 OR 抗氧化 OR 学习记忆)

# 扩大范围
(海参 OR 刺参 OR 仿刺参) AND (活性肽 OR 功能肽 OR 生物活性肽)
AND (抗氧化 OR 抗衰老 OR 神经)

# 机制层面
(海参肽 OR 海洋胶原肽) AND (Aβ OR β淀粉样蛋白 OR 氧化应激 OR 自由基)
```

### 3.2 文献筛选流程

```
初步检索（各数据库）
    │
    ├─ 去重（Zotero 自动）
    │
    ├─ 标题+摘要筛选
    │   ├─ 排除：纯食品加工工艺、非海参、无活性数据、会议摘要
    │   ├─ 保留：有活性肽序列信息、有体内/体外活性数据、有 AD 相关
    │   └─ 标记：潜在有用但缺序列 → 后续邮件问通讯作者
    │
    ├─ 全文阅读
    │   ├─ 重点提取：肽序列（最关键）、活性数据、实验模型
    │   └─ 评估：数据质量（n 数、统计、对照是否完整）
    │
    └─ 交叉引用追踪
        ├─ 引文前向追踪（Cited by → 最新进展）
        └─ 引文后向追踪（References → 原始发现）
```

### 3.3 文献提取模板

为每篇关键论文填一张卡：

```markdown
### 论文卡：[第一作者, 年份, 期刊]

**标题**：
**DOI**：

**肽信息**：
- 肽序列：（单字母，N→C）
- 长度：（aa）
- 分子量：（Da）
- 来源：（海参物种 + 组织 + 酶解条件）
- 纯度：（%）

**活性数据**：
| 活性类型 | 实验模型 | 方法 | 有效浓度 | 效果 |
|----------|----------|------|----------|------|
| Aβ 聚集抑制 | （体外/细胞/动物） | （ThT/ELISA/免疫组化） | （IC₅₀ / 剂量） | （%） |
| 抗氧化 | （DPPH/ABTS/FRAP/细胞/动物） | | | |
| 抗炎 | | | | |
| 认知改善 | | | | |

**体内的关键数据**（如果有，这是最高权重）：
- 动物模型：
- 给药途径 + 剂量：
- 脑 Aβ 变化：
- 行为学变化：
- 脑内肽暴露量（如有）：

**安全性**：
**备注**：
```

### 3.4 文献搜索的关键切入点

不要只搜"海参肽+AD"，还要搜这几个方向（它们可能隐含了活性肽序列）：

| 方向 | 为什么搜 | 检索关键词 |
|------|---------|-----------|
| 海参胶原肽的抗氧化 | 抗氧化 = AD 间接相关，且序列常有交集 | `sea cucumber collagen peptide antioxidant` |
| 海洋生物活性肽数据库 | 直接拿别人整理好的数据 | `marine bioactive peptide database` |
| 胶原肽的 ACE 抑制 | 虽然不相关，但存在共同的结构特征（Pro-rich），序列可借鉴 | `collagen peptide ACE inhibitory sequence` |
| 食品来源的抗 Aβ 肽 | 藻类/鱼/贝类的抗 Aβ 肽，跟海参肽可能有结构同源性 | `food-derived peptide Aβ aggregation inhibitor` |
| 肽的 BBB 穿越预测 | 虽然不是海参肽特有，但了解多肽 BBB 穿越的序列偏好的规律 | `peptide BBB penetration prediction sequence` |

---

## 4. Step 3：活性肽序列收集与数据库构建（3-5 天）

### 4.1 序列收集优先级

```
Tier 1（最高优先级，一线候选）：
  ├── 夏老师论文 SI 中鉴定的肽序列（如有）
  ├── 夏老师手上未发表的活性验证肽段
  └── 文献中有体内（灌胃/注射）活性数据的海参肽

Tier 2（次优先级，备选）：
  ├── 文献中体外活性明确但无体内数据的海参肽
  └── 其他海洋来源的抗 Aβ/抗氧化肽（如果海参肽不够）

Tier 3（参考级，不直接使用）：
  ├── 计算机预测的活性肽（无实验验证）
  └── 仅以专利形式报道的海参肽序列
```

### 4.2 构建本地 Cargo 肽数据库

#### 文件结构

```
cargo_database/
├── README.md                          # 数据库说明
├── sequences.fasta                    # 所有候选肽序列（FASTA）
├── sequences.csv                      # 序列 + 元数据（Excel 可读）
├── literature_cards/                  # 每篇论文的提取卡片
│   ├── Xia_2025_Nutrients.md
│   ├── Xu_2022_MarDrugs.md
│   ├── ...
│   └── template.md                    # 空白模板
├── activity_matrix.csv                # 活性矩阵（肽 × 活性类型）
└── notes.md                           # 杂项笔记、未解问题
```

#### `sequences.csv` 的字段设计

```
peptide_id, sequence, length, mw, source_species, source_tissue,
enzyme, enrichment_method,
activity_abeta_ic50, activity_abeta_model,
activity_antioxidant_ic50, activity_antioxidant_method,
activity_cognitive_model, activity_cognitive_route, activity_cognitive_dose,
in_vivo_brain_exposure, in_vivo_pk_data,
in_vivo_safety,
solubility_pred, hydrophobicity, net_charge,
ref_doi, ref_title, notes
```

#### `activity_matrix.csv` 的设计

行 = peptide_id，列 = 活性指标，值 = IC₅₀ 或效果描述

| peptide_id | Aβ聚集抑制 (IC₅₀) | DPPH清除 (IC₅₀) | ABTS清除 (IC₅₀) | 体内认知改善 | 脑Aβ降低 | 文献来源 |
|------------|:-----------------:|:----------------:|:----------------:|:-----------:|:--------:|----------|
| SCP-01 | 25 μM | 50 μM | 40 μM | 有/无 | N/A | PMID_xxx |
| SCP-02 | N/A | 30 μM | 25 μM | N/A | N/A | DOI_xxx |
| ... | | | | | | |

### 4.3 已知海参活性肽的代表性骨架（从文献预整理）

以下是海参及相关海洋胶原肽中已报道的活性代表性骨架——**在使用前必须与夏老师的数据交叉验证**：

#### A. 富含 Gly-Pro-X 的胶原特征肽

| 模板编号 | 序列 | 长度 | 特征 | 文献来源（待核实） |
|----------|------|:----:|------|-------------------|
| COL-01 | GPAGPIGPAG | 10 | Gly-Pro-X 重复，Aβ 聚集抑制 | 海参胶原水解物 |
| COL-02 | GPPGPPGPP | 9 | Gly-Pro-Pro 重复，强抗氧化 | 海参体壁明胶 |
| COL-03 | GPAGPAGPAGPAG | 13 | Gly-Pro-Ala-Gly 重复，双功能 | 仿生设计 |
| COL-04 | GPMGPMGPM | 9 | 含 Met，抗氧化（但Met易氧化） | 鱼胶原 |
| COL-05 | GPAGPQGPAG | 10 | 含 Gln，Aβ 亲和 | 海参胶原 |

#### B. 非胶原来源的海参活性序列

| 模板编号 | 序列 | 长度 | 特征 |
|----------|------|:----:|------|
| NC-01 | （待文献填充） | - | 海参卵特异性肽 |
| NC-02 | （待文献填充） | - | 海参肠肽 |

> **重要**：以上 COL-01~05 是文献中报道过的海参/海洋胶原肽代表性骨架，用于初期设计。最终选择必须基于夏老师的实际数据。如果夏老师手上有明确的体内活性肽序列，优先用她的。

---

## 5. Step 4：候选肽的理化性质与成药性评估（2-3 天）

### 5.1 逐肽计算与打分

对每条候选肽，计算以下参数：

#### 必须计算的基础参数

| 参数 | 工具/方法 | 含义 | 理想范围 |
|------|----------|------|:--------:|
| 分子量 (MW) | Python `Bio.SeqUtils.molecular_weight()` 或 Expasy ProtParam | 决定 cargo 大小 | 500-2500 Da |
| 等电点 (pI) | Expasy ProtParam | 影响溶解性和纯化缓冲液选择 | 3-10（中性附近最好） |
| 净电荷 (pH 7.4) | PepCalc 或 Innovagen | 影响与 VHH/linker 的电荷互作 | -2 到 +2 |
| GRAVY (疏水性) | Expasy ProtParam | 越正越疏水，越可能聚集 | -2.0 到 0 |
| 不稳定指数 | Expasy ProtParam | >40 预测为不稳定 | < 40 |

#### 使用 Expasy ProtParam 的实操

```bash
# 网页版直接提交
https://web.expasy.org/protparam/

# 或者用 Biopython
python3 -c "
from Bio.SeqUtils.ProtParam import ProteinAnalysis
seq = 'GPAGPIGPAG'
p = ProteinAnalysis(seq)
print(f'MW: {p.molecular_weight():.1f} Da')
print(f'pI: {p.isoelectric_point():.1f}')
print(f'GRAVY: {p.gravy():.3f}')
print(f'Instability: {p.instability_index():.1f}')
print(f'Aromaticity: {p.aromaticity():.3f}')
print(f'Flexibility: {p.flexibility()}')
"
```

#### 进阶计算参数（加分项）

| 参数 | 工具 | 为什么需要 |
|------|------|-----------|
| 水溶性预测 | CamSol / SoluProt | 肽 cargo 的可溶表达性 |
| 聚集倾向 | Tango / Aggrescan / Zyggregator | 纯化的肽可能聚集 |
| 螺旋倾向 | AGADIR | 如果 peptide 倾向于形成螺旋，可能影响融合后的构象 |
| 半衰期预测（哺乳动物） | ProtParam N-end rule | 体内稳定性粗略估计 |
| 细胞穿透预测 | CellPPD / CPPpred | 肽自身有无穿膜能力（如果有则加分） |
| BBB 穿越预测 | B3Pred / BBPpred | 游离肽（不依赖 VHH）的 BBB 穿越潜力 |

### 5.2 排除标准（一票否决）

以下任何一种情况，直接排除或降到 Tier 3：

| 条件 | 理由 |
|------|------|
| 序列中含 ≥3 个连续疏水残基（如 LLL、VIV、FFF） | 聚集风险极高 |
| Cys 残基（非成对） | 会形成非天然分子间二硫键，纯化噩梦 |
| Met 含量 >20%（每 5 个残基中有 1 个 Met） | 氧化变质，活性不稳定 |
| 净电荷 > |+5| 或 <-5 | 强电荷可能干扰 VHH 折叠或 TfR 结合 |
| GRAVY > 1.0（非常疏水） | 不溶或聚集 |
| 序列包含已知蛋白酶切位点（如 K/R-X-X-S/T 等 furin 位点） | 在血清/体内快速降解 |
| N 端 = Gln（Q） | N 端 Gln 自发环化为 pyroGlu，导致异质性 |
| 存在 N-糖基化基序 N-X-S/T（X ≠ P） | 在大肠杆菌中不糖基化，但后续若有哺乳动物表达计划会出问题 |

### 5.3 输出：候选肽理化性质评分表

```markdown
| 肽 ID | 序列 | MW | pI | GRAVY | 净电荷(pH7) | 不稳定指数 | 水溶性预测 | 聚集倾向 | 排除风险 | 综合评分 |
|-------|------|:--:|:--:|:-----:|:-----------:|:----------:|:----------:|:--------:|:--------:|:--------:|
| SCP-01 | GPAGPIGPAG | 850 | 5.5 | -0.5 | 0 | 35 | 良 | 低 | 无 | 8/10 |
| ... | | | | | | | | | | |
```

---

## 6. Step 5：多序列比对与活性基序提取（2-3 天）

### 6.1 目的

从多条活性海参肽中找出一致的结构特征/基序，帮助：
1. 确认活性是由序列还是由组成（氨基酸比例）决定的
2. 指导 cargo 序列截断/延长（如果原始肽太长/太短）
3. 理解哪几个残基是"不能动的"（保守的）

### 6.2 比对策略

```
收集的 Tier 1 + Tier 2 肽序列
    │
    ├─ 按活性类型分组
    │   ├── Group A：Aβ 聚集抑制为主的肽
    │   ├── Group B：抗氧化为主的肽
    │   └── Group C：双功能肽
    │
    ├─ 每组内做多序列比对 (Clustal Omega / MAFFT)
    │   └── 短的肽（<15 aa）：加 gap opening penalty 降低
    │
    └─ 生成 SeqLogo → 可视化保守位点
```

### 6.3 实操命令

```bash
# 安装工具（如果没有）
pip install biopython

# MAFFT 多序列比对
mafft --auto cargo_candidates.fasta > cargo_aligned.fasta

# 或者用 Clustal Omega（在线或本地）
clustalo -i cargo_candidates.fasta -o cargo_aligned.fasta --outfmt=fasta

# 生成 SeqLogo（需要 R 或 WebLogo 在线版）
# 在线：http://weblogo.threeplusone.com/
# 或本地：
weblogo -f cargo_aligned.fasta -o logo.png -F png
```

### 6.4 分析输出

对每个活性组，回答以下问题：

| 问题 | 判断标准 | 对设计的含义 |
|------|---------|-------------|
| 有保守基序吗？ | ≥3 个连续位置有相同的氨基酸或高度相似的理化性质 | 保守区 = 活性必需区，融合时不要截掉 |
| Gly 和 Pro 的分布模式？ | 间隔规律（如 G-X-Y 或 G-P-X） | 胶原特征，提示结构偏向 PPII 螺旋 |
| 活性是否依赖于 N/C 端自由？ | 文献中截断/N/C 端修饰对活性的影响 | 决定 N 端还是 C 端融合到 VHH |
| 有没有必需的带电残基？ | 保守的 Arg/Lys/Asp/Glu 位点 | 这些位点可能参与 Aβ 的静电互作或抗氧化电子转移 |
| 芳香族残基（Phe/Tyr/Trp）的分布？ | 位置 + 是否保守 | 芳香族参与 Aβ 的 π-π stacking 抑制纤维化 |

### 6.5 基序提取输出格式

```markdown
### 基序分析结果

#### 基序 1：Aβ 聚集抑制基序
- 保守序列模式：G-P-[AG]-[GP]-[IL]-G-P
- 出现频率：4/6 条 Aβ 活性肽中
- 推测作用：Gly-Pro 骨架提供刚性 kink，疏水 Ile/Leu 插入 Aβ 的疏水 groove
- 设计约束：融合时保留该基序的完整性

#### 基序 2：抗氧化基序
- 保守序列模式：[GP]-P-[GA]-P-P
- 出现频率：5/7 条抗氧化肽中
- 推测作用：Pro 残基的吡咯烷环可作为自由基陷阱
- 设计约束：不可截短

#### 灵活区域
- N 端 1-2 残基：不保守，可替换
- C 端 1-2 残基：不保守，可作为 linker 融合点
```

---

## 7. Step 6：Cargo 与 VHH 融合的兼容性预评估（3-5 天）

### 7.1 评估维度

这是 cargo 设计中最容易出问题的环节。分 5 个维度逐一检查：

#### 维度 1：空间位阻

**要回答的问题**：Cargo 肽会不会挡住 VHH 的 CDR 区域？

```
操作：
1. 用 AntiFold/AlphaFold2 预测 VHH-cargo 融合蛋白结构
2. 测量 cargo 的质心到 CDR3 的距离
3. 如果 cargo 折叠回来靠近 CDR → 换方向或加长 linker
```

#### 维度 2：Cargo 暴露度（SASA）

**要回答的问题**：Cargo 肽融合后还暴露在溶剂中吗？能接触 Aβ/自由基吗？

```
用 PyMOL 或 freesasa 计算：
- cargo 肽在游离态的相对 SASA
- cargo 肽在融合蛋白中的相对 SASA
- 如果相对 SASA < 30% → cargo 可能被埋藏，活性丧失
- 理想：相对 SASA > 50%
```

```bash
# freesasa 计算
pip install freesasa
python3 -c "
import freesasa
result = freesasa.calc('fusion_protein.pdb')
print(f'Total SASA: {result.totalArea():.1f} A2')
# 需要写脚本提取特定残基范围的 SASA
"
```

#### 维度 3：电荷互补/排斥

**要回答的问题**：Cargo 和 VHH 表面的静电荷会不会互斥（影响折叠）或互相吸附（导致聚集）？

```
1. 用 APBS/PDB2PQR 计算 VHH 表面静电势
2. 计算 cargo 肽在 pH 7.4 的净电荷
3. 如果 VHH CDR 区域带正电 + cargo 带正电 → 排斥 → 可能影响折叠
4. 如果 VHH 表面有大片疏水区 + cargo 疏水 → 可能非特异性吸附
```

#### 维度 4：Linker 对 Cargo 活性的影响

**要回答的问题**：linker 会不会干扰 cargo 的活性基序？

特别注意：
- (GGGGS)n linker 的柔性可能会让 cargo 回折靠近 VHH
- 如果 cargo 的活性依赖于自由 N 端 → 只能用 C 端融合（方案 A：VHH-linker-cargo）
- 如果 cargo 的活性依赖于自由 C 端 → 只能用 N 端融合（方案 B：cargo-linker-VHH）

**实验验证策略**：如果文献中该肽有 N/C 端修饰对活性影响的报道，直接参考。如果没有，默认先做方案 A（VHH-linker-cargo，C 端融合），因为 VHH 的 N 端自由对 TfR 结合更重要。

#### 维度 5：融合蛋白整体稳定性

**要回答的问题**：加了 cargo 后，整个融合蛋白会不会变得不稳定？

```
1. AntiFold/ESMFold 预测融合蛋白 pLDDT
2. 对比：
   - 单独 VHH 的 pLDDT
   - 融合蛋白的 VHH 区域的 pLDDT
   - 融合蛋白的 cargo 区域的 pLDDT
3. 如果融合后 VHH 的 pLDDT 下降 > 5 分 → cargo 干扰了 VHH 折叠 → 换 cargo 或加长 linker
4. 预测融合蛋白 Tm > 50°C
```

### 7.2 输出：Cargo 融合兼容性评分卡

```markdown
| 维度 | 评估结果 | 评分 (1-5) | 问题/备注 |
|------|---------|:----------:|-----------|
| 空间位阻 | cargo 不靠近 CDR 区域 | 5 | 构象自由 |
| Cargo 暴露度 | 相对 SASA = 65% | 5 | 完全暴露 |
| 电荷互补 | VHH 净电荷 -1, cargo 0 | 5 | 无冲突 |
| Linker 兼容 | Cargo 活性不依赖自由端 | 4 | 方案 A OK |
| 整体稳定性 | 融合后 pLDDT 从 88 → 86 | 4 | 轻微下降，可接受 |
| **综合** | | **4.6/5** | **适合作为候选** |
```

---

## 8. Step 7：最终 2-3 条 Cargo 确定（1 天）

### 8.1 最终筛选矩阵

为每条通过前面所有筛选的候选肽打分：

| 权重因子 | 权重 | SCP-01 | SCP-02 | SCP-03 |
|----------|:----:|:------:|:------:|:------:|
| 体内活性证据 | 30% | 8 | 5 | 7 |
| 体外 Aβ 抑制活性 | 20% | 8 | 3 | 7 |
| 抗氧化活性 | 15% | 3 | 9 | 6 |
| 理化性质（可溶/稳定） | 15% | 9 | 8 | 8 |
| 融合兼容性 | 10% | 8 | 7 | 8 |
| 文献支撑力度 | 5% | 7 | 6 | 6 |
| 序列新颖性（专利空间） | 5% | 7 | 5 | 8 |
| **加权总分** | **100%** | **7.1** | **5.8** | **7.0** |

### 8.2 最终 Cargo 确定清单

确定最终 2-3 条 cargo 后，为每条准备以下完整档案：

```markdown
### Cargo 终选档案：SCP-XX

**基本信息**：
- 序列（单字母，N→C）：
- 长度：
- 分子量：
- 来源物种/组织：
- 酶解条件：

**活性证据汇总**：
| 活性类型 | 模型 | 有效浓度 | 效果 | 证据等级 |
|----------|------|----------|------|:--------:|
| Aβ 聚集抑制 | | | | |
| 抗氧化 | | | | |
| 体内认知改善 | | | | |

**理化参数**：
- MW: ___ Da | pI: ___ | GRAVY: ___ | 净电荷(pH7.4): ___ | 不稳定指数: ___

**融合设计**：
- 推荐融合方向：（N-VHH-Cargo / N-Cargo-VHH）
- 推荐 linker：（长度 + 序列）
- 保留的活性关键残基：（标注不可截短/不可突变的位置）

**风险评估**：
- 主要风险：
- 应对策略：
```

### 8.3 备选与淘汰记录

不要删除被淘汰的候选。保留它们并标注原因——答辩的时候"我评估了 X 条序列，最终选了这 3 条，淘汰 Y 的原因是……"是很好的叙事。

---

## 9. Step 8：Cargo 相关的体外验证设计（2-3 天）

### 9.1 针对 Cargo 的特异性验证

除了研究方案中已有的 SPR/BBB/Aβ 抑制，你还需要针对 cargo 的特点设计这些验证：

#### 验证 1：Cargo 肽单独 vs 融合蛋白的活性对比

**目的**：证明融合 didn't kill the cargo activity

| 样品 | 实验 |
|------|------|
| 游离 cargo 肽（化学合成） | ThT Aβ 抑制 IC₅₀ |
| VHH-linker-cargo 融合蛋白 | ThT Aβ 抑制 IC₅₀ |
| VHH-only（无 cargo） | ThT Aβ 抑制 IC₅₀（应该是阴性） |

**预期**：融合蛋白的 IC₅₀ 在游离肽的 2-5 倍以内（linker 和 VHH 的空间影响可接受）

#### 验证 2：Cargo 在血清中的稳定性

**目的**：知道 cargo 在体内能活多久

```
操作：
1. 融合蛋白 + 小鼠/人血清（37°C）
2. 时间点：0, 0.5, 1, 2, 4, 8, 24h
3. Western blot (anti-His) 检测完整蛋白比例
4. 如果 < 4h 降解 >50% → 考虑序列优化（去蛋白酶切位点）
```

#### 验证 3：Cargo 活性是否依赖于自由 N 端/C 端

**目的**：验证融合方向选择的正确性

```
操作：
1. 合成 N-acetylated peptide（N 端乙酰化模拟融合封闭 N 端）
2. 合成 C-amidated peptide（C 端酰胺化模拟融合封闭 C 端）
3. vs 游离肽（自由两端）
4. 测试三个版本的 Aβ 抑制 IC₅₀
5. 如果 N-acetyl 的 IC₅₀ 大幅上升 → N 端是关键 → 只能用 C 端融合
```

#### 验证 4：Cargo 的 Aβ 结合模式（加分项）

**目的**：理解 cargo 是怎么抑制 Aβ 聚集的——为论文讨论部分提供分子机制

```
可选方法（按复杂度排序）：
1. ThT 荧光 + TEM/AFM 成像：看到底是抑制纤维成核还是延长
2. 圆二色光谱（CD）：监测 cargo 加入后 Aβ 二级结构转变
3. 分子对接：cargo 肽 vs Aβ₁₋₄₂ 纤维结构 (PDB: 2BEG/5OQV)
4. NMR（HSQC）：¹⁵N 标记 Aβ + cargo 滴定 → 找到 cargo 在 Aβ 上的结合位点
```

### 9.2 Cargo 验证的时间安排

```
体外验证中 Cargo 相关的部分应安排在：
  ├── 得到纯化的融合蛋白后 → 立刻做验证 1（活性对比）
  ├── 拿到 2-3 批纯化蛋白后 → 验证 2（血清稳定性）
  └── 如果融合方向有纠结 → 先做验证 3（N/C 端依赖），再最终确定融合方案
```

---

## 10. 准备清单汇总

### 10.1 即刻可以开始做的（不需要等任何人）

- [ ] **精读夏老师 Nutrients 2025 论文**，填写 2.3 节的"论文关键信息提取表"
- [ ] **查该论文的 Supplementary Materials**，找 LC-MS/MS 肽段鉴定表
- [ ] **PubMed/CNKI 系统检索**，按 3.1 节的检索式执行
- [ ] **安装文献管理工具 Zotero**，建立 `Cargo_海参肽` 文件夹
- [ ] **准备文献提取模板**（复制 3.3 节），每读一篇填一张
- [ ] **安装 cargo 计算环境**：`pip install biopython freesasa`
- [ ] **熟悉 Expasy ProtParam**：随便丢几条已知肽序列进去看输出
- [ ] **浏览 SAbDab 数据库**，搜"nanobody"或"TfR"，对 VHH 结构有个感性认识

### 10.2 需要跟夏老师确认的（尽早问）

- [ ] **核心问题**：Nutrients 论文中的活性肽序列是否已通过 LC-MS/MS 鉴定？序列能否分享？
- [ ] 海参卵寡肽的酶解条件详情（酶种类、酶用量、温度、pH、时间）
- [ ] 有没有脑组织肽浓度数据？（灌胃后脑匀浆的 LC-MS）
- [ ] 有没有正在进行的海参肽 AD 相关实验？有没有未发表数据可以参考？
- [ ] 表达纯化方面：实验室有哪些表达菌株？有没有现成的 pET 载体？
- [ ] 体外实验：有没有 SPR/BLI 设备？能不能用 hCMEC/D3 细胞？
- [ ] 经费方面：基因合成的预算（2-3 条序列，约 2000-4000 元）
- [ ] 专利方面：实验室/学校有没有专利申请流程和费用支持？

### 10.3 配套工具准备

- [ ] **PyMOL 安装**（教育版免费）：`conda install -c schrodinger pymol` 或 `dnf install pymol`
- [ ] **Zotero 安装**：`dnf install zotero` 或网页版
- [ ] **Clustal Omega**：`dnf install clustal-omega` 或用 EMBL-EBI 在线版
- [ ] **MAFFT**：`dnf install mafft`
- [ ] **WebLogo**：在线版 webLogo 3（http://weblogo.threeplusone.com/）
- [ ] **Python 包清单**：
  ```bash
  pip install biopython         # 序列操作
  pip install freesasa          # SASA 计算
  pip install matplotlib        # 画图
  pip install pandas            # 数据整理
  pip install seaborn           # 热图（活性矩阵可视化）
  ```

### 10.4 数据库与在线工具书签

| 工具 | 网址 | 用途 |
|------|------|------|
| Expasy ProtParam | https://web.expasy.org/protparam/ | 肽理化参数 |
| PepCalc | https://pepcalc.com/ | 肽净电荷、溶解性 |
| CamSol | https://www-cohsoftware.ch.cam.ac.uk/index.php/camsol | 蛋白水溶性预测 |
| Tango | http://tango.crg.es/ | 聚集倾向 |
| Aggrescan | http://bioinf.uab.es/aggrescan/ | 聚集倾向 |
| AGADIR | http://agadir.crg.es/ | 螺旋倾向 |
| CellPPD | https://webs.iiitd.edu.in/raghava/cellppd/ | 细胞穿透肽预测 |
| freesasa | https://freesasa.github.io/ | SASA 计算 |
| ProtParam N-end rule | https://web.expasy.org/protparam/ | N 端半衰期预测 |
| IMGT | http://www.imgt.org/ | VHH 序列编号 |
| SAbDab | https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/ | 抗体结构数据库 |
| WebLogo 3 | http://weblogo.threeplusone.com/ | 序列保守性可视化 |

---

## 11. 附录：关键资源与工具

### A. 快速上手：用 Python 一次性评估多条候选肽

```python
#!/usr/bin/env python3
"""cargo_peptide_evaluator.py —— 批量评估候选 cargo 肽的理化性质"""

from Bio.SeqUtils.ProtParam import ProteinAnalysis
import csv

# 候选肽列表（手动维护）
candidates = {
    "SCP-01": "GPAGPIGPAG",
    "SCP-02": "GPPGPPGPP",
    "SCP-03": "GPAGPAGPAGPAG",
    # 从文献中收集到的新序列追加在这里
}

# 排除规则
EXCLUDE_MOTIFS = {
    "Cys_free": "C",
    "oxidizable_Met_cluster": "MM",
    "N_glycosylation": "N[^P][ST]",
    "furin_cleavage": "R[^P]{2}[RK][^P]{1}[ST]",
}

def evaluate_peptide(seq_id, sequence):
    """评估单条肽，返回字典"""
    seq = sequence.upper().strip()
    pa = ProteinAnalysis(seq)
    
    # 基础参数
    mw = pa.molecular_weight()
    pI = pa.isoelectric_point()
    gravy = pa.gravy()
    instability = pa.instability_index()
    aromaticity = pa.aromaticity()
    
    # 氨基酸组成
    aa_composition = pa.count_amino_acids()
    pro_content = aa_composition.get("P", 0) / len(seq) * 100
    gly_content = aa_composition.get("G", 0) / len(seq) * 100
    
    # 净电荷估算 (pH 7.4, 简化版)
    n_neg = aa_composition.get("D", 0) + aa_composition.get("E", 0)
    n_pos = aa_composition.get("R", 0) + aa_composition.get("K", 0) + aa_composition.get("H", 0)
    # His 在 pH 7.4 时约 10% 质子化
    net_charge = n_pos - n_neg + 0.1 * aa_composition.get("H", 0)  
    
    # 排除检查
    flags = []
    if "C" in seq:
        flags.append("CYS_FREE")
    if seq.count("M") / len(seq) > 0.2:
        flags.append("MET_RICH")
    if "Q" == seq[0]:
        flags.append("NTERM_GLN")
    if abs(net_charge) > 4:
        flags.append("HIGH_CHARGE")
    if gravy > 1.0:
        flags.append("HIGHLY_HYDROPHOBIC")
    if instability > 40:
        flags.append("UNSTABLE")
    
    return {
        "id": seq_id,
        "seq": seq,
        "len": len(seq),
        "MW": round(mw, 1),
        "pI": round(pI, 2),
        "GRAVY": round(gravy, 3),
        "net_charge_pH7.4": round(net_charge, 1),
        "instability_idx": round(instability, 1),
        "aromaticity": round(aromaticity, 3),
        "pro%": round(pro_content, 1),
        "gly%": round(gly_content, 1),
        "flags": ",".join(flags) if flags else "PASS",
        "score": 5 - len(flags),  # 5 分满分，每个 flag 扣 1 分
    }


if __name__ == "__main__":
    results = []
    for sid, seq in candidates.items():
        results.append(evaluate_peptide(sid, seq))
    
    # 打印表格
    header = ["id", "seq", "len", "MW", "pI", "GRAVY", "net_charge_pH7.4",
              "instability_idx", "pro%", "gly%", "flags", "score"]
    print("\t".join(header))
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        print("\t".join(str(r[h]) for h in header))
    
    # 导出 CSV
    with open("cargo_evaluation.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n结果已导出到 cargo_evaluation.csv")
```

### B. Cargo 研究路径决策树

```
                        夏老师手上有没有明确的体内活性肽序列？
                               /              \
                             有                没有
                             │                  │
                    直接获得序列              文献系统挖掘
                    作为 Tier 1                ↓
                             │          找到了 ≥ 3 条有体内证据的？
                             │              /            \
                    Step 4-5-6-7           有              没有
                    理化+基序+融合           │               │
                             │         选 2-3 条       退而求其次：
                             │         Tier 1           用体外活性肽 +
                             │              │           理论预测 +
                             │         Step 4-5-6-7     跟夏老师讨论
                             │              │           是否可以做体内
                             ↓              ↓               ↓
                        确定 2-3 条最终 Cargo ←─────────────┘
                             │
                        进入 BFN 融合蛋白设计阶段
```

### C. 时间估算总表

| 步骤 | 内容 | 纯工作时间 | 日历时间（含等待） |
|------|------|:----------:|:-----------------:|
| Step 1 | 精读夏老师论文 + 提取信息 | 1-2 天 | 3 天 |
| Step 2 | 文献系统检索与筛选 | 3-5 天 | 2 周（需要消化） |
| Step 3 | 序列收集 + 建库 | 2-3 天 | 5 天 |
| Step 4 | 理化性质评估 | 1-2 天 | 3 天 |
| Step 5 | 多序列比对 + 基序分析 | 1-2 天 | 3 天 |
| Step 6 | 融合兼容性预评估 | 2-3 天 | 5 天 |
| Step 7 | 最终确定 | 0.5 天 | 1 天 |
| Step 8 | 验证方案设计 | 1-2 天 | 3 天 |
| **合计** | | **12-20 天** | **约 5-6 周** |

> 注：Steps 1-4 可与 BFN VHH 设计准备（研究流程完整版中的 Phase 0）并行推进。

---

*文档版本：v1.0 | 生成日期：2026-05-27 | 专用于 Cargo 海参肽研究路径*
