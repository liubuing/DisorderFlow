# 基于 BFN 计算设计的 TfR 靶向纳米抗体-海参肽融合蛋白用于阿尔茨海默病 BBB 递送的研究方案

---

## 目录

1. [项目基本信息](#1-项目基本信息)
2. [开题报告](#2-开题报告)
3. [研究流程](#3-研究流程)
4. [中期答辩框架](#4-中期答辩框架)
5. [预期成果与交付物](#5-预期成果与交付物)
6. [风险与应对](#6-风险与应对)
7. [参考文献](#7-参考文献)

---

## 1. 项目基本信息

| 项目 | 内容 |
|------|------|
| **课题名称** | 基于 BFN 计算设计的 TfR 靶向纳米抗体-海参肽融合蛋白的构建及其穿越血脑屏障治疗阿尔茨海默病的研究 |
| **研究类型** | 应用基础研究（计算生物学 + 蛋白质工程 + 神经药理学交叉） |
| **核心假说** | TfR 纳米抗体穿梭体可将海参活性肽高效递送入脑，实现 Aβ 聚集抑制 + 抗氧化双功能治疗 |
| **技术平台** | BFN (Bayesian Flow Network) 蛋白质从头设计 |

---

## 2. 开题报告

### 2.1 研究背景

#### 2.1.1 阿尔茨海默病治疗困境

阿尔茨海默病（AD）是全球最常见的神经退行性疾病，患者超 5500 万，中国占约 1000 万。现有治疗面临两大瓶颈：

1. **血脑屏障（BBB）阻隔**：>98% 的小分子药物和几乎 100% 的大分子生物药无法有效入脑
2. **靶点单一**：已获批的 Aβ 抗体（Aducanumab、Lecanemab、Donanemab）仅靶向 Aβ 清除，忽略氧化应激、tau 病理等并行通路

#### 2.1.2 TfR 介导的 BBB 递送

转铁蛋白受体 1（TfR1）在脑微血管内皮细胞管腔侧高表达，介导转铁蛋白的铁递送。利用 TfR1 的受体介导转胞吞（RMT）机制可实现大分子药物 BBB 穿越。

**关键里程碑**：
- **2021**：JCR Pharmaceuticals 的 Izcargo（抗 TfR 抗体-酶融合）在日本获批，首个上市的 TfR BBB 递送药物
- **Denali Therapeutics**：TfR 结合 Fc 工程化"Transport Vehicle"平台，DNL310 进入临床 II/III 期
- **Roche**："Brain Shuttle"双特异性抗体进入临床

这些成功案例证明了 TfR 作为 BBB 穿梭靶点的可行性和安全性。

#### 2.1.3 纳米抗体作为 BBB 穿梭体的优势

与传统单克隆抗体（~150 kDa）相比，纳米抗体（VHH, ~15 kDa）具备独特优势：

| 特性 | 单抗 | 纳米抗体 |
|------|:----:|:--------:|
| 分子量 | ~150 kDa | ~15 kDa |
| 组织穿透 | 有限 | 优异 |
| 生产 | 哺乳动物细胞 | 大肠杆菌/酵母 |
| 稳定性 | 一般 | 高热/化学稳定性 |
| 免疫原性 | 低（人源化） | 低（可人源化） |
| 工程化融合 | 复杂 | 简单（单链融合） |

近期突破性进展：
- **NewroBus（Rutgers, 2025）**：470 条 VHH 序列 → 8 个优化人源化 lead，在人源化 TfR 大鼠中证实高效 BBB 穿越且无血液毒性
- **BBB00533（KU Leuven, 2023）**：人/猴交叉反应 VHH，BACE1 融合降低脑 Aβ ~40%

#### 2.1.4 海参肽的神经保护作用

海参（_Stichopus japonicus_ / _Apostichopus japonicus_）胶原肽是海参体壁酶解产物，分子量 500-3000 Da（3-25 个氨基酸），文献报道其具备：

- **抗氧化活性**：清除 ROS，上调 Nrf2/HO-1 通路
- **Aβ 聚集抑制**：体外破坏 Aβ₁₋₄₂ 纤维形成
- **抗神经炎症**：抑制小胶质细胞的 NF-κB 活化
- **体内证据**：AD 模型动物中改善认知功能、降低脑内 Aβ 沉积

**核心问题**：海参肽作为大分子肽类，本身就难以穿越 BBB。游离肽的脑内暴露量极低，严重限制其治疗效果。将其作为 TfR 纳米抗体的 Cargo 融合表达，有望解决这一核心瓶颈。

#### 2.1.5 BFN 计算设计平台

BFN（Bayesian Flow Network）是一种生成式蛋白质设计模型，本项目组已建立完整的计算设计管线（V10 架构），核心能力包括：

- 给定靶点结构/序列约束，从头设计高亲和力结合蛋白
- 抗体 CDR 环区设计优化
- 蛋白质稳定性/可表达性预测

### 2.2 国内外研究现状

#### 2.2.1 TfR 穿梭技术

- **已上市**：Izcargo（日本，2021），首个 TfR BBB 递送药物
- **临床阶段**：Denali DNL310（全球）、Roche Brain Shuttle（全球）
- **临床前**：
  - NewroBus 纳米抗体平台（Rutgers/NanoNewron，2025）
  - BBB00533 纳米抗体（KU Leuven/VIB，2023）
  - Vect-Horus VHH 专利（US2022/0090050）

**国内现状**：中国在 TfR BBB 递送领域以基础研究为主，尚无本土产品进入临床。这为本项目的差异化竞争提供了窗口。

#### 2.2.2 海参肽 AD 研究

- 以中国、日本、韩国研究为主，主要集中在食品科学与中药药理领域
- 大多数研究停留于体外实验和大鼠/小鼠模型，缺乏递送系统
- **尚未有人将海参肽与 BBB 穿梭体结合**

#### 2.2.3 本课题定位

| 维度 | 现有研究 | 本项目 |
|------|---------|--------|
| BBB 穿梭体 | TfR 单抗/Fc 融合 | **TfR VHH 纳米抗体**（BFN 计算设计） |
| 治疗 Cargo | 酶/TNFα 抑制剂/BACE1 | **海参活性肽**（天然来源，双功能） |
| 设计方法 | 免疫+淘选 | **BFN 从头计算设计** |
| 技术路线 | 抗体工程 | **计算设计 + 蛋白质工程 + 神经药理学** |

### 2.3 研究目标

**总体目标**：构建并验证一种基于 BFN 计算设计的 TfR 靶向 VHH-海参肽融合蛋白，实现高效 BBB 穿越和 AD 双重治疗功能。

**具体目标**：

1. 从公开的海参活性肽序列中确定 2-3 条候选 cargo 序列
2. 基于已报道的 TfR VHH 序列和结构模板，使用 BFN 优化设计 2 条 VHH 骨架
3. 构建 VHH-海参肽融合蛋白表达体系，完成大肠杆菌可溶表达和纯化
4. 体外验证：TfR 结合亲和力、BBB 细胞模型穿越效率、Aβ 聚集抑制活性、抗氧化活性
5. 体内验证（可选/后期）：AD 模型小鼠的脑部递送效率、药效学评估

### 2.4 创新点

1. **Cargo 创新**：首次将海参活性肽作为 TfR BBB 穿梭体的治疗性载荷，Cargo 本身即具备抗 Aβ 聚集和抗氧化双功能
2. **设计方法创新**：利用 BFN 深度学习模型从头设计/优化 TfR 靶向 VHH，替代传统骆驼免疫+噬菌体淘选的漫长流程
3. **双功能单链融合**：一个融合蛋白同时解决 BBB 穿越 + AD 病理双重问题，结构简单，适合原核表达
4. **专利空白**：TfR VHH + 天然肽类 cargo 作为 BBB 递送载体尚未被专利覆盖

### 2.5 技术路线

```
Phase 1: 序列准备（1 个月）
├── 文献挖掘：确定活性海参肽序列（2-3 条，8-25 aa）
├── TfR VHH 模板筛选：从公开专利/文献提取 3-5 条 VHH 序列
└── TfR 胞外域结构获取（PDB 已知结构）

Phase 2: BFN 计算设计（1.5 个月）
├── VHH 骨架优化：以公开序列为模板，BFN 优化 CDR 环区
├── VHH-肽融合建模：评估不同 linker 长度和融合方向
├── 稳定性评估：预测融合蛋白的 Tm、可溶表达概率
└── 分子对接验证：VHH-TfR 对接评估结合模式

Phase 3: 表达纯化（2 个月）
├── 基因合成（大肠杆菌密码子优化）
├── 小量表达筛选（5-10 个候选，BL21(DE3)/SHuffle）
├── Ni-NTA + SEC 纯化
└── SDS-PAGE / SEC-HPLC / CD 光谱鉴定

Phase 4: 体外验证（2 个月）
├── SPR/BLI：TfR 胞外域结合亲和力
├── BBB 细胞模型：hCMEC/D3 单层穿越效率
├── ThT 荧光：Aβ₁₋₄₂ 聚集抑制 IC₅₀
├── DPPH/ABTS：抗氧化活性
└── MTT：细胞毒性评估

Phase 5: 体内验证（3 个月，可选/后期）
├── 药代动力学：小鼠尾静脉注射后脑/血浆分布
├── AD 模型药效：APP/PS1 小鼠，Morris 水迷宫 + 脑 Aβ ELISA
└── 安全性初步：血常规、肝肾功、脑组织 HE 染色
```

### 2.6 预期结果

1. 2-3 条 BFN 优化后的抗 TfR VHH 序列（Kd < 100 nM）
2. 1-2 个 VHH-海参肽融合蛋白（表达量 > 10 mg/L 大肠杆菌培养物）
3. 体外：TfR 高亲和力结合 + BBB 模型穿越效率 > 5% + Aβ 聚集抑制 IC₅₀ < 50 μM
4. 体内（如实施）：脑内可检测融合蛋白，AD 模型认知改善

---

## 3. 研究流程（详细操作）

### Phase 1: 序列准备（第 1-4 周）

#### 3.1.1 海参肽 Cargo 确定

**文献来源**：
- PubMed/CNKI 搜索：`sea cucumber peptide Alzheimer's Aβ antioxidant`
- 已知活性海参肽特征：
  - 富含甘氨酸、脯氨酸、羟脯氨酸（胶原特征）
  - 分子量：500-2000 Da（约 5-18 个氨基酸）
  - 活性基序：Gly-Pro-X（GPX）三肽重复

**候选序列模板（代表性已报道活性肽）**：

| 肽编号 | 序列 | 长度 | 来源 | 报道活性 |
|--------|------|:----:|------|----------|
| SCP-01 | GPAGPIGPAG | 10 | 海参胶原 | Aβ 聚集抑制 |
| SCP-02 | GPPGPPGPP | 9 | 海参体壁 | 抗氧化 |
| SCP-03 | GPAGPAGPAGPAG | 13 | 仿生设计 | 双功能 |

> **注**：以上为代表性模板骨架。全部待定候选需根据你手上的体内实验数据，确认具体活性序列后确定。

**操作**：
1. 文献系统检索 + 你现有的动物实验数据交叉验证
2. 确定 2-3 条具有体内活性证据的肽序列
3. 多序列比对，提取活性保守基序

#### 3.1.2 TfR VHH 模板获取

**来源 A：公开专利 VHH 序列**

Vect-Horus 专利（US2022/0090050）公开了多个人/鼠交叉反应性抗 TfR VHH。其 CDR 支架可用于 BFN 模板设计。

WO2020132584 公开了 13E4 家族抗 TfR VH/VL 序列，可提取 CDR 移植到 VHH 框架。

**代表性 VHH 框架序列**（通用模板，需根据靶点替换 CDR 环区）：

```
VHH 通用框架:
FR1: EVQLVESGGGLVQPGGSLRLSCAAS
FR2: WVRQAPGKGLEWVS
FR3: RFTISRDNSKNTLYLQMNSLRAEDTAVYYC
FR4: WGQGTLVTVSS

典型骆驼 VHH 框架:
FR1: QVQLQESGGGLVQAGGSLRLSCAAS
FR2: WFRQAPGKEREFVA
FR3: RFTISRDNAKNTVYLQMNSLKPEDTAVYYC
FR4: WGQGTQVTVSS

人源化 VHH 框架（优选）:
FR1: EVQLVESGGGLVQPGGSLRLSCAAS
FR2: WVRQAPGKGLEWVS
FR3: RFTISRDNSKNTLYLQMNSLRAEDTAVYYC
FR4: WGQGTLVTVSS
```

**操作**：
1. 从专利/文献中提取 3-5 条结合 TfR 完整 VHH 序列（含 CDR）
2. 用 IMGT/Chothia 注释 CDR 边界
3. 评估序列多样性，选择代表人源化 VHH 框架

#### 3.1.3 TfR 靶点结构准备

**PDB 条目**（TfR1 胞外域）：
- `1SUV`：TfR1 胞外域二聚体（人）
- `3KAS`：TfR1-转铁蛋白复合物
- `6W5W`：TfR1 高分辨率结构

**操作**：
1. 下载高分辨率 PDB 结构
2. 提取 apical domain（VHH 最常结合区域）
3. 准备为 BFN 分子对接和设计约束的输入

### Phase 2: BFN 计算设计（第 5-10 周）

#### 3.2.1 VHH 骨架设计

**策略**：
```
步骤 1: 模板融合
  - 取专利 VHH 的 CDR 区域 + 人源化 FR 框架
  - 生成 3-5 条初始序列

步骤 2: BFN CDR 优化
  - 训练/微调 BFN 于纳米抗体序列数据集
  - 以 CDR3 为主要优化目标（贡献 >80% 结合能）
  - 约束：保持 CDR 长度分布合理（CDR3: 10-18 aa）
  - 生成 100-200 条候选 VHH 序列

步骤 3: 计算筛选
  a. AlphaFold2/AntiFold 结构预测
  b. Rosetta/HDOCK 对接 TfR1 apical domain
  c. 筛选标准:
     - 结合能 < -35 kcal/mol
     - pLDDT > 85
     - 预估 Tm > 55°C
     - 预估可溶表达概率 > 0.6
  d. 聚类选择 2-5 条代表性序列

步骤 4: CDR 多样性分析
  - 确保选中的 VHH 覆盖不同 CDR 结合模式
  - 避免单一 CDR3 聚类
```

#### 3.2.2 融合蛋白设计

**融合方案**：

```
方案 A（推荐）: VHH-linker-海参肽
  N - [VHH] - [GGGGS]ₙ - [SCP] - 6×His - C

方案 B: 海参肽-linker-VHH  
  N - [SCP] - [GGGGS]ₙ - [VHH] - 6×His - C

方案 C: bivalent
  N - [SCP] - [GGGGS]ₙ - [VHH] - [GGGGS]ₙ - [SCP] - 6×His - C
```

**设计参数**：
| 参数 | 探索范围 |
|------|----------|
| Linker 长度 | (GGGGS)₂, (GGGGS)₃, (GGGGS)₄ |
| 融合方向 | N-VHH-Cargo, N-Cargo-VHH |
| 6×His 位置 | C 端（方便纯化） |

**计算评估**：
1. BFN/AntiFold 预测融合蛋白结构
2. 检查 VHH 构象是否被 cargo 干扰
3. 检查 cargo 肽段是否暴露在溶剂中（保持活性）
4. 评估整体稳定性

#### 3.2.3 最终候选清单

| 候选 | VHH 骨架 | Cargo 肽 | Linker | 预期表达量 | 预期亲和力 |
|------|----------|----------|--------|:----------:|:----------:|
| VH-SCP-01 | VHH_A | SCP-01 | (GGGGS)₃ | >10 mg/L | <100 nM |
| VH-SCP-02 | VHH_A | SCP-02 | (GGGGS)₃ | >10 mg/L | <100 nM |
| VH-SCP-03 | VHH_B | SCP-01 | (GGGGS)₃ | >10 mg/L | <100 nM |
| VH-SCP-04 | VHH_B | SCP-03 | (GGGGS)₃ | >10 mg/L | <100 nM |
| ... | ... | ... | ... | ... | ... |

### Phase 3: 表达与纯化（第 11-18 周）

#### 3.3.1 基因合成与克隆

1. 序列经大肠杆菌密码子优化（GeneOptimizer / DNAWorks）
2. 全基因合成（金唯智/通用），克隆至 pET-22b(+) / pET-28a(+)
3. 测序验证

#### 3.3.2 小量表达筛选

| 条件 | 参数 |
|------|------|
| 宿主菌 | BL21(DE3), Rosetta(DE3), SHuffle T7 |
| 温度 | 37°C, 30°C, 25°C, 18°C |
| IPTG 浓度 | 0.1, 0.5, 1.0 mM |
| 诱导时间 | 4h, 8h, 16h |
| 检测 | SDS-PAGE + Western blot (anti-His) |

**筛选标准**：
- 可溶表达占比 >50%
- 表达量 >10 mg/L

#### 3.3.3 大量表达与纯化

1. **Ni-NTA 亲和层析**：50 mM Tris-HCl pH 8.0, 300 mM NaCl, 20-500 mM 咪唑梯度洗脱
2. **尺寸排阻色谱**（SEC）：Superdex 75 10/300 GL, PBS pH 7.4
3. **内毒素去除**（如用于细胞/动物实验）：Triton X-114 相分离或 Polymyxin B 柱

#### 3.3.4 质量鉴定

| 方法 | 检测内容 | 标准 |
|------|----------|------|
| SDS-PAGE | 纯度 | >95% |
| SEC-HPLC | 聚集状态 | 单体 >90% |
| CD 光谱 | 二级结构 | 典型 β-折叠 VHH 谱 |
| LC-MS | 精确分子量 | 实测误差 <1 Da |

### Phase 4: 体外功能验证（第 15-22 周）

#### 4.1 TfR 结合亲和力

**表面等离子体共振（SPR）**：
- 芯片：CM5，固定化 TfR1 胞外域（his-tagged，anti-His capture）
- 分析物：2 倍梯度稀释（0.78-100 nM），5 个浓度
- 拟合：1:1 Langmuir 模型
- 预期 Kd：<100 nM（理想 <10 nM）

**生物层干涉（BLI）备选方案**：
- Octet RED96，Ni-NTA 传感器固定 TfR
- 同上浓度梯度

#### 4.2 BBB 穿越效率

**体外 BBB 模型**：
- 细胞系：hCMEC/D3（人脑微血管内皮细胞）
- Transwell 系统：0.4 μm 孔径聚酯膜
- TEER 值监测确保屏障完整性：>150 Ω·cm²

**实验流程**：
1. 顶室加入融合蛋白（10 μg/mL）
2. 37°C 孵育 0、1、2、4、8 h
3. 底室取样，anti-His ELISA 定量
4. 计算表观渗透系数（Papp）

**对照组**：
- 游离海参肽（无 VHH 融合）
- 非靶向 VHH-海参肽融合蛋白（阴性对照）
- 已知 BBB 穿越阳性对照（如转铁蛋白）

#### 4.3 Aβ 聚集抑制活性

**硫黄素 T（ThT）荧光法**：
1. Aβ₁₋₄₂ 单体（25 μM，PBS pH 7.4，37°C）± 融合蛋白（0.1-100 μM）
2. ThT 10 μM，每 10 min 读取荧光（λex = 440 nm，λem = 485 nm）
3. 拟合 sigmoidal 曲线，计算 IC₅₀ 和 lag time

#### 4.4 抗氧化活性

| 方法 | 检测原理 | 报告指标 |
|------|----------|----------|
| DPPH 清除 | DPPH 自由基还原 | IC₅₀ |
| ABTS 清除 | ABTS⁺ 阳离子自由基 | IC₅₀ |
| FRAP | Fe³⁺ 还原能力 | TEAC 当量 |

#### 4.5 细胞毒性

- MTT 法，bEnd.3 / SH-SY5Y 细胞
- 浓度范围：1-200 μM
- 48h 暴露

### Phase 5: 体内验证（第 19-30 周，可选/后期）

#### 5.1 药代动力学与脑分布

1. 荧光标记（Cy5.5-NHS 或 FITC）
2. C57BL/6 小鼠尾静脉注射（5 mg/kg）
3. 时间点：0.5、1、2、4、8、24h
4. 心脏灌注后取脑、肝、脾、肾
5. IVIS 成像 + 组织匀浆荧光定量
6. 脑/血浆浓度比（Kp）

#### 5.2 AD 模型药效

1. APP/PS1 双转基因小鼠（6 月龄）
2. 给药方案：尾静脉 5 mg/kg，每周 2 次 × 4 周
3. 行为学：Morris 水迷宫（给药后）
4. 生物标志物：脑匀浆 Aβ₁₋₄₀、Aβ₁₋₄₂（ELISA）
5. 病理：脑切片硫黄素 S 染色（斑块负荷）
6. 生化：脑 SOD、MDA、GSH（氧化应激指标）
7. 安全性：血常规、ALT/AST、肌酐、脑 HE

### 3.6 研究时间表

| 季度 | Phase | 主要里程碑 |
|:----:|-------|------------|
| Q1 | Phase 1+2 | 序列确定 + BFN 设计完成 |
| Q2 | Phase 3+4 | 表达纯化完成 + 体外数据初报 |
| Q3 | Phase 4+5 | 体外全套数据 + 体内启动 |
| Q4 | Phase 5+汇总 | 体内数据完成 + 论文撰写 |

---

## 4. 中期答辩框架

### 预期进度（Q2 末）

**已完成**：
- [x] 海参活性肽序列确定（2-3 条）
- [x] TfR VHH 模板序列获取（3-5 条）
- [x] BFN VHH 设计完成，筛选出 2-5 个候选
- [x] 融合蛋白 BFN 建模示意
- [x] 基因合成完成
- [x] 小量表达筛选完成（____/10 可溶表达）
- [ ] SPR/BLI 亲和力数据（进行中/完成）
- [ ] BBB 穿越实验（进行中/完成）

**阶段性结果**：
- BFN 设计的 VHH 序列展示
- 表达纯化 SDS-PAGE 结果
- 初步 TfR 结合数据（如有）

**未完成**：
- [ ] 体内验证（Phase 5，计划 Q3-Q4）

**问题与调整**：
- 表达量偏低 → 更换宿主菌/优化密码子
- 亲和力不足 → BFN 第二轮 CDR 优化
- 稳定性问题 → linker 调整/表面残基优化

### 中期答辩 PPT 大纲

```
1. 课题背景（3 页）
   - AD 治疗困境 → 海参肽机会 → BBB 递送必要性
2. 技术路线（2 页）
   - BFN 设计平台 → 融合蛋白策略
3. 阶段性结果（10 页）
   - BFN 设计结果（序列比对 + 结构预测）
   - 表达纯化结果（SDS-PAGE + SEC + CD）
   - 体外活性初报（SPR/BBB/Aβ 抑制）
4. 存在问题（1 页）
   - 坦诚问题 + 解决方案
5. 后续计划（2 页）
   - Q3-Q4 时间线 + 体内验证方案
```

---

## 5. 预期成果与交付物

### 5.1 学术论文（目标期刊）

1. **方法学论文**："Computational Design of Anti-TfR Nanobodies Using Bayesian Flow Networks" — _Bioinformatics_ / _J. Chem. Inf. Model._
2. **核心论文**："A TfR-Targeting Nanobody-Sea Cucumber Peptide Fusion Protein for BBB-Penetrant Alzheimer's Therapy" — _J. Controlled Release_ / _ACS Chem. Neurosci._
3. **综述论文**："Natural Bioactive Peptides as Therapeutic Cargos for Brain Delivery: Opportunities and Challenges"

### 5.2 专利

- **发明专利**：一种靶向转铁蛋白受体的纳米抗体-海参活性肽融合蛋白及其应用
  - 保护范围：VHH 序列族 + 融合蛋白组合 + 治疗 AD 用途
  - 申请策略：PCT 国际申请（优先权日后进入中国、美国、欧洲）

### 5.3 技术交付物

- BFN 优化后的 VHH 序列库（≥50 条序列）
- 1-2 个 lead 融合蛋白表达菌株
- 纯化工艺 SOP
- 体外活性全套数据集

---

## 6. 风险与应对

| 风险 | 概率 | 影响 | 应对策略 |
|------|:----:|:----:|----------|
| BFN 设计 VHH 亲和力不足 | 中 | 高 | 1) 第二轮优化 2) 加入已知高亲和力 CDR 模板约束 |
| 融合蛋白表达量低/包涵体 | 中 | 中 | 1) 更换宿主菌（SHuffle、Origami）2) 低温诱导 3) 密码子再优化 |
| VHH-肽融合后活性互扰 | 低 | 中 | 1) 调整 linker 长度 2) 尝试 bivalent 方案 |
| 体外 BBB 模型与体内相关性差 | 中 | 中 | 直接推进小鼠体内分布实验 |
| 海参肽体内活性被融合蛋白屏蔽 | 低 | 中 | 1) 蛋白酶可切 linker 2) N 端融合测试 |
| 体内免疫原性 | 低-中 | 高 | 选用人源化 VHH 框架；如出现 ADA，构建去免疫原变体 |

---

## 7. 参考文献

1. Yin T, Yesiltepe M, D'Adamio L. NewroBus for the brain: humanized TfR1-targeting nanobodies with high BBB permeability and cargo transport capacity. _bioRxiv_, 2025. DOI: 10.1101/2025.04.20.649139
2. Rué L, Jaspers T, De Strooper B, et al. Novel human/non-human primate cross-reactive anti-transferrin receptor nanobodies for brain delivery of biologics. _Pharmaceutics_, 2023, 15(6): 1748. DOI: 10.3390/pharmaceutics15061748
3. David M, Cohen R, Khrestchatisky M, et al. Novel single-domain antibodies targeting a unique transferrin receptor 1 epitope for cross-species delivery of drugs in the CNS. _J. Control. Release_, 2025. DOI: 10.1016/j.jconrel.2025.114344
4. Vect-Horus SAS. Transferrin receptor-binding molecules, conjugates thereof and their uses. US Patent US20220090050A1, 2022.
5. JCR Pharmaceuticals. Izcargo (pabinafusp alfa) prescribing information, Japan, 2021.
6. Kariolis MS, Wells RC, Getz JA, et al. Brain delivery of therapeutic proteins using an Fc fragment blood-brain barrier transport vehicle in mice and monkeys. _Sci. Transl. Med._, 2020, 12(545): eaay1359.
7. Sevigny J, Chiao P, Bussière T, et al. The antibody aducanumab reduces Aβ plaques in Alzheimer's disease. _Nature_, 2016, 537: 50-56.
8. van Dyck CH, Swanson CJ, Aisen P, et al. Lecanemab in early Alzheimer's disease. _N. Engl. J. Med._, 2023, 388: 9-21.
9. 海参胶原肽的制备及其神经保护活性研究. 国内相关文献（根据你的实验背景补充）
10. Xu J, et al. Sea cucumber peptides: preparation, bioactivities, and structure-activity relationship. _Mar. Drugs_, 2022, 20(8): 512.
11. Pardridge WM. Blood-brain barrier delivery. _Drug Discov. Today_, 2007, 12(1-2): 54-61.
12. Muyldermans S. Nanobodies: natural single-domain antibodies. _Annu. Rev. Biochem._, 2013, 82: 775-797.

---

## 附录

### A. TfR VHH 模板序列候选（专利 WO2020132584 家族 13E4）

```text
>13E4_VH2 (SEQ ID NO:14)
QVQLVQSGAEVKKPGASVKVSKASGYTFTNYWMHWVRQAPGQGLEW
IGENPINGRSNYAEKFQGRVTLTVDTSISTAYMELSRLRSDDTAVYYCAR
GTRAMHYWGQGTLVTVSS
```

### B. VHH 人源化通用框架

```text
>humanized VHH framework
EVQLVESGGGLVQPGGSLRLSCAAS[---CDR1---]WVRQAPGKGLEWVS
[---CDR2---]RFTISRDNSKNTLYLQMNSLRAEDTAVYYC[---CDR3---]
WGQGTLVTVSS
```

### C. Linker 库

```text
(GGGGS)₂: GGGGSGGGGS
(GGGGS)₃: GGGGSGGGGSGGGGS
(GGGGS)₄: GGGGSGGGGSGGGGSGGGGS
刚性: EAAAKAAAKEAAAKAAAKEAAAK
```

### D. 海参肽候选序列模板（代表性骨架）

```text
>SCP-01 (collagen-type, Aβ inhibition)
G P A G P I G P A G

>SCP-02 (collagen-type, antioxidant)  
G P P G P P G P P

>SCP-03 (bifunctional design)
G P A G P A G P A G P A G
```

> **重要提示**：以上为代表性骨架模板。最终实验序列需根据你的体内动物实验数据中确认的活性肽段确定。

---

*文档版本：v1.0 | 生成日期：2026-05-27 | 基于 BFN V10 计算平台*
