# IDP 抗体湿实验验证完整方案 V15 — 25 计算候选 → SPR/BLI 结合验证（执行级细节）

**交付**: 用户/实验团队 + DeepSeek V4 | **日期**: 2026-07-05 | **版本**: V15 细节化
**唯一目标**: 对计算产出的 25 候选（Aβ42 靶点）做湿实验结合验证，确定哪些真结合、亲和力多高。
**计算端状态**: 完成。S0 [FALSIFIED] → Way4 降为 selection filter；§5 fallback 产出 25 候选（top25 排序，`idp_design_results/s5_wetlab_candidates.json`），[PENDING]。
**资产三件套**: 候选JSON + `scripts/v12_wetlab_candidates.py` + [PENDING→待湿实验定 VERIFIED/FALSIFIED]。

---

## 0. 计算候选基线（实验团队必读——了解这些候选从哪来、代表什么）

### 0.1 候选来源与分布（25 条 top25）

| 来源类型 | 数量 | 说明 |
|----------|------|------|
| **文献真实抗 Aβ CDR** | 12 | Solanezumab / Donanemab / Crenezumab / Lecanemab / Gantenerumab / Bapineuzumab / Ponezumab 等——天然免疫系统产生的抗 Aβ 抗体 CDR-H3 序列 |
| **PDB 结构抗 Aβ CDR** | 4 | 4HIX / 5CSZ / 3UOT / 6CGZ——晶体/电镜结构中的抗 Aβ 抗体 CDR-H3 |
| **Template-seeded MPNN redesign** | 7 | Pillar A：用 4HIX/5CSZ 框架 + 热点残基冻结 + ProteinMPNN 重设计 CDR |
| **跨 IDP 抗体（交叉反应探针）** | 2 | Anti-tau Fab / Anti-αSyn scFv——测对 Aβ42 是否有交叉反应 |

### 0.2 每条候选字段说明

```
cdr_sequence       — CDR-H3 氨基酸序列（仅 CDR-H3，不含框架）
cdr_length         — CDR-H3 长度（7-80 aa，中位 ~14）
epitope            — 靶向的 Aβ42 区段：N-term(1-11) / mid(12-24) / C-term(30-42)
epitope_sequence   — Aβ42 对应区段序列
epitope_disorder   — 该表位预测 disorder 值（DisProt 训练的 disorder head, 0-1, 越高越无序）
source_antibody    — 来源抗体名称
source_type        — "literature"（文献）/ "PDB XXXX"（结构）/ "template_seeded"（计算设计）
pliability_score   — 方式4 排序分数（CDR 柔性与表位 disorder 匹配度，0-1, 越高越"匹配"）
entropy            — CDR 序列熵（bits, 高=多样）
complexity         — 氨基酸种类丰富度（unique_aa / 20）
solubility         — 预测溶解度（基于 Kyle-Doolittle 疏水性 + 净电荷，0-1, 越高越可溶）
immunogenicity_risk — 预测免疫原性红旗（MHC-I 结合基序密度，0-1, 越高风险越大）
anti_degen         — 无 6-mer 重复退化（全部 true = 已过滤）
composite_score    — 综合评分（pliability + entropy + solubility 加权，0-1）
```

### 0.3 关键缺口（S0 必须补全，否则无法合成）

**当前每条候选只有 CDR-H3 序列，无完整 VH/VL 序列，无框架信息。** 实验合成抗体需要完整的可变区序列（VH + VL，至少 220+ aa）。S0 的任务就是对每条候选补全完整可合成序列。

---

## S0. 序列补全（计算端最后一步，1-2 天）

### S0.1 目标
对 25 候选每条产出：
- **scFv 完整氨基酸序列**：VH-(G₄S)₃-VL-His₆，可直接送基因合成
- **Fab 备选**（若实验团队偏好）：VH + VL 分别列出
- **注释表**：每条标注框架来源 PDB、CDR 位置、预期分子量、消光系数、pI

### S0.2 每条候选的框架来源分配

#### 组 1：PDB 结构来源（4 条）—— 直接取 PDB 的 VH/VL

| 候选 # | source_antibody | PDB | VH 链 | VL 链 | 备注 |
|--------|----------------|-----|-------|-------|------|
| A1 | Anti-Aβ Fab (6CGZ) | 6CGZ | H | L | 全长 Fab 结构 |
| A2 | Crenezumab-like (5CSZ) | 5CSZ | H | L | 全长 Fab 结构 |
| A3 | Aducanumab-like (4HIX) | 4HIX | H | L | 全长 Fab 结构 |
| A4 | Anti-Aβ scFv (3UOT) | 3UOT | VH* | VL* | scFv 已有完整序列 |

\* 3UOT 如已是 scFv 格式则直接使用，否则按 PDB 链标注。

**操作方法**：
1. 用 BioPython `PDB.PDBParser` 解析 PDB，按 chain ID 提取 VH/VL 序列（残基编号按 Chothia/Kabat，取 FR1-CDR3-FR4 完整区段）
2. 用 ANARCI（`pip install anarci`）验证 VH/VL 分配正确
3. 将候选 CDR-H3 序列 **替换** 原生 CDR-H3（保留所有框架残基不变）
4. 若候选 CDR 长度与原生不同 → 保留候选全长，框架残基不变，仅替换 CDR-H3 区段

#### 组 2：文献真实抗体（12 条）—— 查 PDB/文献取 VH/VL

| 候选 # | source_antibody | 框架来源 | 获取方式 |
|--------|----------------|---------|---------|
| B1 | Solanezumab | PDB 4HIX（同源）或专利 US 7,887,802 | PDB/专利序列 |
| B2 | Donanemab | PDB 6D9B 或专利序列 | PDB/专利 |
| B3 | Crenezumab | PDB 5CSZ（同源）或专利 US 8,105,594 | PDB/专利 |
| B4 | Lecanemab (BAN2401) | PDB 7K4V 或专利 WO 2018/007526 | PDB/专利 |
| B5 | Gantenerumab | PDB 6H3R 或专利序列 | PDB/专利 |
| B6 | Bapineuzumab | PDB 4HIX（同源）或专利序列 | PDB/专利 |
| B7 | Ponezumab | 文献序列（US 8,512,697 或类似） | 专利搜索 |
| B8-B12 | 其余文献抗体 | 逐一按名称查 PDB / IMGT / Thera-SAbDab | 数据库检索 |

**操作步骤**：
1. 对每条，先查 Thera-SAbDab（http://opig.stats.ox.ac.uk/webapps/therasabdab/）获取 VH/VL 序列
2. 若 Thera-SAbDab 无收录 → NCBI/专利检索（Google Patents / WIPO PATENTSCOPE）
3. 序列到后，用 ANARCI 编号（Chothia），确定 CDR-H3 起止位置
4. 将候选 CDR-H3 序列替换原生 CDR-H3
5. 若某抗体找不到 VH/VL 序列 → 标注为 UNAVAILABLE，不阻塞整体

#### 组 3：Template-seeded MPNN redesign（7 条）—— Graft 到源 scaffold

| 候选 # | source_antibody | 框架来源 | 备注 |
|--------|----------------|---------|------|
| C1-C4 | PillarA_4HIX | 4HIX VH/VL 框架 | 保留 4HIX 框架 + 替换 CDR-H3 |
| C5-C7 | PillarA_5CSZ | 5CSZ VH/VL 框架 | 保留 5CSZ 框架 + 替换 CDR-H3 |

**操作步骤**：
1. 提取 4HIX / 5CSZ 的 VH/VL 框架序列（保留 FR1-CDR3-FR4 中除 CDR-H3 外的所有残基）
2. 将 template-seeded 的 CDR-H3 序列替换原生 CDR-H3
3. PillarA 设计时已冻结热点残基，不需要额外的结构验证

#### 组 4：跨 IDP 抗体（2 条）—— 取 native 框架 + 标注交叉靶点

| 候选 # | source_antibody | PDB | 靶点 | 备注 |
|--------|----------------|-----|------|------|
| D1 | Anti-tau Fab | 6CBV 或 6H3R | Tau | 测 Aβ42 交叉反应 |
| D2 | Anti-αSyn scFv | 6H1F | α-Synuclein | 测 Aβ42 交叉反应 |

**操作步骤**：
1. 取 native PDB VH/VL 框架（不做 CDR 替换——这些就是原生的跨 IDP 抗体）
2. 标注交叉靶点，S4 时作为交叉反应探针

### S0.3 scFv 构建格式（标准模板）

每条候选产出一个标准 scFv 构建：

```
[VH 序列] - [GGGGSGGGGSGGGGS] - [VL 序列] - [HHHHHH]
```

**详细规范**：
- **VH→VL 顺序**：VH 在 N 端，VL 在 C 端（标准 scFv 格式，不影响大多数抗体的抗原结合）
- **Linker**: (G₄S)₃ = GGGGSGGGGSGGGGS（15 aa，~45 Å，允许链内 VH-VL 配对）
  - 如果实验团队偏好更长的 linker 以降低链间配对 → 可用 (G₄S)₄ = GGGGSGGGGSGGGGSGGGGS（20 aa）
- **His-tag**: C 端 6×His（HHHHHH），用于 Ni-NTA 纯化
- **可选 AviTag**：若需要位点特异性生物素化（BLI SA 传感器固定），在 His-tag 前加 GLNDIFEAQKIEWHE：
  ```
  [VH] - [(G₄S)₃] - [VL] - [GLNDIFEAQKIEWHE] - [HHHHHH]
  ```
- **信号肽**（分泌表达）：N 端加 MGWSCIILFLVATATGVHS（murine Ig κ 信号肽），哺乳细胞表达时自动切除
  - 大肠杆菌表达不加信号肽，用 T7 启动子 + ATG 起始
- **预期分子量**：scFv ~27-30 kDa（~250 aa），加 His-tag 后 ~28-31 kDa
- **消光系数**：按 ExPASy ProtParam 计算（基于 Trp/Tyr/Cys 数目）
- **pI**：按 ProtParam 计算

### S0.4 密码子优化

- **哺乳表达（HEK293）**：用 Homo sapiens codon usage（Kazusa codon table），避免 CpG 岛、隐蔽剪接位点、RNA 不稳定基序（ATTTA）
- **大肠杆菌表达**：用 E. coli K12 codon usage，避免稀有密码子（AGG/AGA/ATA/CTA）连续出现
- **工具**：IDT Codon Optimization Tool 或 Twist Bioscience 在线工具
- **输出**：对每条候选产两份——氨基酸序列（FASTA）+ 密码子优化的 DNA 序列（供基因合成公司）

### S0.5 输出物

```
idp_design_results/s5_wetlab_synthesis_20260705/
├── candidates_aa.fa          # 25 条 scFv 氨基酸序列（FASTA）
├── candidates_dna.fa         # 25 条密码子优化的 DNA 序列
├── candidates_annotation.tsv # 每条：框架来源PDB/VH链/VL链/CDR位置/MW/pI/消光系数
├── candidates_fab.fa         # Fab 备选（VH.fa + VL.fa 分开）
├── unpaired_candidates.tsv   # 无法找到 VH/VL 的候选（如有），标注原因
└── prepare_synthesis_seqs.py # 生成脚本（可复现）
```

### S0.6 Go/NoGo

| 阈值 | 判定 | 动作 |
|------|------|------|
| ≥23/25 获完整 VH/VL | **GO** | → S1 |
| 15-22/25 | **WARN** | → S1，但标注缺框架候选，不阻塞 |
| <15/25 | **NoGo** | 停止。重审框架检索策略，扩大 PDB/专利搜索后再试 |

**三件套**: synthesis FASTA + `prepare_synthesis_seqs.py` + [PENDING→S1 后变 VERIFIED 可合成]。

---

## S1. 抗体表达与纯化（2-4 周）

### S1.1 表达系统选择

| 候选数量 | 推荐系统 | 原因 |
|----------|---------|------|
| **≤25（当前）** | **HEK293F 悬浮 + 瞬时转染** | 哺乳折叠正确、糖基化、分泌表达、纯化简单 |
| 备选 | 大肠杆菌 BL21(DE3) + 包涵体复性 | 更快（1-2 周）但需复性、可能错配二硫键——仅 HEK 失败后启用 |

**推荐方案**：全部 25 走 HEK293F，统一流程，减少变量。

### S1.2 具体操作流程

#### 步骤 1: 基因合成与克隆（外包，1 周）
- **供应商**: Twist Bioscience / GenScript / IDT（选一家）
- **交付**: S0 产出的 `candidates_dna.fa`（25 条 DNA 序列）
- **载体**: pcDNA3.4-TOPO（ThermoFisher, Cat# A14697）或 pTT5（NRC 许可）
  - 酶切位点: HindIII（5'）+ NotI（3'），含 Kozak 序列（GCCACC）在 ATG 前
  - 载体自带 CMV 启动子 + WPRE 增强子 + polyA
- **验证**: 每克隆 Sanger 测序确认 CDR-H3 区段正确

#### 步骤 2: 转染（3-4 天）
- **细胞**: Expi293F（ThermoFisher, Cat# A14527）或 FreeStyle 293F（ThermoFisher, Cat# R79007）
- **培养基**: Expi293 Expression Medium（ThermoFisher, Cat# A1435101）
- **规模**: 每候选 30 mL（小规模，够 SPR 初筛。若需要更多 → 扩至 100 mL）
- **转染试剂**: PEIpro（Polyplus, Cat# 115-100）或 ExpiFectamine 293（ThermoFisher, Cat# A14525）
- **过程**:
  1. 转染前 1 天：传细胞至密度 2×10⁶ cells/mL，活性 >95%
  2. 转染当天：细胞密度 3×10⁶ cells/mL，换新鲜培养基
  3. DNA:PEI = 1:3（w/w），DNA 终浓度 1 μg/mL 培养基
  4. 复合物室温孵育 15 min，缓慢加入细胞
  5. 37°C, 8% CO₂, 125 rpm 悬浮培养
  6. 转染后 20-24h 加 Enhancer（ExpiFectamine 体系）
  7. 96h 收上清

#### 步骤 3: 纯化（2-3 天）
- **方法**: 固定化金属离子亲和层析（IMAC），利用 C 端 His₆
- **树脂**: Ni-NTA Agarose（Qiagen, Cat# 30210）或 HisTrap HP 预装柱（Cytiva, Cat# 17524701）
- **缓冲液**:
  - 结合缓冲液: 50 mM Tris-HCl pH 8.0, 300 mM NaCl, 20 mM 咪唑
  - 洗脱缓冲液: 50 mM Tris-HCl pH 8.0, 300 mM NaCl, 250 mM 咪唑
- **流程**:
  1. 上清 4000g 离心 10 min，0.45 μm 过滤
  2. 上样到预平衡的 Ni-NTA 柱（重力柱或 ÄKTA）
  3. 10 CV 结合缓冲液洗涤
  4. 5 CV 洗脱缓冲液洗脱
  5. 洗脱液用 PD-10 脱盐柱（Cytiva, Cat# 17085101）换到 PBS pH 7.4
- **浓缩**: Amicon Ultra-15 10K MWCO（Millipore, Cat# UFC9010），4000g 至 ~1 mL

#### 步骤 4: QC（每候选）
- **SDS-PAGE**: 4-12% Bis-Tris 胶，还原（+DTT）/非还原各一孔，考马斯亮蓝染色
  - 预期条带: scFv ~28-31 kDa（单体），可能见 ~60 kDa 少量二聚体
- **SEC-HPLC**: Superdex 75 Increase 10/300 GL（Cytiva, Cat# 29148721），PBS pH 7.4, 0.5 mL/min
  - GO: 单体峰占比 > 90%
  - WARN: 70-90%，含聚集但可切单体峰
  - FAIL: < 70% 单体 → 该候选标为 AGGREGATED，不进 S2，重试优化（降温 30°C 表达 or 加 0.5 M 精氨酸）
- **浓度**: Nanodrop A280，用 S0 计算的消光系数
  - GO: ≥ 0.1 mg/mL（30 mL 培养液 → ~0.3-1.5 mg）
  - WARN: 0.05-0.1 mg/mL → 浓缩
  - FAIL: < 0.05 mg/mL → 该候选标为 LOW YIELD，不阻塞但降优先级

### S1.3 Go/NoGo + 意外全预案

| 场景 | 判定 | 响应 |
|------|------|------|
| ≥18/25 可溶单体 | **GO** | → S2，1.5 μg × 25 = ~40 μg 够 BLI |
| 10-17/25 | **WARN** | → S2 仅测通过 QC 的候选 |
| <10/25 | **NoGo** | 停止。批量问题：换表达系统（ExpiCHO 或大肠杆菌）|
| 某候选聚集 | per-candidate | 降温 30°C 表达、加 0.5M 精氨酸、或切 SEC 单体峰 |
| 某候选无表达 | per-candidate | 换密码子优化（大肠杆菌 codon）→ 大肠杆菌 BL21 |
| 全部聚集 | **NoGo** | scFv→Fab（VH+VL 分开表达+体外复性），或加分子伴侣 |
| 产率不足 | per-candidate | 扩培养体量（30→100 mL），或分批浓缩合并 |

**三件套**: 表达纯化记录 + SDS/SEC 图（每候选 pdf/png）+ 每候选标 [VERIFIED 可表达]/[FALSIFIED]。

---

## S2. BLI 初筛（25 → hit 短名单，1-2 周）

### S2.1 仪器与耗材

- **仪器**: Octet RED96e 或 Octet R8（Sartorius），8 通道并行
- **传感器**: Streptavidin（SA）生物传感器（Sartorius, Cat# 18-5019）
  - 备选: Anti-His（HIS2）传感器（Sartorius, Cat# 18-5120）——如果 AviTag 生物素化不可用
- **板**: 黑色 96 孔板（Greiner, Cat# 655209）
- **运行缓冲液**: PBS pH 7.4 + 0.02% Tween-20 + 0.1% BSA（PBST-BSA）
  - 关键: BSA 浓度不要太高（表面活性剂干扰传感）——0.1% 是 Octet 标准浓度

### S2.2 Aβ42 制备（关键步骤，最多踩坑）

> ⚠️ Aβ42 自聚集是头号假阳性来源。每次实验前必须新鲜制备单体。

**HFIP 单体化——标准协议**：
1. **HFIP 处理**（在通风橱操作）：
   - Aβ42 冻干粉（Bachem, Cat# H-1368 或 rPeptide, Cat# A-1002-1）→ 加 100% HFIP（六氟异丙醇，Sigma, Cat# 105228）至 1 mM
   - 涡旋振荡至完全溶解，室温孵育 30 min（解聚所有预存聚集）
   - 分装到低吸附管（Eppendorf LoBind, Cat# 0030108116），每管 50-100 μL
2. **HFIP 去除**：
   - 在 SpeedVac 中蒸发 HFIP（~30 min，不加热或 ≤30°C）
   - 管底见透明薄膜 = 单体化 Aβ42
   - 冻干管存于 -80°C（可存数月）
3. **使用前复溶**：
   - 加 DMSO（无水，Sigma, Cat# D8418）至 5 mM（DMSO 中 Aβ42），超声 10 min
   - 用 PBS pH 7.4 稀释至 100 μM（终 DMSO < 2%）
   - **立即使用**（室温 > 1h 开始聚集）
4. **QC（每次必做）**：
   - ThT 荧光：加 20 μM ThT（Sigma, Cat# T3516），Ex 440 nm / Em 482 nm
     - 单体 Aβ42: ThT 荧光 ≈ 缓冲液背景
     - 寡聚/纤维: ThT 荧光 >> 背景
   - 备选 TEM: 负染（2% 乙酸铀酰），单体应无可见结构

**Aβ42 寡聚体制备**（如果初筛对象是寡聚）：
- 单体 Aβ42 100 μM in PBS，4°C 静置 24h → 毒性寡聚体（~12-50 聚体）
- ThT 荧光轻微上升（vs 单体），TEM 见 ~5-10 nm 球状颗粒

**Aβ42 纤维制备**（S4 特异性时用）：
- 单体 Aβ42 100 μM in PBS + 10 mM HCl（降 pH 促聚集），37°C 搅拌 48h
- ThT 荧光饱和（10-50× 背景），TEM 见 10 nm 直径长纤维

### S2.3 对照设计（每板必含）

| 对照 | 描述 | 名称/来源 | 目的 |
|------|------|----------|------|
| **阳性对照** | 已知抗 Aβ 抗体（取 native CDR-H3） | 4HIX native scFv / 5CSZ native scFv | 确认实验体系有效 |
| **阴性对照 1** | 缓冲液 | PBST-BSA 基线 | 非特异传感器漂移 |
| **阴性对照 2** | 无关 scFv | Anti-HEL scFv（溶菌酶抗体, PDB 1BQL VH/VL 做 scFv） | 排除 scFv 骨架非特异 |
| **Aβ-only** | 无 Aβ42 加载的传感器 + 候选抗体 | 各候选 | 排除抗体-传感器直接结合 |
| **参考通道** | SA 传感器 + 生物素无关 + 候选抗体 | 生物素-BSA（Sigma, Cat# A8549）| 扣减传感器非特异 |

### S2.4 实验设计（96 孔板布局）

**每轮测 8 候选**（Octet 8 通道并行），共 ~4 轮（32 候选，含对照）。

**典型单板布局**：

```
Row A: 8 传感器 → 候选抗体（100 nM，200 μL/孔）
Row B: 8 传感器 → 缓冲液基线再生
Row C: Aβ42 加载（生物素-Aβ42 20 μg/mL in PBST）
Row D: 洗涤（PBST）
Row E: 猝灭（生物素 10 μg/mL，封闭未占 SA 位点）
Row F: 结合相（候选抗体 100 nM，60s）
Row G: 解离相（缓冲液，60s）
Row H: 再生（10 mM 甘氨酸 pH 2.0, 3×5s 脉冲）
```

**BLI 运行参数**：
- 温度: 25°C
- 振荡: 1000 rpm
- 采集率: 5 Hz（标准动力学）
- 传感器预湿: PBST 中浸泡 10 min
- Aβ42 加载: 20 μg/mL 生物素-Aβ42，加载至 ~1 nm 响应（~5-10 min）
- 结合相: 60s
- 解离相: 60s（初筛）→ S3 时延长至 300s（定量）
- 3 复孔 per 候选（同板 3 列重复）

### S2.5 数据分析

- **软件**: Octet Data Analysis HT 12.0（Sartorius）
- **处理**:
  1. 参考通道扣减
  2. 基线对齐（结合相前 5s 平均）
  3. Savitzky-Golay 滤波（窗口 5 点）
- **阈值**:
  - **hit**: 结合相末响应 > 3× 阴性对照标准差的信号（通常 > 0.05 nm shift）
  - 解离相 > 20% 信号下降（有可测解离）
  - 3 复孔 CV < 30%
- **弱 hit**: 0.03-0.05 nm → 提浓度至 500 nM 重测
- **非特异**（flat 不解离）: 解离相信号降 < 10% → 标记可疑，S4 重点筛

### S2.6 Go/NoGo + 意外

| 结果 | 判定 | 动作 |
|------|------|------|
| ≥1 hit | **GO** | 所有 hit → S3 SPR 定量 |
| 0 hit | **NoGo** | 见 M2——可能 thesis 级失败 |
| 阳性对照无信号 | **FAIL** | Aβ42 制备问题，重做 HFIP 单体化 |
| 全板高信号（含阴性） | **FAIL** | 非特异结合（BSA 浓度不够？Aβ42 聚集？），排查后重做 |
| 某轮全无信号 | **WARN** | 该轮候选可能不表达或浓度低，回查 S1 QC |

**三件套**: BLI 原始曲线（.frd 文件）+ hit 列表 CSV + [VERIFIED hit]/[FALSIFIED no-hit]。

---

## S3. 亲和力定量（SPR，仅 S2 hit，1-2 周）

### S3.1 仪器与耗材

- **仪器**: Biacore T200 或 Biacore 8K（Cytiva）
- **芯片**: Series S Sensor Chip SA（Cytiva, Cat# BR100531）——与 BLI 同用生物素-Aβ42 固定
  - 或 CM5 芯片（Cat# BR100530）+ amine coupling 固定 anti-His 抗体→捕获 scFv——倒置格式（Aβ42 做分析物），避免 Aβ42 固定破坏表位
- **运行缓冲液**: HBS-EP+（10 mM HEPES pH 7.4, 150 mM NaCl, 3 mM EDTA, 0.05% P20）

### S3.2 固定策略（二选一，按实际情况）

**策略 1: SA 芯片 + 生物素-Aβ42 固定**（Aβ42 被固定，候选做分析物）
- 优势: 与 BLI 一致，实验结果可比
- 缺点: 生物素化可能破坏 Aβ42 N 端表位（如果候选靶向 N-term 1-11）

**策略 2: CM5 + anti-His 抗体固定 → 候选 scFv 捕获**（倒置格式，推荐）
- 优势: Aβ42 在溶液中（天然状态），不被固定破坏
- 缺点: 需要 anti-His 抗体（Cytiva His Capture Kit, Cat# 28995056）
- 流程:
  1. Amine coupling 固定 anti-His 抗体至 CM5（EDC/NHS 活化 → anti-His → 乙醇胺封闭）
     → 目标固定水平: 8000-10000 RU
  2. 候选 scFv 捕获到 anti-His 表面（5-10 μL/min, ~100-300 RU 捕获水平）
  3. Aβ42 做分析物，0.1-100 nM 6 浓度梯度 + 空白（0 nM）
  4. 每轮再生: 10 mM 甘氨酸 pH 2.0, 30s → 去 scFv 但不伤 anti-His

### S3.3 运行参数

- 温度: 25°C
- 流速: 30 μL/min
- 结合相: 120s
- 解离相: 300s（足够测 KD 到 10 μM 级别）
- Aβ42 浓度梯度: 0.1, 0.3, 1, 3, 10, 30, 100 nM（或按结合强度调整）
- 每浓度 2 复孔
- 每 10 循环跑一次 0 nM（缓冲液空白）+ 中浓度重复（验证芯片活性不降）
- 溶剂校正: DMSO 校正曲线（0.5%-2.5% DMSO in HBS-EP+）

### S3.4 数据分析

- **软件**: Biacore Insight Evaluation 3.0（Cytiva）
- **处理**:
  1. 参考通道扣减（空白 anti-His 表面 + 候选捕获 + 缓冲液注射）
  2. 双参考（0 nM 浓度扣减）
  3. 溶剂校正（若 Aβ42 含 DMSO）
- **拟合模型**:
  1. 先试 **1:1 Langmuir**（最简单的可逆结合）:
     ```
     d[AB]/dt = k_on × [A] × [B] - k_off × [AB]
     ```
  2. 若拟合差（χ² > 10% Rmax, U-value > 15）→ 试 **heterogeneous ligand**（两独立位点）或 **bivalent analyte**（scFv 可能有少量二聚体）
  3. 若结合/解离极快（快于仪器的 1s 分辨率）→ 报告稳态亲和力（steady-state fitting），不报动力学 KD
- **QC 接受标准**:
  - χ² < 10% Rmax
  - 残差无系统偏差（随机分布在 0 附近）
  - 重复浓度一致性（两复孔 Rmax 偏差 < 15%）
  - 芯片活性衰减 < 20%（中浓度重复 vs 首次）

### S3.5 亲和力分级

| KD 范围 | 级别 | 含义 |
|---------|------|------|
| < 1 nM | 强结合 | 成药级别，可做先导物 |
| 1-10 nM | 中强 | 先导物级，可优化 |
| 10-100 nM | 中等 | 常用抗体范围，可亲和力成熟 |
| 100 nM-1 μM | 弱 | 初始 hit，需亲和力成熟 |
| > 1 μM | 边缘 | 可测但弱——诚实标注 |
| 无拟合 | 不可测 | S2 可能假 hit |

### S3.6 Go/NoGo

| 结果 | 判定 | 动作 |
|------|------|------|
| ≥1 候选 KD < 1 μM + 拟合合格 | **GO** | → S4 特异性 |
| ≥1 候选 KD < 100 nM | **强 GO** | 优先标为 lead 候选 |
| 全部 > 1 μM | **NoGo** | M2——弱结合或无 |
| 全无可靠拟合 | **NoGo** | S2 hit 可能假阳性 → 回查 BLI 数据 |

**三件套**: SPR 原始感应曲线 + KD 表 + 拟合参数 + [VERIFIED]。

---

## S4. 特异性与反向筛选（1-2 周）

### S4.1 反向筛选矩阵（所有 S3 GO 候选必做）

**蛋白阵列（SPR 格式，Aβ42 100 nM 分析物）**：

| 靶标蛋白 | 浓度 | 固定水平 | 目的 |
|----------|------|---------|------|
| Aβ42 寡聚体 | 100 nM | —（分析物） | **靶标特异性** |
| Aβ42 单体 | 100 nM | — | 形式对照 |
| BSA | 100 nM | 2000 RU | 排除非特异（blocker 结合？） |
| 人转铁蛋白 | 100 nM | 2000 RU | 阴性对照（无淀粉样） |
| α-Synuclein | 100 nM | 2000 RU | 排除跨淀粉样泛结合 |
| Tau (2N4R) | 100 nM | 2000 RU | 排除交叉（尤其 D1/D2） |
| 人 IgG Fc | 100 nM | 2000 RU | 排除 Fc 结合（scFv 无 Fc，此条低优） |

**判定**:
- **特异**: Aβ42 寡聚体响应 > 10× 所有阴性蛋白响应
- **形式特异**: 寡聚体 vs 单体响应比 > 3:1 → 优势，论文强调
- **交叉**: 对 αSyn 或 Tau 有显著响应 → 非 Aβ 特异，降为 pan-amyloid 工具

### S4.2 表位竞争实验（SPR）

- 对同一表位的多个候选做竞争：候选 A 先结合至饱和 → 再注射候选 B
  - A+B 同表位 → B 响应降低 > 70%
  - A+B 不同表位 → B 响应不变
- 用已知表位抗体做对照：
  - Anti-N-term: 6E10（BioLegend, Cat# 803001, 表位 3-8）
  - Anti-mid: 4G8（BioLegend, Cat# 800701, 表位 17-24）
  - Anti-C-term: 可自制 C-term 肽（30-42）抗体

### S4.3 Aβ42 形式结合偏好

对每个 hit，用 BLI 并行测三种形式（同板，同浓度 100 nM，Aβ42 各形式固定在不同传感器）:

| Aβ42 形式 | 制备 | 固定方式 |
|-----------|------|---------|
| 单体 | 新鲜 HFIP 单体化 → 立即用 | 生物素-Aβ42 单体 |
| 寡聚体 | 单体 4°C 24h | 生物素-Aβ42 寡聚体 |
| 纤维 | 单体 37°C 搅拌 48h | 生物素-Aβ42 纤维（超声后） |

**判定**: 最高响应形式 = 候选偏好形式。寡聚体偏好 → 治疗相关（AD 病理形式），论文核心卖点。

### S4.4 Go/NoGo

| 结果 | 判定 | 动作 |
|------|------|------|
| ≥1 候选 Aβ42 特异 | **GO → 真阳性** | S5 报告 |
| 全非特异（跟 BSA/Tf 也结合） | **NoGo** | 回查 scFv 序列——是否有疏水补丁或聚体？SDS/SEC 回查 |
| 全交叉（跟 αSyn/Tau 结合） | **WARN** | 标为 pan-amyloid 工具抗体。非原目标但不失为产出 |
| 仅单体结合不结合寡聚体 | **WARN** | Aβ42 单体瞬态，生物学相关性有限 → 诚实标注 |

**三件套**: 反向筛选矩阵 + 形式结合偏好表 + [VERIFIED 特异]/[FALSIFIED 非特异]。

---

## S5. 报告与论文支点（1 周）

### S5.1 产出

1. **候选验证总表**: 每行一个候选，含:
   - CDR-H3 序列、来源
   - S1 表达状态（[VERIFIED]/[FALSIFIED]/[LOW YIELD]）
   - S2 BLI hit 状态（[VERIFIED hit]/[FALSIFIED no-hit]）
   - S3 SPR KD 值 + 拟合质量（[VERIFIED] with KD）
   - S4 特异性结论（[VERIFIED 特异]/[FALSIFIED 非特异]/[CROSS-REACTIVE]）
   - 最终标签: **[VERIFIED Aβ42 BINDER]** / **[FALSIFIED NO BINDING]** / **[CROSS-REACTIVE TOOL]**

2. **论文支点矩阵**:

| 验证结果 | 论文叙事 | 目标期刊层级 |
|----------|---------|------------|
| ≥1 KD<100 nM + 寡聚特异 | "BFN/template-seeded 方法产出了高效 Aβ42 寡聚体特异 binder" | Nature Biotech / Cell / Science |
| ≥1 KD 100 nM-1 μM + 特异 | "计算设计产出弱但特异的 Aβ42 binder，可做先导物优化起点" | Nat Commun / PNAS |
| ≥1 结合 + 仅交叉 | "计算设计产出 pan-amyloid 工具抗体，需进一步优化特异性" | ACS Chem Biol / J Biol Chem |
| 0 hit（全部 FALSIFIED） | "84 候选基于计算设计的湿实验验证结果——诚实负面，物理极限确认" | PLoS ONE / Sci Rep / 方法学 |

3. **诚实负面也有价值**: 即使 0 hit，论文可写"大规模 template-seeded + pliability-selected 候选在标准 BLI/SPR 下无 binder，揭示当前计算方法的极限"，配评估墙 6 路证伪数据，是正经负面结果论文。

---

## 2. 多意外全预案（M1/M2/M3）

### M1 表达/纯化意外

| 意外 | 响应步骤 | 不回退条件 |
|------|---------|----------|
| 表达率 < 50%（<13/25） | (1) 尝试 30°C 表达（降温促折叠）(2) 加分子伴侣（Hsp70 共表达）(3) 换 ExpiCHO（CHO 折叠更好） | 仍 < 50% → 启用大肠杆菌备选 |
| 聚集 > 30% 候选 | (1) 加 0.5M L-精氨酸 to 缓冲液 (2) SEC 切单体峰 (3) scFv→Fab 分别表达 | 仍聚集 → 该候选标 AGGREGATED |
| 产率 < 0.05 mg/mL | (1) 扩培养至 100-300 mL (2) 多批合并浓缩 (3) 若需极少量 SPR → 可降至 10 μL | 产率达标 → GO |
| CDR graft 破坏折叠（template-seeded 候选） | 该候选取消。不补救——template-seeded 折叠风险已知 | — |

### M2 结合意外（最关键）

| 意外 | 含义 | 响应 |
|------|------|------|
| **0 hit（S2 全阴）** | 计算候选无 Aβ42 结合 | **不做 S3-S4**。产 S5 负面报告。论文: "诚实负面 + 物理极限"。重审 pliability selection 是否筛掉了真正 binder。保底: 取原生 4HIX/5CSZ CDR 做阳性对照（若连原生抗体也不结合，是 Aβ42 制备问题） |
| **全部非特异（S4 发现）** | S2-S3 信号来自非特异结合 | (1) 加 0.5M NaCl 到缓冲液抑制静电非特异 (2) 加 0.1% Tween-20 抑制疏水非特异 (3) 用对照传感器（无 Aβ42）扣减——若非特异是传感器结合，应被扣减彻底 |
| **仅弱结合（全 KD > 1 μM）** | 方向对但亲和力低 | (1) 构建 scFv 二聚体（加 Cys 做 C 端二硫键链间交联—二价 avidity 增强表观亲和力）(2) 误差-prone PCR 做 CDR 突变库 → 再筛（亲和力成熟）(3) 论文诚实标"弱 binder，可优化" |
| **仅寡聚体结合** | 理想结果！ | 做 S4 形式偏好验证，论文强调寡聚特异——AD 治疗的核心需求 |
| **仅单体结合** | 单体表位 | 警示: 单体 Aβ42 是瞬间过渡态，生理相关性存疑。可做竞争实验（单体结合后，加寡聚体看是否竞争？） |
| **仅交叉（pan-amyloid）** | 跟多种淀粉样蛋白结合 | 检查 CDR 是否有 common amyloid-binding motif（带正电荷+芳香残基）。论文标"pan-amyloid 工具抗体"，或做突变去掉交叉 |
| **全强结合 + 无交叉** | 最佳意外 | 升为 lead 候选，直接进 S5 论文主打 |

### M3 资源/后勤意外

| 意外 | 响应 |
|------|------|
| HEK293 表达实验室资源紧张 | 外包给 CRO（如 Genscript / WuXi Biologics / Sino Biological）做表达纯化 |
| SPR/BLI 仪器不可用 | 委托大学共享平台或 CRO（各 ~¥5-15k/run） |
| Aβ42 批次无效 | 换供应商（Bachem → rPeptide → Anaspec），重新 HFIP 处理 |
| 经费有限 | **简版方案**: top25→top10（取 composite_score 最高的 10 条 + 3 阳性对照 = 13 条）。仅 BLI 初筛，不跑 SPR 定量。经费 ~¥5-8 万。0 hit 则停 |
| 时间紧迫 | S1 用大肠杆菌（1-2 周 vs HEK 2-4 周），S2-S3 合并（BLI 直接定量，跳过 SPR），S4 仅做最关键的反向筛选。总周期压缩至 6-8 周 |

---

## 3. 优先级与预算

### 方案 A: 标准方案（25 候选，推荐）

| 阶段 | 内容 | 周期 | 费用（估算） |
|------|------|------|------------|
| S0 | 序列补全（计算） | 2 天 | ¥0（计算端） |
| 基因合成 | 25 dsDNA 合成（外包） | — | ¥12k-18k（~¥500-700/条） |
| S1 | HEK293 表达 + 纯化 | 3-4 周 | ¥30k-50k（培养基+树脂+耗材+人工，约 ¥1.5-2k/候选） |
| S2 | BLI 初筛（Octet） | 1 周 | ¥8k-15k（机时+传感器+Aβ42+耗材） |
| S3 | SPR 定量（Biacore） | 1-2 周 | ¥10k-20k（仅 S2 hit，假设 3-5 hit） |
| S4 | 反向筛选 | 1 周 | ¥5k-10k（传感器+蛋白+耗时） |
| S5 | 报告 | 1 周 | ¥0（分析） |
| **合计** | | **10-12 周** | **¥65k-113k** |

### 方案 B: 精简方案（top10，经费受限时）

| 阶段 | 内容 | 周期 | 费用 |
|------|------|------|------|
| S0-精简 | top10 序列补全 | 1 天 | ¥0 |
| 基因合成 | 13 dsDNA（10 候选 + 3 阳性对照） | — | ¥7k-9k |
| S1-精简 | HEK293 表达 | 3 周 | ¥15k-25k |
| S2-精简 | BLI 初筛 + 直接定量（不跑 SPR） | 1 周 | ¥5k-8k |
| S4-精简 | 仅核心反向（BSA + αSyn + Tau） | 1 周 | ¥3k-5k |
| **合计** | | **8-10 周** | **¥30k-47k** |

### 方案 C: 最小验证（top5，纯保底）

如果仅需验证一个结论（"计算候选能/不能产生 binder"）:

| 内容 | 周期 | 费用 |
|------|------|------|
| top5 + 阳性 + 阴性 = 7 条 | 6-8 周 | ¥15k-25k |

---

## 4. 协作铁律

| 约定 | 原因 |
|------|------|
| **湿实验是准度真尺** | 评估墙 6 路证伪 + 计算端已到极限，不能替代实验 |
| **4HIX/5CSZ native CDR 每板阳性对照** | 确保 Aβ42 和仪器正常——阳性对照失败则该板全作废 |
| **Aβ42 严格 HFIP 单体起步 + ThT QC** | 聚集污染是 BLI/SPR 实验的头号毁因——单体 QC 必须 PASS |
| **计算排序仅参考，不预设湿实验结论** | 评估墙已死，pliability_score 不代表结合。0 hit 不意外 |
| **每候选三件套 [VERIFIED / FALSIFIED / PENDING]** | [[asset-verification-discipline]] 铁律，两次崩盘教训 |
| **0 hit 也是有效结论** | 正式负面结果。比"没跑"强百倍。可发论文 |
| **S0 产出是实验的先决条件** | 当前候选只有 CDR 序列没有 VH/VL——S0 产物含完整可合成序列 |

---

## 5. 最终产出物清单

| 阶段 | 产出 | 文件路径 |
|------|------|---------|
| S0 | 完整 scFv 氨基酸序列 (25) | `idp_design_results/s5_wetlab_synthesis_20260705/candidates_aa.fa` |
| S0 | 密码子优化的 DNA 序列 (25) | `idp_design_results/s5_wetlab_synthesis_20260705/candidates_dna.fa` |
| S0 | 候选注释表 | `idp_design_results/s5_wetlab_synthesis_20260705/candidates_annotation.tsv` |
| S0 | 框架缺失候选说明 | `idp_design_results/s5_wetlab_synthesis_20260705/unpaired_candidates.tsv` |
| S1 | 表达纯化 QC | SDS-PAGE/SEC 每候选图 + 表达成功率表 |
| S2 | BLI 初筛结果 | 原始 .frd 文件 + hit 列表 CSV |
| S3 | SPR 定量 | 感应曲线 .blr 文件 + KD 表 |
| S4 | 特异性矩阵 | 反向筛选 CSV + 表位竞争图 |
| S5 | 验证总报告 | `docs/wetlab_validation_report.md` + 每候选最终标签 |

---

## 6. 时间总表

| 阶段 | 周期 | 累计 | 依赖 |
|------|------|------|------|
| S0 序列补全 | 1-2 天 | 1-2 天 | — |
| 基因合成（外包） | 1 周 | ~1.5 周 | S0 FASTA |
| S1 表达纯化 | 2-4 周 | ~5.5 周 | DNA 到货 |
| S2 BLI 初筛 | 1-2 周 | ~7 周 | S1 纯化蛋白 |
| S3 SPR 定量 | 1-2 周 | ~9 周 | S2 hit 列表 |
| S4 特异性 | 1-2 周 | ~11 周 | S3 hit |
| S5 报告 | 1 周 | ~12 周 | S1-S4 全部 |

**到 25 候选验证结论的完整周期: ~12 周（~3 个月）。**

---

## 7. 一句话方案

> 25 计算候选（12 真实抗 Aβ + 7 template-seeded + 4 PDB + 2 跨 IDP），S0（计算端最后一步：逐条补全完整 scFv 序列，VH/VL 框架来自 PDB/文献/专利，scFv 格式 VH-(G₄S)₃-VL-His₆，密码子优化后送合成）→ S1 HEK293F 悬浮表达 + Ni-NTA 纯化 + SDS/SEC QC → S2 BLI 初筛（Octet SA 传感器，生物素-Aβ42 寡聚体，严格 HFIP 单体起步 + ThT QC 防聚集假阳性，4HIX/5CSZ 阳性+抗 HEL 阴性+缓冲液对照）→ S3 SPR 定量（Biacore CM5 + anti-His capture 倒置格式，1:1 Langmuir 拟合，χ²<10% Rmax）→ S4 反向筛选（BSA/转铁蛋白/αSyn/Tau 排除非特异 + Aβ42 单体/寡聚/纤维 3 形式偏好 + 表位竞争）→ S5 湿实验验证总报告，每候选标 [VERIFIED Aβ42 BINDER / FALSIFIED]。≥1 KD<1μM 特异 → 高刊方法学论文；0 hit → 诚实负面论文。铁律：阳性对照每板必含 + Aβ42 单体起步 + 计算排序仅参考 + 三件套全备。top25 ~12 周 ≈ ¥65k-113k。S0 是计算端能做的一切，之后转湿实验。
