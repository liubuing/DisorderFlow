# 齐鲁工业大学本科生 Nat Biotech 路线图 — V1

**背景**: 齐鲁工业大学（山东省科学院），本科在读 | **计算资产**: 25 候选 scFv 序列（S0 已完成）| **目标**: Nature Biotechnology 抗体设计论文 | **日期**: 2026-07-06

---

## 0. 先确认三件事（本周，不花一分钱）

### 0.1 校内蛋白表达能力

去**食工楼 A 座**，找生物工程学院的仪器平台管理员问三个问题：

1. AKTA 纯化仪现在什么型号？（2017 年是 AKTA start，现在可能升级了）
2. 有没有人做过**哺乳细胞蛋白表达**（HEK293 悬浮培养）？还是只做大肠杆菌？
3. 能不能预约用？

校内大概率有大肠杆菌表达经验，HEK293 不好说。但 AKTA + 大肠杆菌就够做好第一步了：scFv 在大肠杆菌里表达虽然会形成包涵体，但可以复性——复性后的 scFv 做 BLI 初筛是够用的。

### 0.2 济南的 SPR/BLI 仪器

电脑上打开 **山东省大型科研仪器共享平台**:
> https://dygx.kjt.shandong.gov.cn:8086/

搜索以下关键词，看济南哪家有：
- "Biacore" → SPR 定量 KD
- "Octet" → BLI 初筛（比 SPR 便宜、通量更高）
- "分子相互作用" → 通用搜索
- "Nanotemper" → MST（SPR 替代方案）
- "ITC" → 等温滴定量热（替代方案）

记录每家单位名、仪器型号、预约方式、机时费。这个平台可以用**山东省创新券**补贴费用。

### 0.3 找一位支持你的导师

在生物工程学院找一位做**蛋白质工程/酶工程/发酵工程**方向的老师。不需要他是抗体专家——你需要的是：

1. 有人在学院内部帮你协调仪器使用（AKTA、细胞培养室）
2. 有人帮你签字申请创新券/报销
3. 论文挂名（通讯作者）

带着你的 S0 产物（25 条完整 scFv 序列 + 候选注释表）和分析去聊——比空手去有效得多。

---

## 1. 路线总览

```
Phase 1  ──── Phase 2  ──── Phase 3  ──── Phase 4
校内表达      BLI初筛        SPR定量      写论文
1-2个月       1-2周         1-2周        2-3个月
~¥5k-10k     ~¥3k-5k       ~¥5k-10k      ¥0
```

**全部周期**: 5-8 个月 | **总预算**: ¥20k-50k（用创新券补贴后可能更低）

### 关键决策点

| 节点 | 如果 GO | 如果 NoGo |
|------|---------|-----------|
| Phase 2 BLI | ≥1 hit → Phase 3 SPR 定量 | 0 hit → 诚实负面，投 Nat Comput Sci |
| Phase 3 SPR | ≥1 KD < 100nM → 全力冲 Nat Biotech | 全弱 → 降 Nat Commun / Comm Biol |

---

## 2. Phase 1: 蛋白表达纯化（1-2 个月，¥5k-10k）

### 方案 A: 校内自己做（推荐，最省钱）

**如果学院有 HEK293 悬浮培养能力**：

1. **基因合成**（外包，1 周）
   - 发 `candidates_dna.fa`（S0 已生成）给金斯瑞/Twist/金唯智
   - 25 条密码子优化的 scFv DNA，克隆到 pcDNA3.4-TOPO 或 pET 载体
   - 费用: ~¥500-700/条 × 25 = ¥12k-18k
   - **省钱**: 先只合成 top10，¥5k-7k

2. **小规模表达测试**（2 周）
   - 用学院细胞培养室 + 生物安全柜
   - Expi293F 悬浮细胞（ThermoFisher，约 ¥2k/kit）
   - PEI 转染（Polyplus PEIpro，约 ¥1k/kit）
   - 30 mL 规模 × 10 候选 = 够测 BLI

3. **纯化**（1 周）
   - 食工楼 A 座 AKTA start + Ni-NTA 柱
   - His-tag 一步纯化 → PBS 缓冲液置换
   - SDS-PAGE + Nanodrop QC

**如果学院只做大肠杆菌**：

1. 基因克隆到 pET-22b(+) 载体（pelB 信号肽分泌表达或包涵体复性）
2. BL21(DE3) 表达 → 包涵体洗涤 → 尿素溶解 → 梯度透析复性 → Ni-NTA 纯化
3. 大肠杆菌成本 ~¥2k-3k（LB 培养基 + IPTG + 树脂自己装柱）
4. **风险**: 部分 scFv 复性后可能不折叠或聚集——SEC 筛选单体峰

### 方案 B: 外包给金唯智济南分公司

金唯智（GENEWIZ）在济南有分公司，提供基因合成→表达→纯化一条龙：
- 基因合成: ¥500-700/条
- 小规模表达纯化（scFv, 1-3 mg）: ¥2k-4k/条
- 10 条 ≈ ¥25k-47k（贵但省时间+保质量）

**推荐**: 先校内做 top10（买基因+自己纯化，¥7k-10k），如果表达不畅就外包。

---

## 3. Phase 2: BLI 初筛（1-2 周，¥3k-8k）

### 在哪做

用山东省仪器共享平台搜的结果，选最近、最便宜的一家（山大/省药科院/第一医科大）。

**优先找 Octet（BLI）而不是 Biacore（SPR）**——BLI 更快更便宜，初筛 25 样半天跑完。SPR 贵且慢，留给定量。

### 怎么做

找提供 Octet 的单位，问：
1. 有没有 SA 传感器？（链霉亲和素——固定生物素化 Aβ42 用）
2. 收费标准？（通常 ¥200-500/小时，或 ¥50-200/样品）
3. 能不能帮忙做样品准备/数据分析？还是只能自己操作？

### 实验设计（精简版，top10）

- 10 候选 + 1 阳性（4HIX native scFv）+ 1 阴性（缓冲液）= 12 样
- 生物素-Aβ42 寡聚体固定到 SA 传感器
- 单点动力学（60s 结合 + 60s 解离）
- 3 复孔
- Aβ42 从公司买预制的（Bachem 或 rPeptide，~¥2k-3k/1mg），避免自己制备的坑

### 如果济南找不到 Octet/Biacore

**MST（微量热泳动）作为替代**:
- Nanotemper Monolith 比 Octet 更普及（很多分子互作实验室有）
- 不需要固定（天然状态测结合，更适合 Aβ42 这种易聚集的）
- 需要荧光标记候选 scFv（Cy5-NHS 或 His-tag 标记试剂盒，~¥1k-2k）
- 对 KD 定量精度略逊 SPR 但在 nM-μM 范围足够

### Phase 2 Go/NoGo

- **1+ hit**: → Phase 3
- **0 hit**: 停。回去写 Nat Comput Sci。诚实负面是正经论文。

---

## 4. Phase 3: SPR 精确定量（1-2 周，¥5k-10k）

只做 Phase 2 的 hit 候选（1-5 条）。

在仪器共享平台找 Biacore T200/8K：
- 全套动力学（Kon/Koff/KD）
- 6 浓度梯度（0.1 nM - 100 nM）
- 1:1 Langmuir 拟合
- 反向筛选（Aβ vs BSA vs αSyn vs Tau）

### 如果不敢赌 SPR

BLI 也能做定量——Octet 的多浓度动力学精度不如 Biacore，但对于 nM 级别 KD 是够的。Nat Biotech 不一定要求 SPR。

---

## 5. Phase 4: 论文（2-3 个月，¥0）

无论 Phase 2 结果如何，论文叙事都已备好：

| 结果 | 期刊 | 叙事 |
|------|------|------|
| 1+ KD < 100nM + 特异 | **Nature Biotechnology** | "AI-designed anti-Aβ antibody binder validated by SPR — an undergraduate-led cross-institutional effort" |
| 1+ KD 100nM-1μM | **Nature Communications / Science Advances** | "BFN-based antibody design produces weak but specific Aβ binders — proof of concept" |
| 0 hit | **Nature Computational Science** | "Computational limits of fixed-backbone CDR design for disordered targets — evidence from 6 cascade evaluation walls and wet-lab screening" |

**Nat Biotech 需要的额外配图**:
- 一张 SPR 传感图（浓度梯度曲线）
- 一张 SEC-HPLC（证明 scFv 单体）
- 候选 vs 阳性对照（4HIX native）的亲和力对比
- Pipeline 示意图：计算→湿实验→hit 的全链路

---

## 6. 经费方案

### 最小可行方案（¥12k-18k）

| 项目 | 金额 |
|------|------|
| 基因合成 top5（含阳性对照） | ¥3k-4k |
| 大肠杆菌表达 + Ni-NTA 自纯化 | ¥2k-3k |
| BLI 租机时（Octet, 半天, 租用平台） | ¥3k-5k |
| Aβ42 肽 + 耗材 | ¥3k-5k |
| 预留 | ¥1k-2k |

### 资金来源

1. **大创项目**（国家级/省级）: ¥5k-20k，本科生标配
2. **山东省创新券**: 补贴仪器使用费 30%-60%
3. **导师课题经费**: 如果你找到支持你的导师
4. **学院本科生科研基金**: 省级双一流学科通常有

---

## 7. 时间表

```
2026年7月
├─ W1: 确认校内平台 + 找导师 + 在共享平台搜 SPR/BLI + 申请大创
├─ W2: 下基因合成订单(top10)
│
2026年8月
├─ 表达测试 + 纯化 + QC (SDS-PAGE/SEC)
│
2026年9月
├─ BLI 初筛（最重要的一步——决定一切）
│
2026年10月
├─ 若 hit: SPR 定量 + 反向筛选
├─ 若 0 hit: 开始写 Nat Comput Sci 论文
│
2026年11-12月
├─ 写论文 + 投稿
│
2027年
├─ 审稿 → 修改 → 接收
```

---

## 8. 一句话

> 去食工楼 A 座找做蛋白的老师聊、去山东省仪器共享平台搜 Octet、下 10 条基因合成订单——这三件事本周就能做，不花一分钱。接下来 3 个月表达+BLI，9 月就知道结果。有 binder → Nat Biotech 可冲。没有 → Nat Comput Sci 也够一个本科生吹一辈子。关键是第一步走出去。

---

*方案配套资产*:
- 25 候选 scFv 序列: `idp_design_results/s5_wetlab_synthesis_20260705/candidates_dna.fa`
- 候选注释表: `idp_design_results/s5_wetlab_synthesis_20260705/candidates_annotation.tsv`
- 湿实验完整方案: `WETLAB_VALIDATION_PLAN_V15.md` / `.docx`
