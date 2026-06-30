# 下一阶段大方针 V4 — 从"事后评分 graft 设计"转向"让结合可发生、可度量"

**日期**: 2026-06-30 | **基线**: V17i 生成器(Δ+30pp) + L0 crystal 物理打分有效 + L3/L4 生成 pose 归零证伪
**交接**: DeepSeek V4

---

## 0. 战略级诊断(决定方针的总根因)

### 0.1 横跨两轮评估成立的硬事实

| 事实 | 证据 |
|---|---|
| 所有生成设计 AF2 ipTM ≈ 0.09(噪声) | P3 全规模 720 设计,commit 4567828 |
| L1-3 物理打分在生成 pose 归零(0.207) | commit 258ec0c |
| L4 ESM-IF 在生成 pose 归零,native=scram 同分 | commit c28c97d |
| 两条范式独立路径以**相同方式**失效 | 都被 scaffold backbone 主导 |
| AF2-multimer 对短肽-抗体重折叠系统性失效 | M0 闸门 native CDR 重 fold 仅 0.118 |

### 0.2 据此推出的战略级结论

> **度量无法区分生成设计,极可能不是因为"度量差",而是因为"生成设计之间没有可区分的结合信号"——它们大概率都不结合。** 在一堆不结合的设计上,任何度量都会归零(无信号可分)。L0 在 crystal 上有效,证明度量本身没问题;是**评估对象(graft-to-scaffold 的 CDR)本身没有结合信号**。

**因此方针反转**:不再"修度量去评既有设计",改为**"改设计与评估范式,让结合真正发生并变得可测"**。

---

## 1. 总方针:三支柱 + 一安全港

```
支柱 A (设计范式转移):  从零 graft CDR  →  以已知 Aβ 抗体 CDR 为种子的 redesign
支柱 B (评估范式再转移): scaffold 固定 backbone  →  template 条件化 / CDR 自折叠 / 序列级
支柱 C (可发表安全港):   disorder head 上 CAID 基准(独立于绑定成败)
```

> A 解决"有没有结合信号",B 解决"能不能测到信号",C 保底"无论 A/B 成败都有产出"。

---

## 2. 支柱 A:设计范式转移——Template-Seeded Redesign

### 2.1 为什么要换设计范式

从零 graft CDR 的全部产物 AF2≈0.09 且度量归零,强烈提示这些 CDR **没有真实结合**。而项目已有已知 Aβ 抗体 CDR 作为种子资产:
- 4HIX(solanezumab)+ KLVFFAED
- 5CSZ(gantenerumab)
- 已在 ceiling 结果里用过(3STB_4HIX, 3STB_5CSZ)

**从"已知能结合的 CDR 骨架"出发做局部 redesign,改造可被度量,因为起点就有结合信号。**

### 2.2 具体做法

1. 提取 4HIX/5CSZ 的 native CDR + 配体肽 crystal 界面坐标(已有效,L0 在此 d=2.545);
2. 保留界面热点接触残基(paratope-epitope contact residues 冻结),仅 redesign 非热点 CDR 位点;
3. BFN/MPNN 在"冻结热点 + 自由位点"约束下生成变体;
4. 验收:**变体相对 native 应有可测的界面变化**(接触数/电荷/界面疏水),而非全部趋同。

### 2.3 与从零 graft 的根本区别

| | 从零 graft(现行) | template-seeded(新) |
|---|---|---|
| 起点 | 无结合,纯 scaffold backbone | 已知结合界面 |
| 结合信号 | 无(度量归零) | 有(可继承 + 局部扰动) |
| 度量可分性 | 归零(L3/L4 证伪) | 预期可分(有信号) |

---

## 3. 支柱 B:评估范式再转移(三层降级,逐个验证)

> 不押单一评估法。三个范式都规避"graft-to-scaffold 固定 backbone"这个共同失败的根因。

### B1 ★(最高优先)Template-Conditioned AF2 —— 修 M0 盲重折叠失效

**根因对症**:M0 失效是因为 AF2 **盲重折叠**短肽-抗体(c29 commit 已证)。ColabFold 支持 **template 注入**:把 crystal 抗体-肽复合物坐标作 template 喂入,AF2 只需预测 CDR 突变对界面的影响,**不需要盲猜肽构象**。

- 具体操作:对每个 template-seeded 变体,用 4HIX/5CSZ crystal 作 template 跑 ColabFold-multimer;
- 验收:M0 闸门在 template 模式下 native ipTM > 0.6 且 native vs scram 可分(此前无 template 时是 0.118/不可分);
- 这是**唯一同时修了 M0 失效 + 给生成设计以 crystal 级坐标的杠杆**,且资产就绪(WSL colabfold 1.6.1)。

### B2 Fold-CDR-Then-Score —— 上轮未试的路标

AF2/ESMFold **只折叠抗体单链**(不给肽,L5 已能做,plddt=0.371),取**折叠出的 CDR loop 真实坐标**,在该真实坐标(而非 scaffold 固定 backbone)上重跑 L4/L1-3。
- 验收:此时 native vs scram 能否重新分开(crystal 上能分,scaffold 上归零,fold 后是关键判据);
- 若 B1 已可分,B2 降为佐证;若 B1 不可分,B2 是退路。

### B3 序列级度量 —— 彻底规避结构预测

对已知 Aβ 抗体 CDR 的**序列恢复率 / 相似度**(blosum、CDR 长度、热点 motif 保留),不依赖任何结构预测。
- 完全不受 AF2/graft backbone 任何缺陷影响;
- 作为"设计是否朝已知好抗体靠拢"的廉价硬指标,与 B1/B2 交叉验证。

### B 的判据矩阵

| B 层 | 通过判据 | 角度 |
|---|---|---|
| B1 template AF2 | native ipTM>0.6 且 native vs scram 可分 | 真实结合预测 |
| B2 fold-then-score | CDR 真实坐标上 native vs scram 可分 | 结构物理 |
| B3 序列级 | 变体朝已知抗体 CDR 收敛 | 序列先验 |

---

## 4. 支柱 C:可发表安全港——disorder head CAID 基准

**无论 A/B 成败,这是独立可发表产出。** 项目对外核心差异化能力(disorder 识别)至今未量化。RESULTS §8 已定位为"唯一已成立的差异化能力"。
- 上 CAID 基准 vs IUPred3 / AF2-pLDDT,报 AUC-ROC/PR;
- 不依赖抗体绑定成败,2 周可成;
- 这是项目保底的方法学贡献。

---

## 5. Go/No-Go 决策点(诚实止损)

在 **A(template-seeded) + B1(template AF2)** 跑通后,有三种结局:

| 结局 | 数据 | 决策 |
|---|---|---|
| **GO** | 变体在 B1 template-AF2 下 ipTM 可分、且 >0.4 | 路线成立,扩规模,冲可用抗体 |
| **条件 GO** | B1 不可分但 B2/B3 可分 | 用 B2/B3 作度量,A 继续,湿实验验证 |
| **NO-GO** | A 起点即结合,但所有 B 层都归零 | 纯计算在 IDP-scaffold 抗体设计上达极限 → 转湿实验闭环,或放弃从零设计、退回 native-ish |

**关键**:NO-GO 也是有效结论——它界定了纯计算评估的能力边界,本身可写进论文(limitation)。不要无限延期。

---

## 6. 执行优先级与排期(RTX 5060 + WSL GPU)

| 优先级 | 任务 | 依赖 | 周期 |
|---|---|---|---|
| **P0** | A: 4HIX/5CSZ 热点冻结 + BFN/MPNN template-seeded redesign | 已有 crystal | 3-4 天 |
| **P0** | B1: ColabFold template 注入,重跑 M0 闸门 | WSL colabfold | 2-3 天 |
| **P0** | C: disorder head CAID(安全港,并行) | 无 | 1 周 |
| **P1** | B3: 序列级度量,交叉验证 | A 产出 | 2 天 |
| **P1** | B2: fold-then-score(B1 不可分时启用) | L5 已能 fold | 3 天 |
| **P2** | Go/No-Go 评审 + 路线定稿 | A+B | 1 天 |

**串行硬约束**:B1 不通过(M0 在 template 模式仍不可分)前,不扩规模设计。

---

## 7. DeepSeek V4 协作约定

| 约定 | 原因 |
|---|---|
| **不再在"scaffold 固定 backbone + graft CDR"坐标上跑任何新度量** | L3/L4 已证归零,根因统一 |
| **新设计必须从已知 Aβ 抗体 CDR 种子出发,不从零 graft** | 从零产物无结合信号,度量必然归零 |
| **评估必须用 template-conditioned 或 fold-then-score,禁用盲 AF2 重折叠** | M0 + c28c97d 双重证伪盲重折叠 |
| **每轮同时报:B1 ipTM / B2 物理 / B3 序列 三层**,单层归零不慌、全归零才触发 No-Go | 多证据、避免单度量盲判 |
| **disorder CAID 与绑定主线并行,不绑死** | 保底产出 |
| **到 Go/No-Go 点必须给出明确决策,不延期** | 诚实止损 > 无限调参 |

---

## 8. 本方针与前序的关系

- V3(IDP_METRIC_PLAN_V3):押"给定坐标物理打分"→ 被 c28c97d 证伪(graft backbone 主导);
- 本 V4:承认"给定 scaffold 坐标"范式失败,**改设计起点(template-seeded)+ 改评估坐标(template/fold)**,让结合信号先存在、再被测;
- 不推翻 V17i 生成器——它仍是 redesign 阶段的生成引擎,只是**喂它有结合信号的种子,而非从零 graft**。

---

## 附:一句话方针

> **不再"对不结合的设计修度量"。改让设计从已知结合抗体种子出发(template-seeded),并用 template 条件化 AF2 / CDR 自折叠 / 序列级三层度量去测——先有结合信号,再谈度量。disorder CAID 并行作保底产出。到 Go/No-Go 点诚实判定,不延期。**
