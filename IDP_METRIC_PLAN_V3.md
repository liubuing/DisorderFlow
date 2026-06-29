# IDP 抗体设计:无需 HDOCK 的可靠度量与备选方案 V3

**约束**:HDOCK 二进制在 WSL2/Windows 装不下来,**且不再花精力装它**。
**目的**:在没有 HDOCK 的前提下,真正建立一个**能在生成设计上区分 native vs scrambled、且精度够高**的 IDP 抗体度量,从而让"设计进步可被证明"。
**交接**:DeepSeek V4

---

## 0. 重新定位问题根因(决定方案走向)

### 0.1 M0 闸门的精确含义(被前面误读)

M0 跑的是:`native Fv 序列(VH+VL) + 肽段` → **从序列重新交给 AF2-multimer 折叠** → ipTM。

| 输入 | ipTM | 含义 |
|------|------|------|
| native CDR + 肽段(AF2 重 fold) | 0.118 | AF2 重新预测短肽-抗体复合物 |
| native CDR + 全长 Aβ42(AF2 重 fold) | 0.144 | AF2 重新预测短肽-全长IDP |
| scrambled CDR + 肽段(AF2 重 fold) | 0.116 | 乱序对照 |

**关键纠偏**:这不是"AF2 对什么设计都给 0.09 无区分力"那么简单——而是 **AF2-multimer 对"短肽(8aa) + 抗体 Fv"这类复合物的从头预测整体就弱**。即使 4HIX 这种 crystal 里真实结合的抗体,native CDR 被 AF2 重 fold 也只得 0.14。这是 AF2 家族的**已知系统性缺陷**(短链 + multimer PAE 天然噪声大),不是 DisorderFlow 的 bug。

### 0.2 因此度量必须换范式

```
旧范式(失败):  设计序列 → AF2-multimer 重新折叠 → ipTM      ← AF2 对短肽-抗体系统性失效
新范式(V3):    设计序列 + 构象库 → 对接/物理打分(给定坐标)  ← 不依赖 AF2 重新折叠
```

**核心原则**:不再让任何模型"重新猜结构",而是**给定结构(构象库或对接 pose)做物理打分**。这是 HDOCK 之外唯一稳健的路,且项目已有全部资产。

---

## 1. 方案总览:五层度量,三层备选

> 不押单一工具。一个主路径 + 两个独立备选,任一成立即可解锁后续。**全部纯 Python,零二进制依赖**。

```
                   ┌─ 主路径 L1-L3(纯Python物理打分,无依赖,立即可用)
度量 = 给定坐标 ────┼─ 备选 A:L4 ESM-IF 反向折叠一致性(蛋白模型,纯权重)
                   └─ 备选 B:L5 AF2-单链 pLDDT(规避multimer陷阱,复用已有WSL)
```

| 层 | 度量 | 依赖 | 状态 | 角色 |
|----|------|------|------|------|
| L0 | **有效性闸门**:native vs scrambled 可分? | 已有资产 | 待跑 | **必须先过,否则一切作废** |
| L1 | contact-count(Cb<8Å)+ 电荷 + 疏水 | `idp_dock_score.py` ✅ | 已实现 | 主路径基础 |
| L2 | FFT 刚体对接搜索 pose | `fft_dock.py` ✅ | 已实现(未启用) | 生成多 pose |
| L3 | BSA + shape complementarity + EC | `interface_scorer.py` ✅ | 已实现 | 主路径精化 |
| L4 | ESM-IF 反向折叠似然(序列↔结构一致性) | ESM(纯权重) | 待装(pip) | 备选A,独立 |
| L5 | AF2 **单链** pLDDT(paratope 折叠性) | WSL colabfold ✅ | 已可用 | 备选B,补短链缺陷 |

---

## 2. L0 有效性闸门——当前唯一 P0,且不需装任何东西

> **这是整个方案的第一性任务。** 在没有证明度量可分之前,任何后续都是盲跑。

### 2.1 为什么 L0 能成立(而 AF2 M0 不能)

AF2 M0 把序列重 fold,因短肽 multimer 失效。**L0 不重 fold**——直接用 crystal 真实坐标打分:

| 测试组 | 构造方式 | 预期 |
|--------|---------|------|
| native(n=4: 4HIX/5CSZ/...) | crystal 真实复合物坐标 → L1/L3 打分 | 高分 |
| scrambled(各自 n×3) | 把 native CDR 乱序 graft 回同坐标 → 打分 | **应显著低** |
| decoy(n) | 随机/退化 CDR graft → 打分 | 最低 |

因为 CDR 残基变了但**骨架/对接几何不变**,纯接触物理量(电荷互补、疏水吻合、接触数)在 native 上必然优于 scrambled——**这是物理必然,与 AF2 是否会 fold 无关**。

### 2.2 L0 判据(严格)

- **主判据**:L3 composite 在 native vs scrambled 上 **Mann-Whitney U, p < 0.05,且 AUC > 0.7**。
- **辅判据**:L1 contact-count 的 native 中位数 > scrambled 中位数(已由 commit `d6f16e7` 在 crystal 上初步确认"native vs scrambled 可分")。
- 全部 4 个已知抗体(4HIX/5CSZ/3STB/...)都要过。

### 2.3 L0 输出

```
若通过(p<0.05, AUC>0.7) → L3 composite 确立为 IDP 主度量,进入第3节生成评估流程
若不通过 → 进入备选 A(L4),L0 改用 ESM-IF;再不行 → 备选 B(L5)
```

**这就是"备选 + 真能达到目的"**:不押 HDOCK,押物理打分;物理打分若在 crystal 上都不可分,再退 ESM-IF;再退 AF2 单链。三层降级,总有一层成立。

---

## 3. L0 通过后的生成设计评估流程(主路径落地)

### 3.1 单设计评分流程(每个 BFN/MPNN 生成的 CDR)

```
1. graft CDR 序列 → scaffold backbone(CDR 段只换侧链,backbone 不动)
   ↓
2. 取 Aβ 表位构象库(5-seed,P2 已建)→ 对每个构象:
     2a. FFT 刚体对接搜索(L2)→ top-K pose
     2b. 每 pose 做 L1+L3 物理打分
     ↓
3. 跨构象鲁棒分 = percentile(L3_score, 25)  [不取均值,取最差四分之一]
   ↓
4. 报告三元组:  (L3_contact_score, robust_p25, BFN_OC[降权标记])
```

**为什么 L2 FFT 是 HDOCK 的正确替代**:
- HDOCK 本质=刚体对接+FFT 平移搜索+物理打分。`fft_dock.py` 已用 NumPy FFT 实现了**平移搜索 + 离散旋转采样**,这正是 HDOCK 的核心算法。
- 上次被废(commit `aace49a` 标 "reference")是因为**直接当打分器用错了**——FFT 给的是 pose,不是质量分;质量分应由 L1/L3 在 pose 上算。**两层解耦**是关键修正。
- 纯 Python + scipy,**零二进制**,正是约束所求。

### 3.2 群体评估流程(库筛选 top-N)

对 P3 的 720 设计库:
- 用 L3 跨构象鲁棒分**重新排序**(替代当前未验证的 MPNN+diversity composite);
- 报告 top20 与旧 composite 的排序重合度(Kendall τ),看是不是选出了完全不同的设计;
- 若旧 composite 高分在新度量下也高 → 旧 composite 被部分证实;若**不重合** → 旧 composite(多样性冒充质量)被证伪,以新度量为准。

---

## 4. 备选方案 A:L4 ESM-IF 反向折叠一致性

> 完全独立的第二条路径,不依赖对接几何,可作为 L3 的交叉验证或降级备选。

### 4.1 原理

ESM-IF(已开源,pip 装)给定**结构 + 序列**输出序列似然。思路:
- 对 graft 后的抗体 structure + 设计 CDR 序列,ESM-IF 给 CDR 段的对数似然;
- **好 CDR = 与该 backbone 结构自洽的序列** = 高似然;
- scrambled/退化 CDR = 与结构不自洽 = 低似然。

### 4.2 与主路径正交的价值

- L3 测"界面物理合理性",L4 测"序列-结构自洽性"——**两个独立信号**;
- 若 L3 + L4 排序一致 → 强证据;若冲突 → 提示设计在某维度有缺陷(如物理 OK 但序列不自然)。
- 完全规避 AF2 shortpeptide 缺陷与对接,纯序列-结构评估。

### 4.3 判据

ESM-IF 似然在 native vs scrambled 上 AUC>0.7 即可作独立有效度量。

---

## 5. 备选方案 B:L5 AF2 单链 paratope 折叠性

> 复用已成熟的 WSL ColabFold,但**规避 multimer 短肽陷阱**。

### 5.1 原理与 multimer 的区别

AF2-multimer 对短肽+抗体弱,**但 AF2 对单链抗体自身折叠 pLDDT 是可靠的**。思路:
- 只 fold **抗体单链**(不给肽),取 **CDR 段 pLDDT**;
- 好 paratope = CDR 能稳定折叠成结合态 loop = 高 pLDDT;
- 退化 CDR(TTTTTT)= 无法折叠 = 低 pLDDT。
- **这正好对治 §4 RESULTS 的失效 C(低复杂度退化序列)**,且 V17i 的 spearman −0.71 头对退化序列误判,单链 pLDDT 是独立纠偏。

### 5.2 不重 fold 复合物 = 规避 M0 失效根因

M0 失效是因为重 fold 短肽-抗体复合物。L5 只重 fold 单链抗体(它在 PDB 训练分布内),pLDDT 可信。

### 5.3 角色

不是首选项(它不直接测"结合"),但作为"CDR 可折叠性"过滤是可靠且便宜的,与 L3/L4 组成多维度。

---

## 6. 度量组合与最终判据(多证据融合)

单一度量都可能在某场景失效。**最终用三证据投票**:

| 证据 | 来自 | 测什么 |
|------|------|--------|
| E1 界面物理 | L3 跨构象鲁棒分 | 结合几何合理性 |
| E2 序列自洽 | L4 ESM-IF | CDR-structure 自洽 |
| E3 折叠性 | L5 AF2单链CDR pLDDT | paratope 稳定性 |

**设计保留判据**:E1 高 且 (E2 或 E3 高)。这避免任一单度量的伪阳性。
**最终排名**:以 E1 为主排序,E2/E3 作 hard filter。

> 这才是"精度够高"的含义——不是单指标高,而是**多独立证据收敛**。三证据来源完全独立(对接物理 / 反向折叠序列 / 结构预测折叠),互为正交验证。

---

## 7. 执行计划与优先级(RTX 5060)

| 优先级 | 任务 | 依赖 | 周期 | 产出 |
|--------|------|------|------|------|
| **P0** | L0 闸门:4 已知抗体 native vs scrambled,L1/L3 可分检验 | 现有资产,零安装 | **2 天** | 度量有效性证明 |
| **P0** | 提交 RESULTS.md + IDP_DESIGN_SCHEME_V2.md + 本文档入 git | 无 | 0.5 天 | 决策留痕 |
| **P1** | L0 通过→L2 FFT + L3 跨构象鲁棒分,重排 P3 720 设计库 | L0 | 3–4 天 | 可信 top20 + 旧 composite 验证 |
| **P1** | 备选A:L4 ESM-IF,跑 native vs scrambled AUC | pip install | 2 天 | 第二独立度量 |
| **P2** | 备选B:L5 AF2单链 CDR pLDDT 过滤器 | WSL 已有 | 1–2 天 | 退化序列过滤器 |
| **P2** | 三证据融合 + 重训/重排生成 | L2-L5 | 1 周 | 最终候选 |
| **P3** | P5 CAID(disorder head 基准) | 无 | 1 周 | 独立可发表 |

**串行硬约束**:**L0 不通过禁止进 P1**。这是避免 V11→V17i 盲跑重演的护栏。

---

## 8. 为什么这套方案"真能达到目的 + 够高精度 + 有备选"(论证)

### 8.1 "真能达到目的"
- L0 用 crystal 真实坐标打 native vs scrambled——物理必然可分(电荷/疏水/接触在真实结合界面优于乱序),**不依赖 AF2 会不会 fold**;
- 直接绕过 M0 失效根因(AF2 重 fold 短肽复合物),改走"给定坐标打分"。

### 8.2 "够高精度"
- L3 是 BSA + shape complementarity + 电荷互补三项物理量,等于 HDOCK 打分函数的核心子集,但纯 Python 实现;
- 跨构象取 p25 鲁棒分,对 IDP 构象系综稳健;
- 三证据独立收敛(E1物理∩E2序列∪E3折叠),单一失效不拖垮整体。

### 8.3 "有备选"
- 主路径 L1-L3 失败 → 备选A L4 ESM-IF(完全独立范式);
- 备选A 失败 → 备选B L5 AF2单链(规避multimer);
- 三路径正交,概率上至少一条成立的把握很高;
- **关键:全部零二进制**,HDOCK 装不下不再构成阻塞。

### 8.4 与前几版方案的本质区别
- V2(IDP_DESIGN_SCHEME):仍以 AF2 验证为核心 → 被 M0 证伪;
- V3 本版:承认 AF2 重 fold 在短肽-抗体不适用,**改"给定坐标物理打分"范式** + 多证据投票 + 三层降级。

---

## 9. DeepSeek V4 协作约定

| 约定 | 原因 |
|------|------|
| **先跑 L0,native vs scrambled 不可分则停** | 度量有效性是其余一切的前提 |
| **禁止再用 AF2-multimer 重 fold 短肽-抗体做主判据** | M0 已证伪(AF2 对此类复合物系统性失效) |
| **FFT(L2)只产 pose,质量分由 L3 算** | 两层解耦,修正上次把 FFT 当打分器用错的教训 |
| **三证据用投票,不简单加权** | 单一证据在某场景必失效,投票更鲁棒 |
| **每跑一步留 JSON + 进 git** | 避免 P3 那种 14 个重复未提交 JSON |
| **不花时间装 HDOCK** | 已确定装不下;纯 Python 资产已覆盖 HDOCK 核心算法 |

---

## 附:资产现状速查

| 文件 | 能力 | 状态 |
|------|------|------|
| `modules/idp_dock_score.py` | contact+电荷+疏水+shape, Cb<8Å | ✅ 已实现,L1 |
| `modules/fft_dock.py` | NumPy FFT 刚体对接+旋转采样 | ✅ 已实现,L2(待启用) |
| `modules/interface_scorer.py` | BSA + shape compl + EC | ✅ 已实现,L3 |
| `modules/idp_multiconf_scorer.py` | Kabsch 叠合多构象 | ✅ 已实现,P2 框架 |
| `data/abeta_conformations/` | Aβ42 5-seed | ✅ |
| `data/anti_abeta_refs/4HIX,5CSZ...` | 已知抗体 crystal | ✅ L0 基准 |
| WSL `colabfold 1.6.1 + alphafold` | AF2 推理 | ✅ L5 |
| ESM-IF | 反向折叠 | ⬇️ pip 装,L4 |
| HDOCK | — | ❌ 不装,不再投入 |
