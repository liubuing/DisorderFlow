# Fold-then-Dock 方案 V11 — 破"CDR 侧骨架支配"的唯一物理路径

**交付执行**: DeepSeek V4 | **日期**: 2026-07-02 | **目标**: 让 HDOCK 能区分不同 CDR 设计(当前 graft 直对接 RMSD=0 归零)
**核心认知**: HDOCK CDR 侧归零不是 bug,是刚体对接对"仅侧链突变"的物理结构性盲视。修复在输入侧——给 CDR 不同 backbone。

---

## 0. 问题根因(已核实)

### 0.1 实测证据
- **graft 直对接**:12 设计 CDR backbone CA RMSD = **0.0000**(全同,`graft_cdrs_to_pdb` 只换残基名) → HDOCK score 全同归零;
- **fold 后对接**:8 设计 CDR backbone 两两 RMSD = **9.3–30.6Å(均值 19.4Å)** → 物理上保证 HDOCK 有可分信号。

### 0.2 物理结构约束
- HDOCK FFT 网格 1.2Å,刚体对接评分骨架主导;
- 侧链突变(0.1-0.5Å)在网格上"看不见" → graft 直对接归零;
- backbone 差异(19Å)远超网格 → fold-then-dock 天然有信号;
- **这是范式物理属性,改 HDOCK 参数无用,只能改输入**(让 CDR 有不同 backbone)。

### 0.3 但 fold 后 19Å 存在风险
fold 出的是 **apo 结构**(无抗原),CDR loop 可能过度展开/错位,差异过大不代表结合态。需约束:**只取 fold 的 CDR loop 局部坐标,graft 回 scaffold 原位**(而非用整个 fold 的抗体),保持 scaffold 框架不变、只让 CDR loop 几何随序列变。

---

## 1. 总目标与判据

**目标**:验证 fold-then-dock 能否让 HDOCK 区分 native vs scrambled vs 方式4 设计的 CDR 质量。

**判据**:
| 结局 | 数据 | 决策 |
|---|---|---|
| **GO** | HDOCK score 区分 native vs scram(p<0.05),且方式4 设计 score 合理(>scram,接近native) | fold-then-dock 成立,方式4 候选可纯计算排序 |
| **部分** | native vs scram 可分但方式4 设计分布不合理 | 输入构造需调(见 §3 风险),迭代 |
| **不成立** | fold 后仍不可分 | fold 出 apo loop 差异过大无意义,确认纯计算挑 CDR 极限 |

---

## 2. 执行步骤

### Step 0:CDR loop 局部 graft(关键设计,控 19Å 风险)
**问题**:直接用 fold 的整个抗体会让 scaffold 框架也变(框架本该固定)。
**做法**:
1. 对每个设计,AF2 单链 fold(Blackwell 可用,L5 已通);
2. 从 fold 结构提取 **CDR loop 段的 backbone(N/CA/C/O)**;
3. 将该 loop 段 graft 回 **原 scaffold 的 CDR 位置**(框架区保留原 scaffold 坐标);
4. 输出 receptor.pdb:框架=scaffold(固定),CDR loop=fold 出的真实几何(随序列变)。
- 这样 CDR loop backbone 不同(给 HDOCK 信号),框架不变(保持可比性)。

**验收**:不同设计 receptor 的框架 CA RMSD≈0,CDR loop CA RMSD>1Å。

### Step 1:Native vs Scram 判据(决定性,先跑)
对 4HIX/5CSZ:
- native CDR → fold → CDR loop graft → HDOCK 对接 ×N seed;
- scrambled CDR → fold → CDR loop graft → HDOCK ×N;
- Mann-Whitney p<0.05 且 native score 显著优于 scram → fold-then-dock 成立。

**Step 1 不通过则停**(同 V3 L0 闸门逻辑)。

### Step 2:方式4 候选批量对接(Step 1 GO 后)
- 候选池:V20(V19 亦可先跑)对 Aβ42 各表位 top-N 设计;
- 每设计:fold → CDR loop graft → HDOCK 对 5-seed 肽构象 → 取 p25 鲁棒分;
- 对照:from-zero(P3) / template-seeded(Pillar A) / 方式4 / native;
- 排序看方式4 是否优于 baseline、是否接近 native。

### Step 3:V20 重训(并行,方式4 thesis 真值)
**不依赖 HDOCK**,主线。head_seq 首次真训(V20 bug fix),重跑 P2 探针。
- Step 2 用 V19 候选先验 fold-then-dock 可行性;
- V20 出后用更优候选重跑 Step 2。

---

## 3. 风险与兜底

| 风险 | 兜底 |
|---|---|
| fold 出 apo loop 差异过大(19Å),不代表结合态 | Step 0 只 graft CDR loop 局部,框架不变;若仍过大,用 fold 的 CDR loop + 原 scaffold CDR 坐标加权混合 |
| AF2 单链 fold 对短 CDR loop 不稳(pLDDT 0.3) | 用多 seed fold 取共识;或 ESMFold 替代 |
| fold-then-dock 仍归零(fold loop 在 HDOCK 网格上仍被平滑) | Step 1 先验证,不扩规模 |
| 方式4 fold 出的 CDR 都过长/无序(pLDDT 低) | 反而是机制成立的旁证(高 disorder→可塑 loop),但对接需过滤 pLDDT<0.2 |
| 算力:每设计 fold ~200s + HDOCK ~min | 候选池先 top-20,非全量 |

---

## 4. 与前序方案关系

| 方案 | 状态 | 本方案关系 |
|---|---|---|
| V10(HDOCK 直对接 P1) | ❌ CDR 侧归零 | 本方案是 V10 P1 的"输入侧修复版" |
| V5 fold-then-dock | 提过未执行 | 本方案执行它,资产已就绪(HDOCK+L5) |
| V20 训练 | 待跑 | 并行主线,Step3 |
| 方式4 thesis | r=+0.31(残缺训练) | V20 重测真值,本方案不依赖但协同 |

---

## 5. 协作约定

| 约定 | 原因 |
|---|---|
| **Step 0 必须只 graft CDR loop 局部,框架不变** | 控 19Å apo 风险,保持可比 |
| **Step 1 native-vs-scram 闸门不通过禁做 Step 2** | 同 L0 逻辑,先验可行性 |
| **HDOCK 直对接(graft 仅侧链)永久弃用** | 0.000 RMSD 物理性归零,已证 |
| **fold 用单链(L5 Blackwell 可用),禁 multimer** | multimer 短肽失效 + B1 阻塞 |
| **候选池先小(top-20)** | fold+HDOCK 算力,先验可行性再扩 |

---

## 6. 产出物

| 步骤 | 产出 | 文件 |
|---|---|---|
| Step 0 | CDR-loop-graft receptor 构造器 | `run_v11_fold_graft.py` |
| Step 1 | native-vs-scram 闸门 | `idp_benchmark_results/fold_dock_gate_{ts}.json` |
| Step 2 | 方式4候选对接分 | `idp_design_results/folddock_design_scores_{ts}.json` |
| Step 3 | V20 checkpoint + P2 | `logs/bfn_v20_*/`, `way4_p2_v20_*.json` |

---

## 7. 时间

| 步骤 | 周期 |
|---|---|
| Step 0(graft 构造器) | 0.5 天 |
| Step 1(闸门,fold 8×N + HDOCK ×N) | 1 天 |
| Step 2(候选,fold 20× + HDOCK ×5seed) | 2-3 天 |
| Step 3(V20 训练,并行) | 1-1.5 周 |

**到 Step 1 闸门结论:1.5 天。** GO 则方式4 候选首次可纯计算排序。

---

## 附:一句话方案

> HDOCK CDR 侧归零是刚体对接对"仅侧链突变"的物理盲视(graft RMSD=0),改 HDOCK 无用,只能改输入:fold-then-dock——AF2 单链 fold 每个设计取出真实 CDR loop backbone(fold 后两两 RMSD 9-30Å,vs graft 0.000),局部 graft 回 scaffold(框架不变只 loop 变),再 HDOCK 对接。Step0 构造器→Step1 native-vs-scram 闸门(p<0.05 才进 Step2)→Step2 方式4候选批量对接(跨5-seed取p25)。V20 重训并行(方式4 thesis 真值,head_seq 首次真训)。Step1 闸门 1.5 天出结论,GO则方式4候选首次可纯计算排序,HDOCK 直对接永久弃。
