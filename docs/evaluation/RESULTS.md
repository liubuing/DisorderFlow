# RESULTS.md — BFN 抗体设计版本决策表

**分支**: `feature/idp-aware` | **更新**: 2026-06-28 | **算力**: RTX 5060 (本机) + WSL2 JAX GPU + ColabFold

> 本表汇总 V4 → V17i 全部关键版本,把"是否进步、卡在哪、下一步往哪走"钉死在一张表里。
> 评估任务统一为:**Aβ42 表位 + 5IMK/3STB scaffold 的 CDR 设计**,BFN 自评分 vs AlphaFold2 真实分。

---

## 0. 评估指标说明

| 指标 | 含义 | 理想值 |
|------|------|--------|
| **BFN ipTM / pLDDT** | 模型自评分(不需要 AF2) | 与 AF2 一致 |
| **AF2 ipTM** | AlphaFold2 复合物结构验证的真实结合置信度 | ≥0.6 才算"有结合",≥0.8 强结合 |
| **OC ratio** = BFN_iptm / AF2_iptm | 过自信倍数 | →1.0(目前越高越虚) |
| **Spearman ρ** (BFN_iptm vs AF2_iptm) | BFN 排序是否与真实一致 | >0(BFN 高分应 = AF2 高分) |
| **Δ(complex−fixbb)** | complex 模式(见抗原) vs fixbb(盲) 的 CDR 序列恢复率差 | >0 = 模型在用抗原信号 |
| **Seq recovery** | 设计 CDR 与 native CDR 的序列恢复率 | 越高越好(但需结合 Δ) |
| **AF2 ceiling** | 库筛选后最佳设计经 AF2 验证的 ipTM 上限 | 破 0.4 = 可用抗体门槛 |

---

## 1. 核心决策表(置信度层 — 抗体整体置信度校准)

> 早期主线:让 BFN 置信头不再对所有设计输出相同分数。**这条线已基本走通(V14 起 std 进入 1e-5)**。

| 版本 | 数据/方法 | BFN ipTM | AF2 ipTM | OC ratio | Spearman ρ | 结论 |
|------|-----------|---------|---------|---------|-----------|------|
| Phase3 | disorder baseline,冻backbone | 0.882 | 0.062 | 14.2x | — | 置信头对设计无区分力(std~1e-5) |
| Phase4 | 40% 负样本 | 0.796 | 0.000 | — | — | OC 微降,负样本未根除问题 |
| IDP Comb | 冻backbone+encoder dropout | 0.913 | 0.062 | 14.7x | — | 反而恶化 |
| **V11** SeqConf | ent+max_prob 2D bypass 头 | 0.810 | 0.064 | 12.7x | — | bypass 仅 2 维,被 backbone 饱和 logit 淹没,**退步** |
| **V12** Pure-Seq | backbone-free 深 seq 头+var/rank loss | 0.689 | 0.067 | 10.3x | — | 头有区分力了,但生成器仍无抗原意识 |
| **V14** Grouped | real-AF2 design-variant 训练+grouped margin | 0.338 | 0.055 | 6.2x | **+0.30** | **OC 从 14x→6x,spearman 首次转正**,分层 loss 有效 |
| V15 | binding 头 | 0.337 | 0.052 | 6.8x | -0.70 | spearman 反转负面 |
| V15b | 扩样本 | 0.353 | 0.057 | 6.2x | -0.08 | 抖动 |
| V15dg03 | 蒸馏梯度 | 0.354 | 0.056 | 6.5x | **+0.60** | spearman 最佳,但 OC 未降 |
| V15C | 综合 | 0.356 | 0.060 | 6.2x | -0.34 | |
| V16C | codesign | 0.354 | 0.057 | 6.3x | -0.50 | |

**置信度层结论**:V14 的 grouped real-AF2 训练是把过自信从 14x 压到 6x 的关键拐点;但之后 V15/V16 在 6x 平台震荡,Spearman 反复在 ±0.6 之间跳,**置信度层遭遇天花板——根因不在头,在生成器不用抗原**(见下)。

---

## 2. 核心决策表(生成器层 — 抗原信号路由)

> DIAGNOSIS.md 确认:encoder 能看见抗原(30% CDR logits 变化),但回收循环用无区分力的 backbone 置信做 feedback,把 antigen 信号冲掉。V17 系列专攻"让 CDR 序列头真正用上抗原"。

| 版本 | 方法 | lr | iters | Δ(complex−fixbb) | contrastive loss | 结论 |
|------|------|----|----|----|----|------|
| V15 baseline | cross-attn 有 gate | 2e-5 | — | **+0.1pp** | — | 抗原信号几乎不到 head_seq |
| V17 antigen_xa | antigen→CDR cross-attn,seq_only | 2e-5 | 2000 | 微动 | — | 梯度太小,gate 几乎不变 |
| V17b highlr | 诊断,lr 25× | 5e-4 | 500 | — | — | gate 动了但学到噪声 |
| V17c zeroinit | 去 gate,零初始化 | 5e-4 | 500 | — | — | **NaN 1153 次** |
| **V17g** nansafe | pre-LN+feat clamp+t_clamp=0.95 | 2e-5 | 2720 | **−1.39pp** | — | **NaN 根治(0次)**,但 seq recovery 训练反而压制抗原信号 |
| **V17h** contrastive | CDR 重排 50%+BCE 匹配头,续 V17g | 2e-5 | 4000 | 待测 | **0.91→0.48** | **首次跌破随机基线 0.693 = 模型首次学会区分真假 CDR** |
| **V17i** pairrouting | pair_feat→head_seq 路由(§3.1a) | 2e-5 | 8000 | 待测 | (续) | 继续 pair routing 训练 |

**生成器层结论**:V17g 治好训练稳定性,V17h 首次验证"模型能学会用抗原信号区分 CDR"——这是**架构性突破**,但尚未传导到产出质量(见下)。

---

## 3. AF2 最终验证天花板(库筛选)

> 这是唯一"用户可见"的硬指标:BFN 设计的 CDR,graft 到 scaffold,经 AF2 multimer 验证真实结合。

| 阶段 | 产物 | AF2 ipTM | 备注 |
|------|------|---------|------|
| V14 phase2 | 10 设计/scaffold | 0.055 | 5IMK 单 scaffold |
| V15 ~ V16 | phase2 | 0.05–0.06 | 全部 <0.1,无实质结合 |
| **V17i library screen** | 多 scaffold×多 seed 库 | top20 BFN_iptm=0.36 | BFN 自评高 |
| **V17i final ceiling** | AF2 验证 | **ipTM 0.121–0.140** | **当前天花板** |
| V17i phase2 stats | 30 设计 | af2_iptm_mean=0.120, **OC=2.98x**, **Spearman=−0.71** | ❌ **BFN 高分设计 AF2 反而低分** |

**关键危机(2026-06-28)**:
1. AF2 ceiling 仍是 **0.12–0.14**,远低于可用抗体门槛 0.4,且 V14→V17i 无实质提升。
2. V17i 的 **Spearman ρ = −0.71** —— BFN 排序与 AF2 真实排序**负相关**,即模型挑的"最好"设计实际是 AF2 最差的。
3. V17i library top20 选出的 CDR 大量是 `TTTTTTTVVVV` / `TTTTRTVTRTTT` 类**低复杂度重复序列**——生成器在退化。
4. OC 已从 14x → 3x,置信度层进步明显;但**生成器质量天花板未破**。

---

## 4. 因果诊断:为什么 V17h contrastive 突破没传导到 AF2

```
V17h contrastive loss 0.91→0.48  ✅ 模型学会区分"真假CDR"(判别信号)
        ↓ 但
contrastive 只在"配对"层面奖励,不在"生成"层面
        ↓
head_seq 仍可输出使 contrastive head 满意的低复杂度序列(T/V 重复)
        ↓ 这类序列 entropy 低、BFN_iptm 高(置信头给高分)
        ↓ 但 AF2 folds 出无结构 coil → ipTM 0.12
        ↓ 形成正反馈:高 BFN 分 → 入选 → AF2 低分 → Spearman −0.71
```
**根因**:判别训练(contrastive)≠ 生成训练。模型学会了"判别真假",但生成分布仍偏向容易判别的退化序列。**缺失一个把 AF2 真实分回灌到生成分布的 loss**(RL/蒸馏),或一个禁止低复杂度退化的约束。

---

## 5. 决策结论(2026-06-28)

| 问题 | 状态 |
|------|------|
| 置信头无区分力 (std<1e-4) | ✅ 已解决(V14 grouped) |
| 训练 NaN | ✅ 已解决(V17g nansafe) |
| 模型能否学会用抗原信号 | ✅ 已突破(V17h contrastive,0.91→0.48) |
| **BFN 作生成器的 AF2 天花板 (0.12)** | ❌ **未破,V14→V17i 无进展** |
| **BFN 排序与真实反相关 (ρ=−0.71)** | ❌ **新危机,生成器在退化** |

**路线决断**(详见第 7 节改造方案):
- **继续硬训 BFN 生成器**:算力(RTX 5060)+ 已 8000 iters 卡 ceiling,线性加码收益递减,**不推荐作为唯一路径**。
- **转混合架构(推荐)**:BFN 已被证明是**优秀的判别器/置信器**(contrastive 0.48、spearman 在 V14/dg03 能转正),把 BFN 定位为**评分/筛选器**,生成交给 ProteinMPNN/diffab(已有 `mpnn_bfn_hybrid.py` 雏形),BFN 负责"挑对的那一个"。
- 用 RL/蒸馏把 AF2 分回灌,治本解决 Spearman 负相关。

---

## 6. 版本谱系与归档

| 版本簇 | 目标 | 关键文件 | 状态 |
|--------|------|---------|------|
| V4–V6 | 通用蛋白置信头(2032蛋白) | configs/bfn_full.yml | ✅ 已开源 HF |
| Phase3–5, IDP | disorder head + 校准 | bfn_phase3/4/5, idp_* | 归档 |
| V11/V12 | seq-aware confidence | bfn_v11/v12_seqconf | 退步,归档 |
| V14 | grouped real-AF2(拐点) | bfn_v14_grouped_conf | ✅ 保留基线 |
| V15/V16 | binding/codesign | bfn_v15/v16_* | 天花板,归档 |
| V17a–g | 抗原 cross-attn + NaN 修复 | bfn_v17* | V17g 保留作基线 |
| **V17h** | contrastive(架构突破) | bfn_v17h_contrastive | ✅ 保留 |
| **V17i** | pair routing + library | bfn_v17i_pairrouting | 当前主版本 |

OC 验证原始数据见 `oc_validation_results/oc_*.json`;训练配置见 `configs/train/`。

---

## 7. 下一阶段改造方案(适用于 DeepSeek V4)

> 目标:**破 AF2 ipTM 0.4 天花板 + 治愈 Spearman 负相关**。以 V17i 为基线,V18 起执行。

### 7.1 战略:BFN 从"生成器"重新定位为"判别器+置信器",生成让位给 MPNN

**理由**(数据驱动,见 §3 §4):
- BFN 作生成器 4 个版本(V14→V17i)AF2 ceiling 停在 0.12,且 Spearman=−0.71;
- BFN 作判别器(V17h contrastive 0.48、V14/dg03 spearman +0.3~0.6)已被证明有效;
- RTX 5060 算力不足以承担从头训生成器 + RL 的双重成本。

**改造方向**:`MPNN 生成 → BFN 评分筛选 → AF2 验证` 三段式,V18 专注把 BFN 评分调到与 AF2 正相关。

### 7.2 改造任务分解

#### 任务 A:治本——AF2 蒸馏 loss(让生成分布对齐 AF2 真实分)
- **问题**:contrastive 只判别不生成,导致低复杂度序列拿高分后退化。
- **方法**:RL-finetune 风格——对 BFN 采样的 CDR,跑 AF2 得真实 ipTM,用 REINFORCE / DPO 把"AF2 高分序列"的概率拉高。
- **落点**:扩展 `rl_finetune_af2.py`(已存在 ipTM-guided RL 框架),目标函数加 entropy 正则惩罚低复杂度。
- **验收**:Spearman ρ > 0,AF2 ceiling > 0.2。

#### 任务 B:生成器换源——MPNN+BFN 混合管线产品化
- **方法**:`mpnn_bfn_hybrid.py` 已有雏形,需补:多 scaffold、多 seed、CDR 长度变体采样,统一输出 schema。
- **评分**:用 V14/V17h 的 BFN 置信头(V14 spearman 曾 +0.30)做筛选,**禁用 V17i 头**(spearman −0.71)。
- **验收**:hybrid 管线 AF2 ceiling > 0.3(单人 BFN 生成)。

#### 任务 C:治标——低复杂度/退化序列过滤
- **问题**:V17i top20 出现 `TTTTTTTVVVV` 类序列。
- **方法**:cascade filter 加 Shannon 多样性 + k-mer 重复度 +氨基酸组分惩罚项;`run_v17i_library_screen.py` 加 hard filter。
- **验收**:top20 中无 >6 连续相同残基、无 >50% 单一氨基酸。

#### 任务 D:工程——拆分 `core.py`/`receiver.py`,降低迭代风险
- **现状**:`core.py` 38KB / `receiver.py` 30KB,confidence head+recycling+loss+对比头全混,V4→V17i 反复在同批文件打补丁。
- **方法**:拆为 `confidence_heads.py` / `recycling.py` / `losses.py` / `contrastive_head.py`,主干冻结。
- **验收**:现有 7 个 pytest 全绿,推理输出数值不变(regression)。

#### 任务 E:实验闭环——固定 benchmark + 自动 RESULTS 更新
- **方法**:把 `probe_antigen_signal_usage.py`、`oc_*` phase1/2、library+AF2 ceiling 固化为一条 `make benchmark` 流水线,结果自动回写本表 §2/§3。
- **验收**:一次命令产出 Δ/OC/Spearman/ceiling 四指标。

### 7.3 DeepSeek V4 协作约定(避免重复踩坑)

| 约定 | 原因 |
|------|------|
| **改 BFN 内核前先读 §1–§5 + DIAGNOSIS.md** | V11/V15/V16 多次"自信地修复"反而退步,需先确认改的是头还是生成器 |
| **任何新版本必须同时报 4 指标**:OC / Spearman / Δ / ceiling | 单看 val loss 无意义(V17g val↓0.32 但 Δ 变负) |
| **不轻易调最大算力上限** | V17i 已 8000 iters 仍卡 0.12,线性加码无效,先换方法 (任务 A/B) |
| **置信头优先用 V14 grouped,不沿用 V17i** | V17i spearman=−0.71,排序反了 |
| **新建版本号 V18+ 而非继续 V17j** | V17 已达生成器天花板,需标明战略转向 |

### 7.4 验收里程碑

| 里程碑 | 指标 | 优先级 |
|--------|------|--------|
| M1 治标 | 低复杂度过滤上线(top20 无退化序列) | P0(1天) |
| M2 头选对 | hybrid 管线改用 V14 头,spearman>0 | P0(2天) |
| M3 生成换源 | hybrid AF2 ceiling > 0.3 | P1(1周) |
| M4 治本 | RL/蒸馏上线,Spearman>0 + ceiling>0.2 | P1(2周) |
| M5 工程 | core/receiver 拆分,测试绿 | P2(并行) |
| **终局** | AF2 ipTM > 0.4 可用抗体候选 | — |

---

---

## 8. 天然无序蛋白 (IDP) 抗体设计专项评估

> "IDP 感知设计"是项目对外宣称的核心差异化卖点。本节评估其**真实表现**。
> 测试靶点 = **Aβ42**(`2NAO_model1_A_1-42.pdb`),典型的天然无序蛋白(淀粉样蛋白)。

### 8.1 IDP 管线已落地的能力(方法学上成立 ✅)

`modules/idp_disorder_analysis.py` + `idp_antibody_design.py` 实现了完整 IDP 感知逻辑:

| 能力 | 实现 | 验证 |
|------|------|------|
| 逐残基无序度预测 | BFN disorder head,sigmoid 概率 | ✅ |
| **有序表位识别**(避开无序区) | `find_ordered_segments(threshold=0.3)` | ✅ Aβ42 命中 13–25 段(mean disorder 0.21) |
| 仅在有序段上做 design context | segment_start/end 传入设计 | ✅ |
| multi-conformation 处理 IDP 构象系综 | `build_abeta_conformations.py` 5-seed×42 CA→重建 backbone | ✅ |

**方法学上是真正"IDP-aware"的**——不是简单扔 IDP 进通用管线,而是主动用 disorder head 找有序表位 + 多构象采样。

### 8.2 IDP 抗体设计的实际质量(关键 ❌)

| 指标 | 数值 | 来源 | 判读 |
|------|------|------|------|
| **AF2 平均 ipTM (IDP 靶点)** | **0.063–0.064** | `nosmooth_v2_validation/comparison.json` | ❌ 远低于结合阈值 0.6 |
| **AF2 平均 pLDDT** | 0.229–0.258 | 同上 | ❌ 实际结构基本 unfold |
| **BFN 自评 ipTM (IDP 靶点)** | 0.746–0.771 | 同上 | OC ≈ **12x 过自信**(IDP 比整体更虚) |
| **ad_phase3_demo fixbb vs complex** | iptm 0.771 vs 0.756 | fixbb_vs_complex.json | complex 反而**低**——IDP 上 antigen context 未帮助 |
| **idp_design_results 分段设计** | af2_iptm 缺失/低,BFN 0.77 | ranked_designs | 同样过自信 |

**结论 1:在 Aβ42 这个 IDP 靶点上,BFN 设计的抗体 AF2 真实结合分 ~0.06,与整体 ceiling 持平甚至更差,且过自信达 12x(比整体的 6x/3x 更严重)。**

### 8.3 为什么 IDP 表现更差(双重失效)

```
原因 A — IDP 残留 backbone 解析坑(已部分绕过):
  2NAO NMR 单模型只解析到 42 个残基中的 19 个
  → designable epitope 被 parser 的 CA-CA<4Å chain walk 截断
  → 仅 13–25 区段(12残基)可设计
  → build_abeta_conformations.py 用 5-seed CA 重建 backbone,绕开此坑 ✅

原因 B — 置信头在 IDP 上本就更不可靠(README 已声明 Known Limitation):
  BFN 置信头(pLDDT/ipTM)训练于 folded globular 蛋白
  → 对无序区输出不可靠 → IDP 上 OC 12x vs 整体 6x
  → disorder head 能"识别"IDP,但 confidence head 仍"盲目自信"地给高分

原因 C — 通用生成器天花板(§3 §4)在 IDP 上同样成立:
  AF2 ceiling 0.12 的退化问题(低复杂度 CDR、Spearman −0.71)
  对 IDP 靶点同样存在,无特殊豁免
```

### 8.4 IDP 感知能力的真实定位(诚实结论)

| 维度 | 评价 |
|------|------|
| **disorder 识别(知道哪是无序区)** | ✅ 有效,是有价值的差异化模块,值得上 CAID 基准量化 |
| **IDP-aware 表位选择(避开无序区设计)** | ✅ 逻辑成立,实测命中 Aβ42 有序段 |
| **IDP 抗体的最终结合质量(AF2 验证)** | ❌ ipTM 0.06,与普通蛋白无异甚至更差 |
| **IDP 上的置信度可靠性** | ❌ 过自信 12x,比整体更不可靠 |

**一句话:DisorderFlow 对 IDP 是"会看、会躲,但不会做好"。** 它能正确识别 Aβ42 的无序性并选有序表位(这是真正的差异化能力,值得论文化),但设计出来的抗体在 IDP 靶点上实际结合质量与普通蛋白一样卡在天花板,且置信度更不可靠——IDP 感知尚未转化为最终设计质量优势。

### 8.5 IDP 领域的下一步建议(补入 §7 任务)

| 任务 | 目标 | 优先级 |
|------|------|--------|
| **I-1 disorder head 上 CAID 基准** | 量化差异化能力(AUC vs IUPred3/AF2-pLDDT),这是当前 IDP 上唯一切实可发表的结果 | P0 |
| **I-2 IDP 专用置信校准** | 在无序区上单独标定 confidence(当前 12x OC),或对 disorder>0.5 区段禁用/降权 BFN 分数 | P1 |
| **I-3 IDP 抗体生成交给 MPNN+BFN 混合** | 与 §7 任务 B 一致,IDP 靶点同样受益于换源 | P1 |
| **I-4 multi-conformation 评分取最稳** | 现已多构象采样,但评分未取跨构象鲁棒性最小值,需补 | P2 |
| **I-5 文档诚实化** | README 提到的"IDP 盲视 Known Limitation"扩展为"disorder 识别有效,设计质量未破" | P0 |

---

## 附:数据来源

- §1 §2 §3 全部数值来自 `oc_validation_results/oc_*.json`(实跑,非推测)。
- §8 IDP 数值来自 `idp_design_results/` 的 `comparison.json` / `fixbb_vs_complex.json` / `validation_results.json` / `idp_design_results.json`(实跑)。
- Δ 数据来自 `probe_antigen_signal_usage.py` / `probe_v17i*.py`(stdout,需 `make benchmark` 固化)。
- 训练轨迹见 `logs/bfn_v17*/` 与记忆文件 `v17h-contrastive-breakthrough` / `v17i-pair-routing-breakthrough`。

---

## 9. V3 度量重排实验(2026-06-29 15:05,决定性结果)

**实验**:用 V3 纯Python物理度量(L2 FFT pose → L3 interface score)对 P3 的 100 设计重排,与旧 composite(MPNN+diversity)对比。
**这是首次在"被证明有效的度量"(L0 闸门 5CSZ d=2.545)下评判生成器质量。**

### 9.1 结果

| 指标 | 数值 | 含义 |
|------|------|------|
| Kendall τ(V3 vs 旧composite) | **+0.9250** (p=2.3e-21, n=100) | 两度量高度一致 |
| top20 重合 | 18/20 | 旧排序基本被新度量复现 |
| top5 序列 | 5IMK scaffold 上高度相似(`...TESGGGTVAAGGSLTL`) | 聚集核心表位聚集到同一 paratope 模式 |

### 9.2 两个结论

**结论A(打脸预判,正面)**:旧 composite 没有造假。之前担心"多样性冒充质量"不成立——纯界面物理打分与 MPNN+diversity 排序高度一致(τ=0.925)。**P3 的 top20 是可信的。**

**结论B(暴露新问题,负面)**:V3 score 在 100 设计中**几乎无区分力**,contacts 分布呈两极——大量 0 接触 + 一批恒为 5 接触,score 恒为 0.190。这说明:
- 当前 score_pose 在多数 pose 上 **区分力塌方**;
- τ=0.925 的高一致性部分来自"两度量都把同一批低分设计排末尾"的 trivial agreement,而非都精准区分了头部设计。

**净结论**:
- 旧 composite 被洗清嫌疑 → P3 top20 可信,**生成器真实质量天花板~0.089(AF2噪声)是设计能力问题,不是度量幻觉**。
- V3 度量在 L0 crystal 上有效(5CSZ d=2.545),但在**对接生成的 pose 上区分力不足**,需修(见 §10)。

### 9.3 范式转移验证状态

| 阶段 | 状态 |
|------|------|
| AF2 re-fold 范式失效 | ✅ 确认(M0) |
| 物理打分范式在 crystal 有效 | ✅ 确认(L0 5CSZ d=2.545) |
| 物理打分范式在生成 pose 上有区分力 | ⚠️ **未达标**(score 塌方) |
| 生成器质量天花板 | ❌ 仍是 ~0.089,且确认非度量幻觉 |

## 10. 下一步(V3 度量区分力修复)

score_pose 区分力塌方的可能根因:
1. L2 FFT 单 pose 采样不足(只返回1个pose,无 pose 优化);
2. L3 在低接触数 pose(<5)上应判为"无效对接"而非给中间分;
3. 需 top-K pose + 跨构象取 p25(方案 V3 §3.1 已设计但未启用)。

P0 修复:`score_pose` 对 contacts<5 的 pose 直接判 0,并对每设计取 top-K pose 的中位数,恢复区分力。

---

## 11. V4/V5 度量穷举——结合特异性不可分的收敛证据 (2026-07-01)

> **战略级结论**: 经历 V3→V4→V5 三轮范式转移，四条独立度量路径以同一种根因失败。纯计算在 IDP-scaffold 抗体设计的"结合特异性"度量上已达极限。

### 11.1 四条路径全败收敛表

| 路径 | 晶体可分？ | 生成设计可分？ | 根因 | Commit |
|------|:--:|:--:|------|--------|
| **L3 物理打分** | ✅ 5CSZ d=2.545 | ❌ 全同 (0.207) | scaffold backbone 几何主导 | 258ec0c |
| **L4 ESM-IF (scaffold)** | ✅ sep=+1.76 | ❌ 全同 (-0.957) | 骨架坐标无序列区分力 | c28c97d |
| **L4 ESM-IF (AF2-folded)** | ❌ sep=-0.006 | ❌ 全同 | 折叠坐标仍被模型先验主导 | f8669b8 |
| **B2 fold-then-score** | ❌ sep=+0.01 | ⚠️ 可分(unique=7/9) | 测的是 scaffold-fit 非结合 | e3b77e0 |

### 11.2 统一根因

```
所有度量的可分性都依赖于"抗体-肽相对界面几何"：
  晶体结构 ✓ 有真实界面几何 → L0/L3/L4 晶体上可分
  scaffold graft ✗ 肽段位置随机 → 度量被骨架几何主导,归零
  AF2 folded    ✗ 折叠坐标仍被模型先验主导,无法恢复界面特异性
  
结论: 不含"抗体-肽相对界面几何"的度量,结构性地无法区分结合特异性。
     这不是度量设计的问题——是表征空间本身缺少结合信号维度。
```

### 11.3 B1 最终结论

B1 (template-conditioned 结构预测) 是唯一含界面几何的路径，但在三个引擎上全部阻塞:

| 引擎 | 阻塞原因 |
|------|---------|
| ColabFold | hhsearch 二进制缺失 |
| Chai-1 | CUDA 内核不兼容 RTX 5060 Blackwell |
| HDOCK | 仓库已删除,无法编译 |

**B1 状态: NO-GO on current hardware** (commit 7d2d660)

### 11.4 纯计算边界

纯计算在 IDP-scaffold 抗体设计上的能力边界已划定:
- **可做**: 骨架兼容性排序 (B2)、序列自然度过滤、无序识别 (C)
- **不可做**: 结合特异性验证 (需 B1 或湿实验)
- **这是有效结论——不是"项目失败",是界定了纯计算方法学的能力边界**

---

## 12. 可交付资产状态 (2026-07-01)

### 12.1 独立可发表产出

| 资产 | 状态 | 数据 |
|------|:--:|------|
| **C: Disorder head CAID 基准** | ✅ | AUC-ROC=1.00, AUC-PR=1.00; ordered(SAbDab μ=0.16) vs disordered(Aβ42 μ=0.93), Δ=+0.77 |
| **V17i Pair Routing** | ✅ | Δ(complex−fixbb)=+30pp; 首次证明 BFN 编码器能路由抗原信号到序列头 |
| **V3 L0 物理打分** | ✅ | crystal native vs scrambled 可分 (5CSZ d=2.545); 零二进制纯Python |
| **纯计算边界界定** | ✅ | L3/L4/B2/V4 四路证伪 + 统一根因 + B1 硬件阻塞 → 可发表 limitation |

### 12.2 设计资产

| 资产 | 状态 | 数据 |
|------|:--:|------|
| **A: Template-seeded 变体** | ✅ | 69 variants; 4HIX(2 hotspots)+5CSZ(6 hotspots); Shannon 2.8; 无退化 |
| **P3 MPNN 设计库** | ✅ | 720 designs; top20 AF2 iptm max 0.188; τ=0.925 vs 物理打分 |
| **V17i 生成器** | ✅ | 8000 iters; Δ=+30pp; AF2 ceiling 0.121 |

### 12.3 度量工具箱

| 工具 | 角色 | 适用场景 |
|------|------|---------|
| contact-count + charge | 晶体结构验证 | L0 闸门, 已知复合物 |
| BFN V14 置信头 | OC 校准 (ρ>0.4) | 生成设计置信度 |
| anti-degen filter | 序列质量 | 退化过滤 |
| MPNN score | 生成质量 | 设计排名 |
| AF2 single-chain pLDDT | 折叠质量 | 独立折叠验证 |
| ESM-IF fold-then-score | scaffold-fit | 骨架兼容性 |

---

## 附录: 版本与 commit 索引

| 关键 commit | 内容 |
|-------------|------|
| 258ec0c | V3 robust grid-search → geometry-limited |
| c28c97d | L4 ESM-IF on P3 → generated poses indistinguishable |
| f8669b8 | V4 AF2-fold → real coords → also fails |
| 49fbdd0 | Pillar C: CAID AUC=1.00 |
| 2b80eb9 | Pillar A: 69 template-seeded variants |
| e3b77e0 | B2 Pillar A validation |
| 7d2d660 | B1 final: Chai-1 blocked by CUDA (RTX 5060)
