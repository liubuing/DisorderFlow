# 方式 4 下一阶段方案 V9 — 放大效应 + 固化可发表 + 推向 binding

**交付执行**: DeepSeek V4 | **日期**: 2026-07-02 | **基线**: V8 P2 实测 thesis 成立(unique_aa r +0.309, entropy r +0.198)
**前置铁律**: 不依赖 B1/Chai(本机全阻塞);binding 验证交湿实验或云 GPU(sm_90)。

---

## 0. 当前状态(V4 必读)

### 0.1 已成立(实测)
- **方式 4 训练期 thesis 双维度成立**(V8/V19 P2):
  - unique_aa vs disorder: +0.309(V7 仅 +0.091)✅
  - entropy vs disorder: +0.198(V7 −0.254,翻转转正)✅
  - 高 disorder unique_aa 6.0、TopAA% 0.323、无 6mer 退化
- V8 三处修复(diversity loss / max_prob anti-degen / 分位数阈值)全部对症生效
- 推理期方式 4:Spearman +0.60

### 0.2 诚实限定
- **机制成立≠binding 成立**。binding 墙(B1 全阻塞)纯计算不可达,交湿实验。
- **效应量中等**(r=+0.31,非强)。可放大。
- **CAID 红斑**:disorder head AUC=1.0 是单一 Aβ42(42 残基)vs 15327 折叠残基,不可外推,需换标准数据集复测。

### 0.3 V19 训练配置(影响放大策略)
- `freeze_backbone=true`, `unfreeze_encoder_layers=0`
- 可训仅:`disorder_proj` + `head_seq.0`(首层),4.6% 参数
- **这是效应量中等的可能原因**:head_seq 后续层 + encoder 冻结,disorder 信号虽进 head_seq 但下游表达力受限

### 0.4 可用资产
- `data/anti_abeta_refs/`:8 个已知抗 Aβ 抗体(4HIX/5CSZ/3UOT/6CGZ/6D9B/6H3R/7K4V)
- `data/confidence_idp_v3/`:DisProt 2673 IDP + 143 文献 IDP
- `data/idp_antigen_subset_v2/`:V7 三档 IDP 训练子集(gate PASS)

---

## 1. 总目标

把方式 4 从"thesis 实证成立(r=+0.31)"推进到:
1. **效应放大**:机制梯度 r→+0.5+,高 disorder unique_aa→7+;
2. **可发表资产固化**:双梯度 Figure + CAID 标准复测 + 方式4 方法论成文;
3. **binding 推进**:产出湿实验/云 GPU 候选,诚实标注验证路径。

---

## 2. 三条并行线

### 线 A:效应放大(计算,本机可做,1-1.5 周)

**目标**:把 unique_aa r 从 +0.31 推到 +0.5+。

**根因诊断**:V19 只训 `disorder_proj`+`head_seq.0` 首层(4.6%),disorder 信号进 head_seq 后下游 MLP 冻结,表达力受限 → 效应量中等。

**做法**(三步渐进,每步重跑 P2 验证):
1. **A1 解冻 head_seq 全段**:config `unfreeze` 扩到 head_seq 所有层(仍冻 encoder+backbone),让 disorder 信号有完整 MLP 表达。重训 4000 iters → P2。
2. **A2 提 diversity 权重**:若 A1 r 仍 <0.5,`diversity: 0.03→0.05`(V8 方案已留余地)。监控 seq loss 不涨过 94.0。
3. **A3 解冻 encoder 末层**:若 A2 仍不足,`unfreeze_encoder_layers=1`(GA encoder 最后一层),让 antigen→CDR 路由可调。风险高(可能扰动 V17i antigen routing),需严格 P2 对照。

**A 线判据**:
- GO: unique_aa r > +0.5 且 entropy r 维持 >+0.15 且无退化;
- STOP: 任一步导致 seq loss 涨过 94.5 或 entropy r 翻负 → 退回上一步,接受当前 r=+0.31 作论文结果。

**风险**:解冻越多越可能扰动 V17i 已有的 antigen routing(Δ+30pp)。**必须每步 P2 对照 + 保留 V19 作回滚 baseline。**

---

### 线 B:可发表资产固化(并行,1.5-2 周)

**目标**:把已成立的结果固化成可发表产出,不依赖 binding 成败。

**B1 双梯度 Figure**(方式4 核心):
- 用 V8 P2 数据(V7 vs V19 对照表)作论文核心 Figure:unique_aa/entropy/disorder 三轴梯度图;
- 补 V18(标量)作第三对照,展示"标量→逐残基→+diversity loss"的演进;
- 产出:`docs/paper_figures/way4_mechanism_gradient.pdf` + 数据 JSON。

**B2 CAID 标准复测**(修红斑):
- 现状:42 Aβ42 残基 vs 15327 折叠,AUC=1.0 不可外推;
- 换 **标准 DisProt/CAID9 benchmark**(多 IDP,平衡正负);
- 复测 disorder head AUC-ROC/PR vs IUPred3/AF2-pLDDT;
- 产出:`benchmark_disorder_caid_standard.py` + 标准基准结果。
- 若标准基准 AUC 仍 >0.85 → disorder head 作独立方法学贡献成立。

**B3 方式4 方法学成文**:
- thesis:对 IDP 表位的抗体不需锁死 CDR,需匹配表位可塑性的 CDR;
- 证据链:disorder head(识别)→ 推理期方式4(+0.60)→ 训练期方式4(+0.31 双梯度)→ anti-degen 内建;
- limitation:binding 验证待湿实验(纯计算评估墙已证极限,诚实)。

---

### 线 C:binding 推进(最后一公里,2-3 周)

**目标**:产出可绑定验证的候选 + 诚实验证路径。

**C1 候选生成**:
- 用 V19(或 A 线最优)对 Aβ42 各表位生成 top-N 设计;
- 对抗 Aβ 的 8 个已知抗体(4HIX/5CSZ 等)作 template-seeded baseline(Pillar A 已有 69 变体);
- 筛选:高 disorder 表位 + 高 unique_aa + 无退化 + 单链 fold pLDDT 合理(L5,Blackwell 可用)。

**C2 验证路径(二选一或并行)**:
- **湿实验**(首选,最可靠):SPR/BLI 测 top 候选对 Aβ42 的结合;诚实标注"计算设计+湿实验验证";
- **云 GPU**(论文 supplement):租 sm_90 卡跑 Chai-1/Boltz template-conditioned,绕开本机 Blackwell 阻塞,**仅作 supplement 不作主判据**(纯计算评估墙已证极限)。

**C3 诚实标注**:
- 论文明确:机制(P2)可独立验证 ✓;binding 验证交湿实验/云 GPU,非本机纯计算可达;
- 不声称"纯计算证明结合"。

---

## 3. 执行优先级与依赖

```
线A 效应放大(本机) ──┐
线B 资产固化(本机) ──┼── 并行,互不阻塞
线C binding推进 ─────┘
         │
         ├─ A 最优先(放大机制,计算可独立完成)
         ├─ B 并行(固化产出,不依赖 A/C 成败)
         └─ C 依赖 A 选出最优生成器 + 湿实验/云GPU资源
```

**优先级**:
1. **A1**(解冻 head_seq 全段)—— 最先,1 步即可能把 r 推过 0.5,且风险可控;
2. **B2**(CAID 标准复测)—— 并行,修红斑,保底资产;
3. **B1**(双梯度 Figure)—— A 出最优结果后做,数据现成;
4. **C1**(候选生成)—— A 选定最优生成器后;
5. **C2**(湿实验/云 GPU)—— 资源-dependent,周期长。

---

## 4. Go/Stop 判据

| 线 | GO | STOP |
|---|---|---|
| A | unique_aa r>+0.5 无退化 | seq loss 涨过 94.5 或 entropy 翻负 → 退回,接受 r=+0.31 |
| B | CAID 标准 AUC>0.85;Figure 成型 | CAID 标准 AUC<0.7 → disorder head 需重训,降级为辅助信号 |
| C | 湿实验有候选显著结合 | 湿实验无结合 → 机制 thesis 仍成立(binding 是另一问题),写 limitation 收尾 |

**关键**:A/B 的 STOP 都是"接受当前结果作论文产出",不是项目失败——方式4 thesis 已实证成立,这是底线。

---

## 5. DeepSeek V4 协作约定

| 约定 | 原因 |
|---|---|
| **A 线每步重跑 P2 对照,保留 V19 回滚 baseline** | 解冻越深越可能扰动 V17i antigen routing |
| **B2 必须换标准 DisProt/CAID9,不再用单一 Aβ42** | 红斑已确认,AUC=1.0 不可外推 |
| **C 线 binding 不在本机纯计算验证** | B1 三引擎全阻塞,本机已达极限 |
| **不重试已证伪路径**(AF2-multimer refold / L3L4B2 graft / B1 引擎) | 全有 commit 级证据 |
| **每步留 JSON+commit** | 决策链不断 |
| **方式4 thesis 已成立,所有 STOP 都是"接受作论文产出"** | 项目底线已守住 |

---

## 6. 产出物清单

| 线 | 产出 | 文件 |
|---|---|---|
| A | V20 checkpoint + P2 对照 | `logs/bfn_v20_*/`, `way4_p2_v20_*.json` |
| B1 | 双梯度 Figure | `docs/paper_figures/way4_mechanism_gradient.pdf` |
| B2 | CAID 标准复测 | `benchmark_disorder_caid_standard.py`, `caid_standard_*.json` |
| B3 | 方法学初稿 | `docs/way4_method_draft.md` |
| C1 | binding 候选 | `idp_design_results/way4_binding_candidates_*.json` |
| C2 | 验证报告(湿/云) | 待资源 |

---

## 7. 时间预估

| 线 | 周期 |
|---|---|
| A(3 步渐进) | 1-1.5 周 |
| B1+B2+B3 | 1.5-2 周(并行 A) |
| C1 | 3-5 天(依赖 A) |
| C2 | 2-3 周(资源-dependent) |

**到 A+B 闭环(可发表资产就绪):约 2-2.5 周。**

---

## 附:一句话方案

> 方式4 thesis 已实测成立(r=+0.31 双梯度),下一阶段三线并行:A 放大效应(解冻 head_seq 全段,目标 r+0.5+,每步 P2 对照保 V19 回滚);B 固化可发表资产(双梯度 Figure + CAID 换标准 DisProt 复测修红斑 + 方法学成文,不依赖 binding);C 推向 binding(用最优生成器产候选,交湿实验或云 GPU sm_90,诚实标注本机纯计算已达极限)。优先 A1+B2 并行。所有 STOP 都是"接受当前 r=+0.31 作论文产出"——底线已守。约 2-2.5 周到可发表资产就绪。
