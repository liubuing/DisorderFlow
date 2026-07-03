# 方式4 可发表资产报告 — V13.1 §4

**日期**: 2026-07-04 | **状态**: 待§2消融结果填充
**基线**: 三漏洞修正后, [VERIFIED]资产仅剩 V17i(待复测)、V15dg03置信头、评估墙limitation

---

## 1. 资产验证状态 (2026-07-04 最终)

| 资产 | 标签 | 说明 |
|---|---|---|
| V17i 通用抗原路由 Δ+30pp | `[PENDING]` | V20真训对antigen routing影响**待测** |
| 方式4训练期 V19 r=+0.31 | `[FALSIFIED]` | head_seq冻结artifact, V20证实为假 |
| 方式4训练期 V20 r=−0.45 reframe | `[PENDING]` | **待§2消融判定** |
| 推理期 disorder_guided | `[FALSIFIED]` | 从未独立验证;旧+0.60是置信头误标 |
| V15dg03 置信头 OC +0.595 | `[VERIFIED]` | 真实资产, 置信度校准贡献 |
| Disorder head CAID 标准0.47 | `[VERIFIED]` | 无效(结论站得住=无效) |
| 评估墙 CDR 侧3路证伪 | `[VERIFIED]` | 物理极限真结论 |
| V20 多样性突破 (3× unique_aa, 0 degen) | `[PENDING]` | V20训练期附产物, 非独立thesis |
| Wet-lab 候选 | `[PENDING]` | 基于动摇叙事, 需重审 |

---

## 2. 可发表叙事 (按§2消融结果分叉)

### 2A: reframe 站住 (A1 r 维持负)

**核心 triplet**:
1. **V17i 通用抗原路由** (主, 待复测确认不退步)
2. **V15dg03 置信头 OC +0.595** (置信度校准, 独立贡献)
3. **方式4训练期 disorder-constrains-design** (辅, reframe消融验证)

**负/中性**:
4. 评估墙 CDR 侧物理极限 (真结论: 刚体对接对CDR构象盲视)
5. Disorder head 标准基准无效 (诚实 limitation)

**叙事**: "BFN抗体设计系统在V17i实现通用抗原路由, V15dg03提供置信度排序, WAY4揭示disorder约束CDR设计的机制——三组件构成完整设计pipeline的可行基础。"

### 2B: reframe 站不住 (A1 r 回正, loss 副产物)

**核心 triplet**:
1. **V17i 通用抗原路由** (主, 仅剩正面)
2. **V15dg03 置信头 OC +0.595** (置信度校准)
3. 方式4训练期诚实作"混合/负面结果" (不强行正面叙事)

**负/中性**:
4. 评估墙 CDR 侧物理极限
5. Disorder head 标准基准无效
6. 方式4 diversity+anti_degen loss 的 r=−0.45 确认为 loss 副产物

---

## 3. 方法学论文 outline

### Section 1: Introduction
- Antibody CDR design problem
- Disorder-conditioned paradigm (动机, 即使结果mixed)

### Section 2: Architecture
- BFN backbone (V17i antigen routing)
- Disorder conditioning (disorder head + conditioning modes)
- Confidence head (V15dg03 OC calibration)

### Section 3: Results
- V17i: Universal antigen routing (Δ+30pp, AF2 ipTM +49%)
- V15dg03: Confidence calibration (OC Spearman +0.595)
- WAY4 training: Head_seq unfreezing + diversity breakthrough (3× unique_aa, 0 degen)
- WAY4 mechanism: [待§2消融结果填充 — reframe 站住 vs 站不住]

### Section 4: Discussion
- Limitation 1: 评估墙 CDR 侧 — 刚体对接盲视, binding 终交湿实验
- Limitation 2: Disorder head 标准基准无效 (结构输入固有限制)
- Limitation 3: [待§2填充 — WAY4 训练期信号的性质]

### Section 5: Methods
- Training setup, data, evaluation protocols
- 资产三件套验证铁律

---

## 4. 待执行项 (按优先级)

| 优先级 | 项目 | 状态 |
|---|---|---|
| P0 | §2 reframe消融 (A1/A2/A3 训练+P2) | 进行中 |
| P0 | V17i V20复测 (确认V20对antigen routing不退步) | 待算力 |
| P1 | Wet-lab候选重审 (基于V17i+方式4最终叙事) | 待叙事确定 |
| P2 | 方法学论文撰写 | 待叙事确定 |

---

## 5. 铁律复检

- [x] 所有资产带 [VERIFIED]/[PENDING]/[FALSIFIED] 标签
- [x] 禁把 [PENDING] 当"已成立"
- [x] CAID 红斑 1.0 不进论文
- [x] 推理期 +0.60 不进论文 (是置信头误标)
- [x] 评估墙 CDR 侧不追加投入
- [ ] §2 消融完成前 reframe 不进结论
- [ ] V17i V20 复测确认前不声称"通用"
