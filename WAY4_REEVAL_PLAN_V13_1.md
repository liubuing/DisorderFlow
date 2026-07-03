# 方式4 重估与收尾方案 V13.1 — 三漏洞修正后

**交付执行**: DeepSeek V4 | **日期**: 2026-07-04 | **基线**: V20翻转(r=−0.449) + CAID标准无效(AUC=0.47) + 推理期+0.60误标(实为置信头OC Spearman)
**前置铁律**: 评估墙CDR侧已终局死亡(3路证伪), binding交湿实验, 本机不追结构验证(RTX 5060 OOM)。
**数据真实性底线**: 全部如实记录+标注验证状态, 不造假(三漏洞都是表述错位, 非数据造假)。

---

## 0. 三漏洞修正(V4 必读)

### 0.1 漏洞1: V19 +0.31 "thesis成立"(FALSIFIED)
- V20真训(head_seq bug修复)后unique_aa vs disorder r=**−0.449**(p=0.002)
- V19的+0.31是head_seq冻结artifact, 原thesis(高disorder→可塑多样CDR)不成立
- reframe: disorder-constrains-design(r=−0.45, 信号方向反转, 模型仍利用disorder)
- **reframe待消融**: r=−0.45是"生物学约束"还是"diversity×anti_degen loss拉锯副产物"未定

### 0.2 漏洞2: CAID AUC=1.0 "保底资产"(FALSIFIED)
- 红斑版(单一Aβ42 42残基): AUC=1.0, separation=+0.772
- 标准版(18 IDP, 207残基, 均衡): **AUC-ROC=0.470**(低于随机0.5), separation=−0.011
- disorder head标准基准无效甚至反向, 红斑版不能进论文

### 0.3 漏洞3: "推理期方式4 +0.60"(FALSIFIED, 误标)
- 真相: +0.60实为`oc_v15dg03`的`spearman_bfn_vs_af2_iptm=0.595`(置信头vs AF2的OC排序相关)
- 它是**置信度校准指标**, 不是disorder-guided pliability信号
- 真·推理期方式4(core.py pliability采样)**从未独立验证过**
- **V13原第0步(核查推理期+0.60是否V20翻转)取消——无待核查项, 那个资产不存在**

### 0.4 共性
三漏洞都是"表述把A当B"或"待复测当成立", 非数据造假。资产验证铁律([VERIFIED/PENDING/FALSIFIED]三件套)正是为杜绝。

---

## 1. 资产验证状态盘点(2026-07-04, 强制标签)

| 资产 | 状态 | 说明 |
|---|---|---|
| V17i生成器 Δ+30pp | `[PENDING]` | V20真训对antigen routing影响未测 |
| 方式4训练期 V19 +0.31 | `[FALSIFIED]` | head_seq冻结artifact |
| 方式4训练期 V20 −0.45 reframe | `[PENDING]` | 待消融(本方案§2) |
| ~~方式4推理期 +0.60~~ | `[FALSIFIED]` | 误标,实为置信头OC Spearman;真·推理期disorder_guided从未验证 |
| disorder head CAID红斑1.0 | `[FALSIFIED]` | 标准基准0.47 |
| disorder head 标准基准0.47 | `[VERIFIED]` | 无效(但结论站得住=无效) |
| V15dg03置信头OC Spearman +0.595 | `[VERIFIED]` | 真实资产,但属置信头校准非方式4 |
| 评估墙CDR侧3路证伪 | `[VERIFIED]` | 物理极限真结论 |
| wet-lab候选 | `[PENDING]` | 基于动摇叙事 |

**评分7.5**: 两个核心资产(disorder head+方式4thesis)动摇, 真实[VERIFIED]只剩V17i(待复测)、V15dg03置信头、评估墙limitation。

---

## 2. 第1步: reframe消融(立即, 决定性)

> 定方式4训练期V20的r=−0.45是约束还是loss副产物。这是把方式4训练期从[PENDING]推向[VERIFIED]或[FALSIFIED]的关键。

### 2.1 消融实验
基于V20 config(`bfn_v20_amplify_xpu.yml`), 各跑4000 iters + P2探针:

| 消融 | config改动 | 预期(若reframe=约束,站得住) | 预期(若loss副产物,站不住) |
|---|---|---|---|
| **A1 去diversity** | diversity: 0.05→0 | r维持负(约束不依赖diversity项) | r回正(负是diversity压的) |
| **A2 去anti_degen** | anti_degen: 0.02→0 | r维持负 | r变化(anti_degen干扰) |
| **A3 仅head_seq解冻** | unfreeze_encoder_layers: 2→0(仅head_seq) | 隔离encoder解冻vs head_seq真训影响 | 定位翻转来源 |

### 2.2 判据
- **A1 r维持负**: reframe站得住(disorder信号本身起约束作用, 非diversity loss造的)
- **A1 r回正**: reframe站不住(r=−0.45是diversity loss与主loss拉锯副产物)
- **A3**: 隔离出"head_seq真训"vs"encoder解冻"哪个导致翻转

### 2.3 资产三件套
每消融留: checkpoint + P2产物JSON + config, 标注[VERIFIED] or [FALSIFIED]。

---

## 3. 第2步: disorder head处理(独立项, 与§2并行)

> 标准基准AUC=0.47必须诚实处理, 两路径都需。

### 3.1 选项
- **A 重训**: 用标准DisProt/CAID9数据重训disorder head, 目标AUC>0.7。需算力+数据, RTX 5060可承担则1-2周;
- **B 降级**: 不作独立贡献, 仅作方式4的conditioning输入(即便AUC低仍提供梯度), 诚实标注;
- **C 放弃**: 标准基准无效, 论文作负面结果/不提。

### 3.2 建议先评估
重训成本: 数据(DisProt/CAID9现成?) + 算力(单head重训, 小)。若可承担试A, 否则B(诚实降级)。
红斑版AUC=1.0: 保留产物+标注[FALSIFIED], 不删除不隐瞒, 论文不进结论。

---

## 4. 第3步: 真实可发表资产收尾

> 仅基于[VERIFIED]或消融后[VERIFIED]资产, 不含任何[FALSIFIED]/[PENDING]当成立。

### 4.1 可发表叙事(按§2消融结果)
**若reframe消融站得住(A1 r维持负)**:
1. V17i通用抗原路由(主, 待V20对其复测确认不退步);
2. 方式4训练期disorder-constrains-design(辅, reframe消融验证后);
3. V15dg03置信头OC Spearman +0.595(置信度校准贡献, 独立);
4. 评估墙CDR侧物理极限limitation(真结论: 刚体对接对CDR构象盲视);
5. wet-lab候选(基于V17i+方式4, 重审);

**若reframe消融站不住(A1 r回正, loss副产物)**:
1. V17i通用抗原路由(主);
2. V15dg03置信头OC Spearman(独立);
3. 评估墙limitation(主负面结论);
4. 方式4诚实作"disorder-conditioned探索的混合/负面结果"(不强行正面叙事);
5. disorder head同§3处理。

### 4.2 方法学成文
- 证据链仅含[VERIFIED]项;
- limitation诚实: 评估墙物理极限 + disorder head标准基准无效(若§3选B/C) + 方式4训练期结果性质;
- 三漏洞如实作"方法学探索中的发现", 不隐瞒。

---

## 5. 协作约定

| 约定 | 原因 |
|---|---|
| **资产强制三件套+[VERIFIED/PENDING/FALSIFIED]标签** | 三漏洞都是无标签导致表述错位 |
| **禁把[PENDING]当"已成立"** | V19+0.31和CAID红斑都崩过 |
| **reframe必须消融(A1/A2/A3)才进论文** | r=−0.45可能loss副产物, 审稿人必查 |
| **disorder head红斑版1.0不进论文** | 标准基准0.47, 红斑是假象 |
| **不投入对接/结构预测** | 评估墙3路证伪 |
| **每步留JSON+commit+标签** | 可复现+决策链不断 |
| **数据不造假=如实记录+标注状态** | 三漏洞非造假, 是表述; 强制标签杜绝 |

---

## 6. 产出物

| 步骤 | 产出 | 文件 |
|---|---|---|
| §2消融 | A1/A2/A3 checkpoint+P2 | `logs/bfn_v20_ablate_*/`, `way4_ablate_*_p2_{ts}.json` |
| §2判定 | reframe站住/站不住 | `docs/reframe_ablation_verdict.md` |
| §3 | disorder head决策 | `docs/disorder_head_assessment.md` |
| §4 | 可发表资产报告 | `docs/way4_final_asset_report.md` |

---

## 7. 时间

| 步骤 | 周期 |
|---|---|
| §2 reframe消融(A1/A2/A3各4000 iters+P2) | 3-4天 |
| §3 disorder head重训(若选A) | 1-2周(并行) |
| §4 资产收尾+方法学 | 1周 |

**到reframe判定: 3-4天。** 到可发表收尾: 约2周。

---

## 8. 一句话方案

> 三漏洞修正(V19+0.31 FALSIFIED / CAID红斑 FALSIFIED / 推理期+0.60误标FALSIFIED)后, 真实[VERIFIED]资产只剩V17i(待复测)、V15dg03置信头+0.595、评估墙CDR侧limitation。唯一P0: reframe消融——V20去diversity(A1)/去anti_degen(A2)/仅head_seq(A3)各重训+P2, 定r=−0.45是生物学约束(reframe站住, 走正面收尾)还是loss副产物(站不住, 走诚实负面收尾)。disorder head标准0.47独立处理(重训或降级, 红斑1.0不进论文)。资产强制三件套+[VERIFIED/PENDING/FALSIFIED]标签, 禁把待复测当成立。3-4天到reframe判定, 约2周到可发表收尾。
