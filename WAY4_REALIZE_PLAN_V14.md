# 方式4 真正实现方案 V14 — 高质量高准度 IDP 天然抗体设计

**交付执行**: DeepSeek V4 + 用户 | **日期**: 2026-07-05
**唯一目标**: 针对天然无序蛋白(IDP, 主靶 Aβ42)产出**经湿实验验证的结合抗体**, 且"disorder-pliability 匹配"设计原则有可验证证据。
**前置根因(V4 必读, 三次失败教训)**:
1. entropy ≠ plasticity: 用序列熵当"可塑性"代理, V19+0.31/V20−0.45 全是 loss artifact(消融证伪).
2. BFN disorder head 标准基准 AUC=0.47 无效, 红斑 1.0 是假象.
3. 评估墙 CDR 侧 3 路证伪(graft归零/fold 5.2Å扰动归零/AF2 OOM), 本机不可计算验证 binding.
4. 数据墙: SAbDab 全是折叠抗原, 0.07% IDP, 直接训 disorder 信号学不到.

---

## 0. 核心概念修正(本方案根基)

**前几次方式4 全错的根本: 把"序列熵"当"结构可塑性"。**
方式4 真实 thesis: **抗 IDP 抗体的 CDR(尤 H3)构象可塑性应匹配表位柔性**.
"可塑性" = 结构层面 RMSF(残基均方根涨落), **不是** AA 分布熵.
- 熵高 ≠ loop 柔(熵可在少数残基抖动, loop 仍刚性)
- RMSF 高 = 真·构象柔

**本方案一切围绕 RMSF, 不再用 entropy 做 thesis 代理.** 这是与 V8/V19/V20 的根本区别.

---

## 1. "高质量高准度"的可验证定义

| 维度 | 可验证标准(非"loss 下降") |
|---|---|
| 设计原则成立 | 真实 anti-IDP 抗体观测: epitope RMSF ↑ ↔ CDR-H3 RMSF ↑(S0 统计显著) |
| 生成器学到原则 | 生成 CDR 的 MD-RMSF 与 target epitope RMSF 正相关, 且 survive 消融(非 loss artifact) |
| 抗体可用 | 湿实验 SPR/BLI 测得对 Aβ42 特异结合(KD 量级), **这是"高准度"唯一真尺** |
| 副产物 | 单链 fold pLDDT 合理(可溶性), 无免疫原性红旗 |

**计算验证只作筛选, 湿实验是准度真尺.** 这点不接受替代.

---

## 2. 五阶段方案(每阶段: 主路径 / Go-NoGo / 多意外降级 / 三件套)

### S0 观测性预检 — 最便宜 Go/NoGo, 先于一切训练(3-5 天)

**目的**: 在花任何训练算力前, 先验证方式4 thesis 在真实数据上是否成立. 若不成立, 整个方式4 死, 省下后续所有成本.

**做法**:
1. 收集真实抗 IDP 抗体复合物(8 个 anti-Aβ 已有: 4HIX/5CSZ/3UOT/6CGZ/6D9B/6H3R/7K4V; 从 SAbDab/PDB 补 anti-tau/anti-αSyn);
2. 收集对照: 抗折叠蛋白抗原的抗体(SAbDab 主库, epitope disorder≈0);
3. 测每对抗体的 CDR-H3 **结构与动态柔性**:
   - 晶体 B-factor(CDR-H3 vs framework, 已可用, 4HIX B=26.3);
   - NMR ensemble 结构的 CDR-H3 RMSF(有则用);
   - 短 MD(可选, 5ns)的 CDR-H3 RMSF;
4. 测对应 epitope 柔性: IUPred3 / AlphaFold pLDDT(低=柔) / Aβ42 5-seed RMSF(已有);
5. 统计检验: epitope 柔性 ↔ CDR-H3 柔性 相关(Spearman, p<0.05).

**S0 Go/NoGo**:
- **GO**: 真实抗 IDP 抗体 CDR-H3 显著比折叠抗原抗体更柔, 且与 epitope 柔性正相关 → thesis 成立, 进 S1.
- **NoGo**: 无相关 → 方式4 thesis 在真实数据上不成立, **停止, 不训练**, 转 §5 终极降级.

**多意外**:
- NMR ensemble 不够 → 退 B-factor + 短 MD(够便宜).
- B-factor 受晶格污染 → 用 MD RMSF 为主, B-factor 验证.
- anti-tau/αSyn 结构少 → 至少需 n≥15 对(IDP+折叠各), 不够则从 AlphaFold 预测补, 标注来源.
- MD 本机跑不动(RTX 5060 用于 AF2 已紧)→ MD 用 CPU(短 5ns CDR-only, 小系统可)或云 CPU.

**三件套**: `way4_s0_observational.json` + `collect_anti_idp.py` + `[VERIFIED]`/`[FALSIFIED]` 标签.

---

### S1 真实 disorder/flexibility 信号(弃 BFN disorder head, 1 周)

**目的**: 用外部已验证的 disorder 预测器替代 AUC=0.47 的 BFN head.

**做法**:
1. epitope 柔性信号源(任选可微/可查):
   - **IUPred3**(标准, 长程 disorder);
   - **AlphaFold/ColabFold pLDDT**(低 pLDDT=柔性, 全蛋白可查);
   - **ESpritz**(集成);
   - Aβ42 用已有 5-seed RMSF(实测, 非预测);
2. 对 S0 每个训练样本生成逐残基 epitope flexibility profile;
3. **不重训 BFN disorder head**(它标准基准无效), 直接用外部信号作 conditioning 输入.

**S1 Go/NoGo**:
- **GO**: 外部信号在 S0 数据上一致(IUPred3/pLDDT/MD 三者方向同) → 信号可信, 进 S2.
- **NoGo**: 三者不一致 → 用 MD RMSF 为准(最物理), 标注 IUPred3 偏差.

**多意外**:
- IUPred3 装/跑不动 → 用 AF2 pLDDT(WSL ColabFold 已通)+ 现有 BFN head 作辅助(标注 AUC 0.47).
- 信号在折叠抗原上全 0(无梯度)→ 这正是数据墙, S0 已要求含 IDP 样本, S1 仅处理 S0 的 IDP 子集.
- 逐残基 profile 对齐 CDR 复杂 → 沿用 V7 pair_aggr 映射(已验证可传).

---

### S2 生成目标重定义: RMSF 条件化, 非 entropy loss(1-1.5 周)

**目的**: 训练生成器使其生成 CDR 的**结构柔性(RMSF)** 匹配 epitope 柔性. 这是方式4 的真正实现.

**做法**:
1. 训练 target: 对 S0 真实 anti-IDP 抗体, 其 CDR-H3 的 MD-RMSF profile 作**监督信号**(非 entropy MSE);
2. 生成器(V17i/V20 基础)条件化: 输入 epitope flexibility profile(RMSF/IUPred3), 输出 CDR 序列;
3. **损失**: 让生成 CDR 的**预测 RMSF**(用轻量预测器或 MD)匹配 target profile. 关键: target 是物理 RMSF, 不是序列熵.
4. 若 RMSF 预测器不可微 → 用 Gly/Ser 比例 + CDR 长度 + 低复杂度区(已验证与 RMSF 相关的序列特征)作**中间代理**, 但必须**先用 S0 验证该代理与 MD-RMSF 相关**(否则重蹈 entropy 覆辙).

**S2 Go/NoGo**:
- **GO**: 生成 CDR 的 MD-RMSF 与 target epitope RMSF 正相关(held-out, Spearman p<0.05)→ 学到原则, 进 S3.
- **NoGo**: 无相关 → 若用序列代理, 先验代理 vs MD-RMSF 相关; 若连代理都不相关, thesis 在生成层不成立, 转 §5.

**多意外**(最关键阶段, 多种不通过):
- 序列代理(RMSF 预测器)与真 RMSF 不相关 → **不允许**用该代理, 必须直接 MD(慢但真); RTX 5060 MD 慢则云 CPU.
- 生成器在 held-out 上无相关 → 与 V20 同病. 此时**先消融**(去 loss 项看 r 是否消失)→ 若消失=artifact, thesis 倒; 若不消失=真信号, 进 S3.
- 学到的相关方向反(高 epitope 柔 → 低 CDR 柔)→ 不要 reframe(已证 reframe 是 artifact 陷阱), 直面 thesis 可能错, 回 S0 重审数据.
- 训练 NaN/不稳 → 沿用 V17g nansafe 配置.

**三件套**: ckpt + held-out RMSF 产物 + 消融对照.

---

### S3 防 artifact 训练 + 多 probe 验证(1 周)

**目的**: 杜绝 V19/V20"loss 造信号"重演. 任何信号须 survive 多个独立 probe.

**做法**:
1. **内置消融**(训练时即跑, 非事后):
   - 去 conditioning(epitope flexibility 不输入)→ 信号应消失;
   - 去 loss 项 → 信号应消失(loss 应是必要但非充分);
   - 信号须来自 conditioning + 监督, 非 loss 单独可造;
2. **多 probe**:
   - held-out 真实 anti-IDP 抗体: 生成 CDR 是否匹配其 epitope 柔性;
   - 生成 CDR 的独立 MD-RMSF(非训练用的预测器);
   - 与 S0 观测分布对比;
3. 信号须在 ≥2 个独立 probe 上显著, 且消融证 conditioning 必要.

**S3 Go/NoGo**:
- **GO**: 信号 survive 消融 + ≥2 probe 显著 → 方式4 真实现, 生成可信候选, 进 S4.
- **NoGo**: 仅 loss 依赖(去 loss 信号消失)→ artifact, thesis 在生成层不成立, 转 §5.

**多意外**:
- 单 probe 显著、多 probe 不一致 → 信号不稳, 降级为"探索性"不作论文主张.
- 消融时去 conditioning 信号仍在 → 其实是 loss 造的(同 V20), thesis 倒.
- 算力不足跑全部消融 → 至少跑"去 conditioning"一项(最关键).

---

### S4 验证: 湿实验 binding + MD-RMSF proxy(2-4 周, 资源依赖)

**目的**: "高准度"真尺. 计算作筛选, 湿实验定结合.

**做法**:
1. **候选生成**: S3 通过的生成器对 Aβ42 各表位产 top-N(高 epitope 柔 → 高 CDR 柔, 满足方式4); 对照: Pillar A template-seeded + 从零 V17i;
2. **计算筛选**(非验证, 仅筛):
   - 单链 AF2 fold pLDDT(可溶性, Blackwell 单链可跑);
   - MD CDR-RMSF 是否匹配设计意图(plasticity proxy);
   - 抗退化/免疫原性过滤;
3. **湿实验**(准度真尺): SPR/BLI 测 top 候选对 Aβ42(及寡聚体/单体区分)结合;
4. **云 GPU 结构验证(可选, 论文 supplement)**: 租 sm_90 卡跑 Boltz/Chai template-conditioned, 仅 supplement 不作主判据(评估墙 CDR 侧已证本机不可);

**S4 Go/NoGo**:
- **GO**: 湿实验测得特异结合(KD 量级)→ 方式4 产出可用 IDP 抗体, 目标达成.
- **Partial**: 部分候选结合 → 方式4 方向对, 迭代表位/scaffold.
- **NoGo**: 多批候选无一结合 → 方式4 不产出 binder, 但 thesis(pliability 匹配)可能在 S0/S3 已立; 转 §5.

**多意外**:
- 湿实验排期长/资源缺 → 先冻结计算候选, §5 维持进展.
- Aβ42 寡聚体 vs 单体结合难分 → 设计前明确靶点构象(单体/寡聚/纤维).
- 结合但非特异(也结合其他淀粉样) → 加 counter-screen(转铁蛋白/BSA).
- 云 GPU 仍跑不通 Boltz/Chai → 放弃结构 supplement, 只靠湿实验.

---

## 3. 全方案级多意外总表

| 意外 | 触发条件 | 响应 |
|---|---|---|
| thesis 数据上不成立 | S0 无相关 | 停训, 转 §5(template-seeded+湿筛) |
| disorder 信号不可得 | S1 三预测器不一致 | 以 MD-RMSF 为准, 标注偏差 |
| 序列代理 ≠ RMSF | S2 代理与 MD 不相关 | 禁用代理, 直接 MD(云 CPU) |
| 信号是 loss artifact | S3 去 loss 信号消失 | thesis 在生成层倒, 转 §5 |
| 信号方向反 | S2 学到负相关 | 不 reframe, 回 S0 重审 |
| MD 本机跑不动 | RTX 5060 紧 | CDR-only 短 MD(CPU)或云 |
| 湿实验资源缺 | S4 排不上 | 冻结候选, §5 维持 |
| 云 GPU 仍阻塞 Boltz/Chai | sm_90 也失败 | 弃结构 supplement, 仅湿实验 |
| IDP-抗体数据太少 | S0 n<15 | AF2 预测补, 标注; 或 few-shot |

---

## 4. 资产三件套 + 验证标签(贯穿全程)

每阶段产出强制: 原始产物(JSON) + 脚本 + `[VERIFIED/PENDING/FALSIFIED]` 标签.
- 仅 `[VERIFIED]` 入论文主张;
- `[PENDING]` 不可当成立(V19/CAID/推理期+0.60 三次教训);
- `[FALSIFIED]` 留档不删, 作负面结果/limitation.
- 任何"信号"须消融 survive 才标 `[VERIFIED]`.

---

## 5. 终极降级: 方式4 若全倒, 仍达成"IDP 抗体设计"目标

> 用户唯一目标 = IDP 天然抗体设计. 方式4 是路径之一, 非唯一. 若 S0-S3 证伪方式4 机制, 仍可通过下述达成目标(方式4 降为 selection 辅助):

1. **template-seeded redesign**(Pillar A 已有 69 变体): 从 4HIX/5CSZ 已知抗 Aβ 抗体种子出发, 冻结热点, MPNN redesign;
2. **已知抗 IDP 抗体库筛选**: 直接从文献/PDB 抗 Aβ/tau/αSyn 抗体序列, 人源化 + 亲和力成熟(计算辅助);
3. **湿实验高通量筛选**: 噬菌体/酵母展示对 Aβ42 筛选(湿实验驱动, 计算预测不主导);
4. 方式4 的"pliability 匹配"若 S0 成立但生成层倒, 作**筛选准则**(排除刚性 paratope 抗柔性表位的设计), 非生成驱动;

**底线**: 即便方式4 生成路径全倒, 通过 template-seeded + 湿筛选仍能产出针对 Aβ42 的结合抗体(目标达成), 方式4 作 selection 辅助或诚实负面结论. 这条路**不依赖任何被证伪的计算验证墙**.

---

## 6. 协作铁律

| 约定 | 原因 |
|---|---|
| **S0 是绝对前置, 不通过禁训任何模型** | 三次训练全 artifact, 先验 thesis 真实性 |
| **RMSF 为准, 禁用 entropy 当 plasticity 代理** | V19/V20 全错在此 |
| **外部 disorder 信号(IUPred3/pLDDT), 弃 BFN head** | AUC 0.47 无效 |
| **任何信号须消融 survive + ≥2 probe** | 杜绝 loss artifact |
| **湿实验是准度真尺, 计算只筛不验** | 评估墙 CDR 侧已死 |
| **MD/RMSF 本机不够→云 CPU/GPU, 预算 upfront** | RTX 5060 fold/dock 已证 OOM |
| **[PENDING] 不当成立, [FALSIFIED] 留档** | 三次教训 |
| **不 reframe 负相关** | reframe 是 artifact 陷阱(V20 已证) |

---

## 7. 时间与里程碑

| 阶段 | 周期 | 里程碑 |
|---|---|---|
| S0 观测预检 | 3-5 天 | thesis 真实性 Go/NoGo(最便宜) |
| S1 真信号 | 1 周 | 外部 disorder 信号就绪 |
| S2 RMSF 条件化 | 1-1.5 周 | 生成器学原则(含消融) |
| S3 多 probe 验证 | 1 周 | 信号 [VERIFIED] |
| S4 湿实验 | 2-4 周(资源) | 结合抗体(目标达成) |
| §5 降级(若需) | 2-4 周 | template-seeded + 湿筛 |

**S0 是关键: 3-5 天判定方式4 thesis 是否值得继续.** 若 S0 倒, 直接 §5, 省数周训练.

---

## 8. 一句话方案

> 三次方式4 失败根因: 把序列熵当结构可塑性(错)、BFN disorder head 无效(AUC0.47)、评估墙死、数据墙. 本方案核心修正: **一切围绕 RMSF 真结构柔性, 弃 entropy 代理**. 五阶段: S0 观测性预检(真实抗IDP抗体CDR-H3是否随epitope柔性而柔, Spearman p<0.05)→不通过即停转§5; S1 外部disorder信号(IUPred3/pLDDT,弃BFN head); S2 RMSF条件化生成(序列代理须先验与MD-RMSF相关否则禁用); S3 内置消融+多probe(信号须survive去conditioning,杜artifact); S4 湿实验SPR/BLI为准度真尺+MD-RMSF proxy. 多意外全预案: thesis数据不成立/信号不可得/代理不符RMSF/loss artifact/方向反/MD本机不够/湿实验缺/云GPU阻塞/数据少 — 每条有降级. 终极降级§5: 方式4全倒仍可通过template-seeded(Pillar A)+湿筛选达成IDP抗体设计目标, 方式4降为selection辅助. 铁律: S0前置禁训、RMSF为准禁entropy、外部信号弃BFN head、消融survive、湿实验真尺、[PENDING]不当成立、不reframe负相关.
