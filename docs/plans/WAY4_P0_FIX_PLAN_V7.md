# 方式 4 下一步方案 V7 — 修复 P0 数据命门 + 逐残基 disorder profile

**交付执行**: DeepSeek V4 | **日期**: 2026-07-01 | **目标**: 把 P0 闸门从 FAIL 翻成 PASS,让 P2 机制梯度可建立
**前置铁律**: 不依赖 B1/Chai/Boltz(全阻塞)。所有验证用单链 fold(Blackwell 可用)。

---

## 0. 为什么是这两步(对症 P0 FAIL 的两个根因)

V4 执行 V6 方案后,P0 闸门 FAIL,有两个**独立且都致命**的根因(已核查 commit + JSON):

### 根因 A:SAbDab 抗原无 IDP,disorder 方差近零
- P0 实测:`data/idp_antigen_subset/disorder_stats.json` — 405 个 SAbDab 抗原,disorder mean=0.152、**std=0.0074**,n_flexible=0,n_ordered=0,**gate=FAIL**。
- SAbDab 天然全是折叠抗原,无 IDP。在这种数据上 disorder-aligned loss 学不到梯度。
- **V6 方案 P0 原本要求"DisProt IDP augment",但 V4 只扫了 SAbDab,没做 augment 这一步。** 这是方案漏执行的一半。

### 根因 B:epitope_disorder 当前是标量,非逐残基 profile
- P2 commit 明确:`"Disorder gradient: not established (epitope_disorder scalar, no per-residue gradient)"`。
- 即使数据整体低方差,SAbDab 局部柔性残基 max disorder 0.44 仍有可利用方差——但只有逐残基 profile 才能利用它。标量把所有残基抹平成同一个值,梯度信息被丢弃。

**结论**:A 补数据(横向,造梯度),B 改表征(纵向,利用已有局部方差)。两者**互补且都必做**——只做 A 仍用标量会丢局部信息,只做 B 没有整体梯度也无意义。

---

## 1. 总目标与判据

**目标**:P0 重建 IDP-有梯度的训练数据 + P1 升级为逐残基 disorder profile,P2 机制梯度成立。

**P0 翻盘判据(新)**:
- 子集中 **n_flexible(epitope disorder_mean > 0.4)≥ 30**,且整体 std > 0.1(有梯度);
- 或:逐残基 profile 上,柔性残基占比 > 5%(局部有梯度)。
- 两条任一成立即 PASS。

**P2 机制判据(不变)**:disorder-conditioned 生成的高 disorder 表位 CDR 可塑性 > 低 disorder 表位(Mann-Whitney p<0.05),且无退化。

---

## 2. 阶段 P0-fix-A:DisProt IDP 抗原 augment(1 周)

> 补 V6 方案 P0 漏掉的一步。目标:给训练数据注入真正的 IDP 抗原,造横向梯度。

### 2.1 数据来源(现成)
- `data/confidence_idp_v3/`:DisProt 2673 IDP + 143 文献 IDP,**带 per-residue disorder 标注**(`idp_summary.json` 显示 n_entries=143,MobiDB per-residue);
- `data/anti_abeta_refs/`:4HIX/5CSZ/3UOT/6CGZ/6D9B/6H3R/7K4V(已知抗 Aβ 抗体,真实 IDP-抗体对);
- 现有 SAbDab phase3(405 扫描过的,保留作低 disorder 对照)。

### 2.2 构造方法
1. **IDP 作抗原**:从 DisProt 取高 disorder(mean>0.5)的 IDP 结构(重建 backbone,复用 `build_abeta_conformations.py` 的 CA→backbone 逻辑);
2. **配对抗体 CDR**:
   - 若有已知抗该 IDP 抗体(如抗 Aβ 的 4HIX/5CSZ、抗 tau、抗 α-syn)→ 用真实 CDR;
   - 否则用 anti-Aβ 抗体 CDR 作近似 anchor,标注 `anchor_provenance`(供消融时区分真实 vs 近似);
3. **混合分层**:构造三档训练样本——
   - **高 disorder 档**:DisProt IDP 抗原(disorder_mean > 0.5),目标 ~15-20%;
   - **中 disorder 档**:SAbDab 含局部柔性残基的(max disorder > 0.4),~20-30%;
   - **低 disorder 档**:SAbDab 折叠抗原(现有 405),~50-65%;
4. **过采样**:训练时高 disorder 档过采样到 batch 占比 ~20%。

### 2.3 产出
- `data/idp_antigen_subset_v2/`:三档样本 + per-residue disorder 标注;
- `data/idp_antigen_subset_v2/disorder_stats.json`:重跑统计(必须 n_flexible≥30 或 std>0.1)。

### 2.4 P0-fix-A 验收
**gate=PASS** 才进 P0-fix-B。若 DisProt 全力 augment 后 n_flexible 仍 < 30 → 数据硬约束成立,转 §5 No-Go 收尾,不强行训练。

---

## 3. 阶段 P0-fix-B:epitope_disorder 标量 → 逐残基 profile(2-3 天)

> 补 V6 方案 P1 的表征缺陷。目标:利用局部柔性残基方差。

### 3.1 改动
**文件**:`disorderflow/modules/bfn/receiver.py`(V18 的 disorder_proj 段)

现状(P1 commit dcdfed7):`epitope_disorder` 是标量 → head_seq 输入 1 维。
改动:升为逐残基 profile
```python
# epitope_disorder_profile: (N, L_epi) per-residue [0,1] from disorder head
# 对齐到 CDR 残基:通过 pair_aggr 或 interface mapping 把表位逐残基 disorder
# 传播到每个 CDR 残基(如:CDR_i 看到的表位 j 的 disorder 加权和)
disorder_cond_per_res = aggregate(epitope_disorder_profile, cdr_to_epitope_map)  # (N, L_cdr, d_proj)
head_seq_input = concat(res_feat, pair_aggr_proj, antigen_pooled, disorder_cond_per_res)
```
- 复用现有 pair_routing 的 pair_aggr 机制(它已做 CDR↔antigen 对齐,disorder 沿同一映射传播);
- `disorder_proj` 输入从标量升维到 profile(零初始化,strict=False resume,沿用 V18 训练 fix `80e4325`)。

### 3.2 数据侧
- 训练 batch 必须带 `epitope_disorder_profile`(逐残基),而非标量;
- SAbDab 样本:用 disorder head 对抗原逐残基打分(已能做),取界面残基 profile;
- DisProt IDP 样本:直接用 DisProt per-residue 标注。

### 3.3 P0-fix-B 验收
- 训练不 NaN;
- 探针:逐残基 profile 在 SAbDab 局部(max 0.44)上能产生 CDR 级梯度(非全常数)。

---

## 4. 阶段 P2 重验:机制梯度(3-5 天)

P0-fix A+B 通过后,重跑 P2 探针(`run_way4_p2_probe.py`),三对照:
| 模式 | 期望 CDR 可塑性 |
|---|---|
| V18-per-residue + IDP-aug(高 disorder 表位) | 最高 |
| V18-per-residue(低 disorder 表位) | 低 |
| V17i unconditional | 无梯度 |

可塑性度量(V6 不变,单链 fold + 轻量 MD,避 Blackwell 阻塞):
- (a) AA 多样性熵;
- (b) AF2 单链 ensemble 的 CDR Cα RMSD(L5,Blackwell 已跑通);
- (c) 退化检查(无 6 连同残基)。

**P2 判据**:高 vs 低 disorder 表位 CDR 可塑性显著分离(p<0.05)且无退化 → **GO**。

---

## 5. No-Go 收尾(若 P0-fix-A 仍 FAIL)

若 DisProt 全力 augment 后数据仍无梯度(<30 flexible):
1. **写 limitation**:SAbDab+DisProt 仍不足以训 disorder-conditioned 梯度,需专门 IDP-抗体复合物数据集(如 curate anti-tau/α-syn 结构);
2. **保底资产**:推理期方式 4(+0.60 Spearman,不依赖训练数据梯度)仍是可发表贡献;V17i 生成器 + anti-Aβ template-seeded(Pillar A)69 变体作为方法学产出;
3. **binding 交湿实验**,纯计算评估墙已死(B1 阻塞)。
4. **No-Go 是有效结论**,界定数据能力边界,不延期。

---

## 6. 执行顺序与硬约束

```
P0-fix-A DisProt augment ──(gate PASS: n_flexible≥30 或 std>0.1)──→ P0-fix-B
                                                                         ↓
                                              P0-fix-B 逐残基 profile(2-3天)
                                                                         ↓
                                              P2 重验机制梯度(3-5天)
                                                                         ↓
                                              GO / No-Go
```

**硬约束**:
- P0-fix-A 不通过 → 停,转 §5 No-Go(数据硬约束,不强行训);
- 逐残基 profile 必须沿 pair_aggr 同映射传播(复用 V17i 已有对齐,不自造);
- 全程不碰 B1/Chai/Boltz;
- anti-degen loss 保留(V18 已生效,勿回退);
- 每阶段留 JSON + commit(避免未提交堆积)。

---

## 7. DeepSeek V4 协作约定

| 约定 | 原因 |
|---|---|
| **P0-fix-A 先做且单独验收(DisProt augment 是上次漏执行的)** | 上轮 P0 FAIL 主因就是这步没做 |
| **逐残基 profile 沿 pair_aggr 传播,不自造对齐** | 复用 V17i 已验证的 CDR↔antigen 映射,降风险 |
| **DisProt IDP 用真实抗 IDP 抗体 CDR 优先,近似 anchor 要标 provenance** | 消融时能区分真实 vs 近似,保诚实 |
| **anti-degen loss 保留,勿回退** | V18 已防住 TTTTTT 退化,是已验证资产 |
| **不重试 B1/AF2-multimer/L3L4B2 graft 度量** | 全有 commit 级证伪证据 |
| **P0-fix-A FAIL 即 No-Go,不延期** | 数据硬约束是有效结论 |
| **顺手提交 WAY4_TRAINING_PLAN_V6.md / 本 V7 入 git** | 决策链入库,避免未提交堆积 |

---

## 8. 产出物清单

| 阶段 | 产出 | 文件 |
|---|---|---|
| P0-fix-A | IDP-aug 子集 + disorder stats | `data/idp_antigen_subset_v2/`, `disorder_stats.json` |
| P0-fix-B | 逐残基 profile 改动 | `receiver.py`(disorder_proj 段)+ 数据 batch 带 profile |
| P2 重验 | 机制验证报告 | `idp_design_results/way4_mechanism_v2_*.json` |
| 收尾 | RESULTS.md §13 + GO/No-Go | 决策落档 |

---

## 9. 时间预估(RTX 5060 + WSL)

| 阶段 | 周期 |
|---|---|
| P0-fix-A DisProt augment | 1 周 |
| P0-fix-B 逐残基 profile | 2-3 天 |
| P2 重验 | 3-5 天 |
| **到 Go/No-Go 决策点** | **约 2 周** |

---

## 附:一句话方案

> **P0 FAIL 的两个根因对症下药:A 补 DisProt IDP 抗原造横向梯度(上次漏执行)、B 把 epitope_disorder 从标量改逐残基 profile 利用局部方差。A 先做单独验收(n_flexible≥30 或 std>0.1 才 PASS),A 不通过即 No-Go 数据硬约束收尾。A+B 通过后重跑 P2 机制验证(单链 fold,不需 B1),梯度显著且无退化即 GO。anti-degen loss 保留,全程不碰已阻塞引擎。约 2 周到决策点。**
