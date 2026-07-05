# 方式 4 训练期实施方案 V6 — Disorder-Conditioned CDR 生成

**交付执行**: DeepSeek V4 | **日期**: 2026-07-01 | **主线**: 方式 4(平衡有序/无序)
**前置铁律**: 本方案不依赖 B1/Chai/Boltz(B1 三引擎因 hhsearch + RTX 5060 Blackwell CUDA 不兼容全阻塞,已弃)。所有验证用**单链 fold**(Blackwell 可用,L5 已跑通 plddt=0.371)。

---

## 0. 当前状态评估(V4 必读上下文)

### 0.1 已成立的事实(核查过,皆真)
1. **生成器 V17i**:`pair_routing` 已把抗原特征路由进 head_seq(`receiver.py`,concat(res_feat, pair_aggr, antigen_pooled)),Δ(complex−fixbb)=+30pp。但这是**通用**抗原路由,不含 disorder 信号。
2. **Disorder head**:重训有效(Aβ42 pred 0.28→0.59),可对任意蛋白输出逐残基 disorder [0,1]。CAID AUC=1.0 但有红斑(42 IDP 残基 vs 15327 折叠残基,365:1 失衡,单一 Aβ42)。
3. **方式 4 推理期已实现且有效**:`core.py:461-568` 的 `disorder_guided` pliability-scaled entropy,Spearman ρ 从 −0.08→+0.60。
4. **方式 4 训练期:完全未启动。** head_seq 无 disorder 输入项;全代码库无 disorder-aligned entropy loss。
5. **训练数据命门**:phase3 训练集 2976 个抗原,IDP 相关仅 2 个(**0.07%**)。直接在此数据加 disorder loss 学不到信号(epitope disorder 方差≈0)。

### 0.2 已证伪的旁支(V4 不要重试)
- AF2-multimer refold 对短肽-抗体系统性失效(M0: native CDR 重 fold 仅 0.118)。**禁用 AF2 multimer 做 binding 评估。**
- graft-to-scaffold 的所有纯计算度量(L3 物理 / L4-scaffold / L4-fold / B2 ESM-IF)在 native-vs-scram 上四路归零,根因统一:**不含抗体-肽相对界面几何**。
- B1 三引擎(ColabFold/hhsearch、Chai-1 Blackwell CUDA、Boltz)全阻塞。**不再攻坚 B1。**

### 0.3 战略定位
方式 4 是项目**唯一**符合原初 thesis 且真正未做的方向。其核心主张是**机制性**的(disorder 高的表位 → CDR 更可塑),可**独立验证**(单链 fold + MD 测可塑性梯度),**不需要 binding 墙**。binding 最终交湿实验。

---

## 1. 总目标与判据

**目标**:实现训练期 disorder-conditioned CDR 生成,并验证"高 disorder 表位 → 高可塑 CDR"的机制梯度成立。

**三态判据**:
| 结局 | 数据 | 决策 |
|---|---|---|
| **GO** | P0 子集 epitope-disorder 有分布;P2 机制梯度显著(disorder_high CDR 可塑性 > disorder_low,且无退化) | thesis 成立,扩规模,接湿实验测 binding |
| **Partial** | 机制弱(梯度不显著) | 数据 augment 不足 / loss 偏,迭代 |
| **No-Go** | IDP 数据不够造不出子集,或 disorder-conditioned 完全无梯度 | 数据硬约束,外采 IDP-抗体数据,暂停训练侧 |

---

## 2. 阶段 P0:IDP-抗原训练子集(命门,1-1.5 周)

> **P0 不通过则停。** 0.07% IDP 数据上训 disorder loss = 白训。

### 2.1 目标
构建 IDP/柔性抗原复合物训练子集,目标 epitope-disorder 有实际分布(IDP/柔性占比 ≥ 15-20%)。

### 2.2 数据来源(现成资产)
- `data/confidence_idp_v3/`:DisProt 2673 IDP + 143 文献 IDP,带 per-residue disorder 标注;
- `data/phase3_expanded/` + `data/sabdab_phase3_processed/`:现有抗体-抗原复合物(2976 train);
- `data/anti_abeta_refs/`:已知抗 Aβ 抗体(4HIX/5CSZ/3UOT/6CGZ/6D9B/6H3R/7K4V),作 IDP-抗体 anchor。

### 2.3 构造方法
1. **对现有 SAbDab 抗原打 disorder 分**:用重训后的 disorder head 给每个 antigen chain 输出逐残基 disorder,计算 epitope(界面残基)disorder_mean;
2. **筛选柔性复合物**:保留 epitope_disorder_mean > 0.3 的(预期存量少);
3. **IDP 来源 augment**:从 DisProt 取 IDP 结构作"抗原",对每个 IDP:
   - 若有已知抗该 IDP 抗体(PDB) → 用真实 CDR;
   - 否则用 anti-Aβ 抗体 CDR 作近似 anchor(标注来源,供消融);
4. **过采样混合**:P1 训练时 IDP 子集过采样到 ~20%(与折叠集混合),使 batch 内 disorder 有梯度。

### 2.4 产出
- `data/idp_antigen_subset/`:结构 + CDR + epitope disorder 标注;
- `data/idp_antigen_subset/disorder_stats.json`:子集 epitope-disorder 分布直方图。

### 2.5 P0 判据
子集 epitope-disorder_mean 分布**非退化**(有 >0.3 的实际质量,且与折叠集 <0.1 有可分梯度)。否则 No-Go。

---

## 3. 阶段 P1:训练期 disorder-conditioned 实施(架构 + loss,1 周)

### 3.1 改动 1:decoder 条件化(小改,接 V17i)
**文件**:`disorderflow/modules/bfn/receiver.py`(V17i head_seq 段)

现状(head_seq pair_routing 分支):
```python
head_seq_input = concat(res_feat, pair_aggr_proj, antigen_pooled)  # 维度见代码
```
改动:追加 disorder profile 维度
```python
# epitope disorder per-residue (从 disorder head 或 batch 标注) → 对齐到 CDR 维度
# antigen disorder profile 池化/广播到每个 CDR 残基作为条件
head_seq_input = concat(res_feat, pair_aggr_proj, antigen_pooled, disorder_cond)  # +1 维组
```
- `disorder_cond`:表位 disorder profile(逐残基 [0,1]),按 CDR 残基对齐;无抗原时(fixbb)置 0;
- 改动量:receiver.py 数十行 + checkpoint `strict=False` resume;
- **冻 backbone,只训 head_seq + disorder 通道**(算力友好,RTX 5060 可承担)。

### 3.2 改动 2:disorder-aligned entropy loss ⚠️(最大风险点)

> **核心陷阱:无约束高熵 loss 会制造 TTTTTT 退化**——这正是项目一直在过滤的东西。必须把 anti-degen 内建进 loss。

**loss 设计**(在 `disorderflow/modules/bfn/core.py` 训练 loss 段):
```
对每个 CDR 位 i,设 e_i = 对应表位残基 disorder:
  if e_i > τ_high (高柔性表位):
      鼓励 AA 分布高熵(多样性),BUT:
        - 本地 6-mer 禁同残基重复(惩罚 6 连同)
        - Gly/Ser 偏置(柔性骨架),惩罚 T/V 连续
      ⇒ "高多样性 + 柔性残基 + 禁退化",而非"无约束高熵"
  elif e_i < τ_low (有序框架区):
      鼓励低熵(确定性,锁死)
  else (中间):
      标准 loss
L_disorder_align = 加权和,作为辅助 loss(权重 λ=0.05~0.1,可调)
```
**实现要点**:
- 高熵用 AA 分布熵 `H(p_i)` 的上界推动;
- anti-degen 用 k-mer 重复惩罚(可复用现有 `cascade_filter` 的判定逻辑,但放进训练 loss);
- Gly/Ser 偏置用 logit bias 或 loss 项;
- 必须验证:训后 top 设计**不出现** poly-Thr/poly-Val(对比 V17i 的 TTTTTT 问题)。

### 3.3 改动 3:训练配置
**文件**:`configs/train/bfn_v18_disorder_cond_xpu.yml`(新建,基于 v17i)

关键项:
```yaml
resume_from: logs/bfn_v17i_*/checkpoints/best.pt   # 续 V17i
strict: false                                         # 新 disorder 通道
freeze_backbone: true                                 # 只训 head_seq+disorder
max_iters: 4000
lr: 2.e-5
loss:
  seq: 1.0
  disorder_align: 0.05    # 辅助 loss
  anti_degen: 0.02         # 内建退化抑制
data:
  idp_oversample: 0.20    # IDP 子集占 batch 20%
  train_subset: data/idp_antigen_subset/
```

### 3.4 P1 验收
- 训练不 NaN(沿用 V17g nansafe 配置);
- val loss 下降;
- 探针:P1 模型 vs V17i,生成 CDR 的熵/disorder 对齐度。

---

## 4. 阶段 P2:机制验证(绕开 binding 墙,3-5 天)

> **核心:验证机制梯度,不测绝对 binding。** 不用 AF2 multimer、不用 B1/Chai。

### 4.1 三对照实验(同一组表位)
固定一组 Aβ42 表位(含高 disorder 段如 16-24,和相对有序段),对每个表位三种生成模式:
| 模式 | 模型 | 期望 CDR 可塑性 |
|---|---|---|
| A: disorder-conditioned | P1 训练期模型 | 高 disorder 表位 → 最高;低 → 低(梯度) |
| B: unconditional | V17i 原版 | 无梯度(中间值) |
| C: 推理期 pliability | V17i + core.py 推理期 | 中等(已证 +0.60) |

### 4.2 可塑性度量(避 Blackwell 阻塞)
- **(a) AA 多样性**:CDR 采样集的 AA 分布熵、unique 序列数 — 直接统计;
- **(b) 结构可塑性**:对每条设计的**抗体单链**(不给肽)跑 AF2 单链 ensemble(L5,Blackwell 已跑通),取 CDR loop 的:
  - 不同 seed/recycle 下的 CDR Cα RMSD(越大越可塑);
  - 轻量 MD(可选,GROMACS,100ps 短跑)验证 RMSF;
- **关键判据**:`pliability(high-disorder epitope) > pliability(low-disorder epitope)` 在 A 模式成立,在 B 不成立。

### 4.3 退化检查
P1 生成 top20 必须无 >6 连同残基、无 >50% 单一氨基酸(对比 V17i 的退化问题)。

### 4.4 产出
- `idp_design_results/way4_mechanism_{ts}.json`:三模式 × 多表位的可塑性度量;
- 明确 A vs B 的梯度显著性(Mann-Whitney p)。

---

## 5. 阶段 P3:binding 诚实标注(非阻塞,论文级)

- P2 机制验证是**可发表 thesis**:生成了 pliability-matched paratope;
- binding 是否提升:
  - 本机结构预测验证墙(B1)已死,**不再追**;
  - **交湿实验**(SPR/BLI)或云 GPU(租 sm_90 卡跑 Chai/Boltz,仅论文 supplement 需要);
- 论文/报告诚实标注:**"机制可独立验证(P2);binding 验证待湿实验/云 GPU"**。

---

## 6. 执行顺序与依赖

```
P0 IDP-抗原子集 ──(命门闸门:disorder 分布非退化)──→ P1 训练
                                                        ↓
                                              P2 机制验证(单链 fold,不需 B1)
                                                        ↓
                                              GO/Partial/No-Go 决策
                                                        ↓ GO
                                              P3 binding 交湿实验
```

**硬约束**:
- P0 不通过 → 停,不进 P1(数据命门);
- P1 训练若反噬退化(出现 TTTTTT)→ loss 的 anti-degen 项不足,回调;
- P2 机制无梯度且 P1 已对齐 loss → Partial/No-Go;
- **全程不碰 B1/Chai/Boltz**(已证阻塞)。

---

## 7. DeepSeek V4 协作约定

| 约定 | 原因 |
|---|---|
| **P0 是命门,先做且单独验收** | 0.07% IDP 数据上训 disorder loss 必失败 |
| **entropy loss 必须内建 anti-degen,非事后过滤** | 无约束高熵 = TTTTTT 退化(项目老问题) |
| **机制验证用单链 fold,禁用 multimer/template** | AF2 multimer 短肽失效 + B1 三引擎全阻塞 |
| **不重试已证伪路径**(AF2 refold 评估、L3/L4/B2 graft 度量、B1 引擎) | 都有 commit 级证据 |
| **每阶段留 JSON + commit** | 避免 P3 时代 14 个重复未提交 JSON |
| **CAID 红斑待修**(换标准 DisProt/CAID9 数据集) | 单一 Aβ42 的 AUC=1.0 不可外推,需多 IDP 复测 |
| **binding 不在本方案验证范围** | 诚实交 P3/湿实验 |

---

## 8. 产出物清单

| 阶段 | 产出 | 文件 |
|---|---|---|
| P0 | IDP 子集 + disorder 分布 | `data/idp_antigen_subset/`, `disorder_stats.json` |
| P1 | 训练后 checkpoint + 探针 | `logs/bfn_v18_disorder_cond_*/`, `probe_v18.py` |
| P1 | loss/架构改动 | `receiver.py`, `core.py`, `bfn_v18_disorder_cond_xpu.yml` |
| P2 | 机制验证报告 | `idp_design_results/way4_mechanism_*.json` |
| P3 | binding 标注(待湿实验) | RESULTS.md §13 |

---

## 9. 时间预估(RTX 5060 + WSL)

| 阶段 | 周期 |
|---|---|
| P0 数据子集 | 1-1.5 周(含 SAbDab 抗原打 disorder 分 + DisProt augment) |
| P1 训练 + 探针 | 1 周(冻 backbone,4000 iters) |
| P2 机制验证 | 3-5 天(单链 fold ensemble + 统计) |
| P3 | 待湿实验(不阻塞) |

**到 P2 GO/No-Go 决策点:约 2.5-3 周。**

---

## 附:一句话方案

> **B1 三引擎全阻塞 → 评估墙死,binding 交湿实验。方式 4 训练期是唯一未走的原初 thesis,且机制可独立验证(单链 fold,避开 Blackwell 阻塞)。方案:P0 构 IDP-抗原训练子集(解决 0.07% 命门,不通过则停)→ P1 decoder concat disorder + 内建 anti-degen 的 entropy loss(避退化反噬)→ P2 单链 fold ensemble 测可塑性梯度(机制验证,不需 B1)→ P3 binding 交湿实验。判据:P0 数据有梯度、P2 机制显著且无退化即 GO。不再碰 B1/Chai/Boltz。**
