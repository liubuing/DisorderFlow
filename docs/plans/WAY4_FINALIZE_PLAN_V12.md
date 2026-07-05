# 方式4 收尾方案 V12 — V20 P2 真值判定 + 分叉收尾

**交付执行**: DeepSeek V4 | **日期**: 2026-07-03 | **主线**: 方式4 thesis 真值定论 + 可发表资产固化
**前置铁律**: 评估墙 CDR 侧已终局死亡（3 路证伪），**不再投入对接/结构预测验证**；binding 终交湿实验；本机 RTX 5060 8GB 对 fold/dock 普遍 OOM，不追本机结构验证。

---

## 0. 项目状态(V4 必读上下文)

### 0.1 已成立(实测)
- **方式4 thesis 机制成立**(V8/V19 P2): unique_aa vs disorder r=**+0.309**, entropy r=+0.198, 抗退化。**但 V19 是残缺训练**——head_seq 一直被冻结(train.py freeze bug)。
- **V20 真训完成**(head_seq bug 修复后): 4000 iters, val loss **92.10**(vs V19 93.68, 显著降), diversity loss 有实际梯度(0.02-0.04)。**V20 P2 重测未跑——thesis 真值悬**。
- **disorder head** 有效(Aβ42 0.28→0.59), CAID AUC=1.0 但有红斑(单一 Aβ42 42 残基 vs 15327 折叠)。
- 推理期方式4: Spearman +0.60。

### 0.2 已证伪,勿重试(V4 必读)
- **评估墙 CDR 侧 3 路全死**: (1) sidechain graft RMSD=0 归零; (2) fold-then-dock 5.2Å 扰动仍归零(GATE FAIL); (3) AF2 fold 在 RTX 5060 OOM 跑不了。物理结论: 刚体对接对 receptor 内部 CDR 构象盲视。**binding 终交湿实验**。
- B1 三引擎(ColabFold/hhsearch、Chai-1 Blackwell CUDA、Boltz)全阻塞。
- AF2-multimer refold 对短肽失效(M0)。
- pure-Python fft_dock / L3/L4/B2 graft 度量四路归零。

### 0.3 战略定位
项目到**两条主线各自收敛、只差一个数**的状态。方式4 thesis 是机制级可发表(不依赖 binding),是底线。唯一悬而未决: V20 真训后 r 多少。**本方案第 0 步定这个数,后续按结果分叉。**

---

## 1. 总目标与判据

**目标**: 定方式4 thesis 在 V20 真训下的真值,并按结果走可发表收尾路径。

**第 0 步判据(V20 P2 重测)**:
| 结局 | unique_aa r (vs V19 +0.309) | 路径 |
|---|---|---|
| **A 强化** | > +0.5 | §3-A 资产固化 + wet-lab 候选(可发表正面收尾) |
| **B 持平** | +0.2 ~ +0.5, ≈V19 | §3-B 接受 r=+0.31 作论文结论,直接收尾 |
| **C 需重理解** | < +0.2,显著低于 V19 | §3-C 诊断+隔离变量重训 |

---

## 2. 第 0 步: V20 P2 重测(立即,10-20 分钟)

### 2.1 操作
**文件**: `probe_way4_v7_per_residue.py`(已有, V7/V19 P2 用过)

**改动**: 把 `CKPT` 常量指向 V20 best.pt:
```python
# 原(约 line 20):
CKPT = 'logs/bfn_v18_disorder_cond_xpu_2026_07_01__15_51_01_v18_way4/checkpoints/best.pt'
# 改为:
CKPT = 'logs/bfn_v20_amplify_xpu_2026_07_02__21_32_16_v20_win/checkpoints/best.pt'
```

**运行**(WSL, 单链推理不触发 multimer OOM):
```bash
wsl -d Debian -- bash -c 'source venv_wsl/bin/activate; cd /mnt/c/biological/DisorderFlow && python probe_way4_v7_per_residue.py 2>&1 | tee idp_design_results/probe_v20_log.txt'
```

### 2.2 提取关键指标
对照 V19 实测值, V20 须报同样的:
- `unique_aa vs ag_disorder_max` Spearman r (V19=+0.309)
- `entropy_mean vs ag_disorder_max` Spearman r (V19=+0.198)
- 高 disorder unique_aa (V19=6.0)、TopAA% (V19=0.323)、6mer 重复(0/6)
- 落盘: `idp_design_results/way4_v20_p2_mechanism_{ts}.json`, schema 同 V7 P2

### 2.3 验收
三对照须完整: V20_per_residue vs V19(或 V18_scalar) vs unconditional。统计 n≥100。**不通过(跑不出)则先查 GPU 状态/ckpt 加载, 不进 §3。**

---

## 3. 按 r 分叉的收尾路径

### 3-A 路径 A: 强化(r > +0.5) ✅ 最理想
方式4 thesis 在 head_seq 真训下显著强化。
1. **双梯度 Figure**(论文核心): V7(标量)/V18(标量+loss)/V19(逐残基残缺训)/V20(逐残基真训) 四档对照, unique_aa/entropy/disorder 三轴梯度图;
2. **CAID 标准复测**(修红斑): `benchmark_disorder_caid_standard.py`, 换标准 DisProt/CAID9(多 IDP 平衡), disorder head vs IUPred3/AF2-pLDDT, 报 AUC-ROC/PR。AUC>0.85 作独立方法学贡献;
3. **wet-lab 候选生成**: V20 最优生成器对 Aβ42 各表位 top-N(高 disorder + 高 unique_aa + 无退化 + 单链 pLDDT 合理), 交湿实验 SPR/BLI;
4. **方法学成文**: thesis + 证据链(disorder head→推理期+0.60→训练期 r>0.5→anti-degen) + limitation(binding 交湿实验, 评估墙 CDR 侧物理极限)。

### 3-B 路径 B: 持平(r ≈ +0.31, ±0.1)
head_seq 真训未放大, 但 thesis 成立性不变。
1. 接受 r=+0.31 作论文结论, **诚实标注效应量中等**;
2. 不追加训练(边际收益低, 算力约束);
3. 直接进 3-A 的 1/2/3/4(资产固化, Figure 用 V19 数据即可)。

### 3-C 路径 C: 需重理解(r < +0.2)
V19 的 +0.31 部分依赖 head_seq 冻结的偶然配置, thesis 真实效应需重评。
1. **诊断**: 是 unfreeze_encoder_layers=2 扰动了 V17i antigen routing(Δ+30pp), 还是 diversity loss 在真训下被主 loss 压制?
2. **隔离变量重训**: V19 config(disorder_align=0.05, diversity=0.03, unfreeze=0) + **仅解冻 head_seq**(不动 encoder), 隔离"head_seq 真训"与"encoder 解冻"两个变量;
3. 重跑 P2 对比, 定位 r 下降来源;
4. 若隔离重训 r 回升 → thesis 成立, encoder 解冻是干扰; 若仍低 → thesis 效应需重新理解, 降级为弱结论。

---

## 4. 协作约定

| 约定 | 原因 |
|---|---|
| **第 0 步必须先做, 不通过禁进 §3** | r 真值是分叉前提, 跑不出先查环境 |
| **probe 须三对照(V20/V19/unconditional), n≥100** | V19 结论的可比性要求 |
| **不再投入对接/结构预测(HDOCK/fold/template)** | 评估墙 CDR 侧 3 路证伪, 物理极限 |
| **binding 终交湿实验, 不追本机结构验证** | RTX 5060 fold/dock OOM + B1 全阻塞 |
| **CAID 必须换标准 DisProt/CAID9, 不用单一 Aβ42** | 红斑已确认, AUC=1.0 不可外推 |
| **每步留 JSON + commit** | 决策链不断 |
| **方式4 thesis 机制级可发表, 是底线** | 无论 r 多少, 不依赖 binding |

---

## 5. 产出物

| 步骤 | 产出 | 文件 |
|---|---|---|
| 第0步 | V20 P2 报告 | `idp_design_results/way4_v20_p2_mechanism_{ts}.json` |
| 3-A/B | 双梯度 Figure | `docs/paper_figures/way4_mechanism_gradient.pdf` |
| 3-A/B | CAID 标准复测 | `benchmark_disorder_caid_standard.py`, `caid_standard_{ts}.json` |
| 3-A/B | wet-lab 候选 | `idp_design_results/way4_wetlab_candidates_{ts}.json` |
| 3-A/B | 方法学初稿 | `docs/way4_method_draft.md` |
| 3-C | 隔离重训 ckpt + P2 | `logs/bfn_v20b_headseq_only_*/` |

---

## 6. 时间

| 步骤 | 周期 |
|---|---|
| 第0步(V20 P2) | 10-20 分钟 |
| 3-A 或 3-B 收尾(Figure+CAID+候选+初稿) | 1.5-2 周(并行) |
| 3-C 诊断+隔离重训 | 1 周 |

**到方式4 可发表收尾: 路径 A/B 约 2 周; 路径 C 多 1 周诊断。**

---

## 7. 一句话方案

> 评估墙 CDR 侧已终局死亡(3 路证伪), binding 终交湿实验, 本机不追结构验证(RTX 5060 OOM)。唯一 P0: 跑 V20 P2 重测(probe 脚本改 CKPT 指向 V20 best.pt, 10-20 分钟), 定方式4 thesis 真值——V19 的 r=+0.309 是残缺训练(head_seq 冻结)结果。按 r 分叉: >0.5 强化→资产固化(双梯度Figure+CAID标准复测修红斑+wet-lab候选+方法学)收尾; ≈0.31 持平→接受机制级结论(效应量中等)直接收尾; <0.2→诊断+隔离变量重训(head_seq only, 不动 encoder)。方式4 thesis 机制级可发表是底线, 不依赖 binding。约 2 周到可发表收尾(C 路径多 1 周)。
