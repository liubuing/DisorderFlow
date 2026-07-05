# HDOCK 复活后新方案 V10 — 评估墙验证 + 方式4 binding 对接验证

**交付执行**: DeepSeek V4 / 用户 | **日期**: 2026-07-02 | **基线**: 方式4 thesis 实测成立(r=+0.31,V8) + HDOCK 二进制可执行(差 libfftw3)
**前置铁律**: HDOCK 是带侧链感知+全局搜索的对接打分,正是 V3 纯 Python fft_dock 归零时缺的能力——可能复活评估墙,但**是否真复活必须 native-vs-scram 实测判定,不能假设**。

---

## 0. 项目状态变化(V4 必读)

### 0.1 两个并存突破
1. **方式4 训练期 thesis 实测成立**(V8/V19):unique_aa r=+0.309, entropy r=+0.198,抗退化。机制级可发表。
2. **HDOCK 复活**:之前判"B1 全阻塞、评估墙死"基于 HDOCK+hhsearch 装不下。现 HDOCKlite 本体可执行(`HDOCKlite-v1.1/hdock`, ELF, WSL 加载正常),仅缺 `libfftw3.so.3`。

### 0.2 关键不确定(本方案核心)
HDOCK 能跑 ≠ 能区分 native vs scram CDR。**这是决定性判据**:
- V3 纯 Python fft_dock 四路归零,根因=无侧链感知、backbone-dominated;
- HDOCK 带**侧链感知 + 物理打分 + 全局 FFT 搜索**,理论上是 L0 纯物理(d=2.545)的工业强化版;
- 但需实测:对 4HIX/5CSZ 的 native vs scrambled CDR 对接,看 score 是否可分。

**本方案第一条命脉:先做这个判据,HDOCK 真复活才谈后续。**

---

## 1. 总目标与判据

**目标**:
1. 验证 HDOCK 是否复活评估墙(native vs scram 可分?);
2. 若复活:用 HDOCK 对方式4/Pillar A 候选做对接验证,纯计算验证 binding(不必等湿实验);
3. 若不复活:确认纯计算评估极限,V9 方案不变,binding 交湿实验。

**判据**:
| 结局 | 数据 | 决策 |
|---|---|---|
| **评估墙复活** | HDOCK score 区分 native vs scram(Mann-Whitney p<0.05,sep 显著) | 方式4候选批量对接验证,V9 线C从"湿实验"升级为"本机HDOCK" |
| **部分复活** | native 高于 scram 但 sep 弱(p>0.05) | HDOCK 作辅助排序候选,主验证仍需湿实验 |
| **不复活** | native vs scram 不可分 | HDOCK 也证伪,确认纯计算极限,V9 不变 |

---

## 2. 执行阶段

### 阶段 P0:HDOCK 跑通 + 决定性判据(立即,2-3 小时)

#### P0-1 环境就绪(需 sudo)
```bash
# WSL 内(用户执行,我无法 sudo)
sudo apt install libfftw3-dev
# 验证
cd /mnt/c/biological/DisorderFlow/HDOCKlite-v1.1
./hdock  # 应输出 usage 而非缺库错误
```

#### P0-2 跑通自带示例(零风险环境验证)
```bash
cd HDOCKlite-v1.1
./hdock 1CGI_r_b.pdb 1CGI_l_b.pdb -out test.out
./createpl test.out top100.pdb -nmax 100 -complex -models
# 验收: test.out 有 score,top100.pdb 含对接 pose
```

#### P0-3 决定性判据:native vs scrambled CDR
对 4HIX、5CSZ 各自:
- **native**: 已有 `idp_benchmark_results/4HIX_receptor.pdb` + `4HIX_ligand.pdb`(V3 L0 用过);
- **scrambled**: 用 V3 L0 同款 scrambled CDR graft(可从 `idp_benchmark_known_abs.py` 的 `scramble_cdr` 复现);
- 跑 HDOCK 对接: `./hdock native_receptor.pdb ligand.pdb -out native.out` × N + scrambled × N;
- 提取 top-1 score,比 native vs scram 分布。

**P0-3 验收**:
- native mean score 显著高于 scram(p<0.05)→ 评估墙复活,进 P1;
- 不显著 → 出 §5 No-Go 收尾。

---

### 阶段 P1:方式4 候选批量对接验证(P0 复活后,1-2 天)

> 评估墙复活才做。把方式4 与 baseline 候选批量对接打分。

#### P1-1 候选池
- **方式4 候选**: V19(V8) 对 Aβ42 各表位的 top-N 设计(高 disorder + 高 unique_aa + 无退化);
- **Baseline**: Pillar A 的 69 template-seeded 变体(2b80eb9);
- **Native 参照**: 4HIX/5CSZ 真实 CDR(已知结合,作 score 上界)。

#### P1-2 流程
对每个候选 CDR:
1. graft 进 scaffold → receptor.pdb;
2. Aβ42 表位肽 → ligand.pdb(用 5-seed 构象);
3. HDOCK 对接 → top-1 score;
4. 跨 5-seed 构象取**鲁棒分**(p25,非均值,V3 方案原设计)。

#### P1-3 排序与对照
- 方式4 候选 vs Pillar A baseline vs native 的 HDOCK score 分布;
- **关键判据**: 方式4 候选是否 HDOCK score > from-zero baseline(P3 设计)?是否接近 native?

**P1 验收**: 产出 `hdock_design_scores.json`,方式4 候选 score 显著高于 from-zero baseline → 方式4 不仅机制成立,产出也对接合理。

---

### 阶段 P2:效应放大(V9 线A,并行,1-1.5 周)

**不依赖 HDOCK,本机计算**。V19 只训 4.6%(disorder_proj+head_seq.0 首层),效应量中等。
- A1 解冻 head_seq 全段 → 重训 → P2 探针;
- A2 提 diversity 权重 0.03→0.05;
- 目标 unique_aa r>0.5。

**与 HDOCK 协同**: A 选出效应更强的生成器后,P1 用它产更优候选,HDOCK 验证。

---

### 阶段 P3:可发表资产固化(V9 线B,并行,1.5-2 周)

- **B1 双梯度 Figure**(V7/V18/V19 对照);
- **B2 CAID 换标准 DisProt/CAID9 复测**(修红斑,现 42 Aβ42 vs 15327);
- **B3 方法学成文**:
  - 证据链:disorder head → 推理期方式4(+0.60)→ 训练期方式4(+0.31)→ **HDOCK 对接验证**(若 P1 成功)→ anti-degen;
  - limitation 标注随 P0 结果调整(评估墙复活则 limitation 弱化,不复活则保留)。

---

## 3. 三线并行结构

```
P0 HDOCK 判据(2-3h)──┬─ 复活 → P1 方式4候选对接验证(1-2天)
                      └─ 不复活 → §5 No-Go,V9 不变

P2 效应放大(线A,1-1.5周)─── 本机,不依赖HDOCK
P3 资产固化(线B,1.5-2周)─── 本机,不依赖HDOCK

P1 依赖 P0 复活 + P2 选最优生成器
```

**优先级**: P0 最先(2-3h 判据)→ P2/P3 并行启动 → P0 复活则 P1。

---

## 4. 与 V9 方案的关键差异

| 维度 | V9(评估墙死) | V10(HDOCK 复活待验) |
|---|---|---|
| binding 验证 | 交湿实验/云GPU | **可能本机 HDOCK 纯计算**(P0 判据) |
| 线 C | 资源-dependent,周期长 | P0 复活则立即本地执行 |
| limitation | "纯计算评估极限" | 随 P0 结果,可能弱化 |
| 风险 | 低(机制已成立) | P0 是新变量,需先验 |

---

## 5. No-Go(P0 HDOCK 不复活)

若 HDOCK native vs scram 不可分:
1. 确认纯计算评估在 IDP-scaffold binding 上达极限(L3/L4/B2/HDOCK 四范式同根失效);
2. binding 交湿实验,V9 方案恢复(线C 资源-dependent);
3. 方式4 机制 thesis(P2 r=+0.31)仍独立可发表,不受影响;
4. No-Go 是有效结论。

---

## 6. 风险

| 风险 | 兜底 |
|---|---|
| libfftw3 装后仍有缺库 | `ldd hdock` 查全缺库,逐个 apt |
| HDOCK 对短肽对接不如全长蛋白 | 这是已知;若短肽 score 噪声大,P0 判据用多次 seed 取中位 |
| HDOCK 跑得慢(分钟/对接) | 候选池先小(top-20),非全量 |
| P0 复活但 P1 方式4候选 score 不高 | 说明机制成立但产出待优化,回 P2 放大效应 |

---

## 7. DeepSeek V4 协作约定

| 约定 | 原因 |
|---|---|
| **P0 必须先做,P0 不通过禁做 P1** | HDOCK 复活是假设,需 native-vs-scram 实测 |
| **HDOCK score 跨 5-seed 取 p25 鲁棒分,非均值** | V3 方案原设计,IDP 构象系综需鲁棒 |
| **native 4HIX/5CSZ 作 score 上界参照** | 已知结合,定标尺 |
| **对照: from-zero(P3) vs template-seeded(Pillar A) vs 方式4(V19)** | 三档对照看方式4 产出是否真优 |
| **不重试 AF2-multimer refold / L3L4B2 graft** | 已证伪,HDOCK 是新路径非旧路径 |
| **libfftw3 安装需用户 sudo** | V4 不能 sudo,标清楚 |

---

## 8. 产出物

| 阶段 | 产出 | 文件 |
|---|---|---|
| P0 | HDOCK 判据报告 | `idp_benchmark_results/hdock_native_vs_scram_{ts}.json` |
| P1 | 方式4候选对接分 | `idp_design_results/hdock_design_scores_{ts}.json` |
| P2 | V20 checkpoint + P2对照 | `logs/bfn_v20_*/`, `way4_p2_v20_*.json` |
| P3 | Figure + CAID标准 + 方法学 | `docs/paper_figures/`, `caid_standard_*` |

---

## 9. 时间

| 阶段 | 周期 |
|---|---|
| P0(装机+示例+判据) | 2-3 小时 |
| P1(候选批量对接) | 1-2 天 |
| P2(效应放大) | 1-1.5 周(并行) |
| P3(资产固化) | 1.5-2 周(并行) |

**到 P0 结论: 2-3 小时。** P0 复活则 +2 天到 P1 方式4 binding 验证。

---

## 附:一句话方案

> HDOCK 二进制可执行(差 libfftw3 一条 apt),可能复活之前判死的评估墙——它是带侧链感知+全局搜索的对接打分,正是 V3 纯 Python fft_dock 归零时缺的能力。但"能跑"≠"评估墙复活",必须先做 P0 决定性判据:对 4HIX/5CSZ 的 native vs scrambled CDR 跑 HDOCK,看 score 是否可分(p<0.05)。复活 → 方式4 候选立即本机对接验证 binding(V9 线C从"交湿实验"升级为"本机HDOCK"),不复活 → 确认纯计算极限,V9 不变。P2 效应放大、P3 资产固化并行不受影响。方式4 机制 thesis 已立(P2 r=+0.31),是底线。先 2-3 小时出 P0 结论。
