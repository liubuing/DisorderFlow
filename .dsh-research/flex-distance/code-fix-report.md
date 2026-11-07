# 代码 bug 修复报告：FLEX 打分路径的链角色编码错误

- 日期：2026-11-01
- 范围：`flex_j0_score_v1.py` / `flex_j0b_score_v1.py` / `flex_c1_score_and_rank_v1.py` 的批次构建
- 原则：冻结产物一律不改；修正版另写目录（`results/flex_j0_v2_antigen_roles/`、`results/flex_c1_v2_antigen_roles/`）

---

## 1. 结论（TL;DR）

**FLEX 线 J0a / J0b / C1 的全部数字，是在「抗体链被编码成 padding、抗原被编码成 Heavy、`mask_antigen` 为空」的批次上算出来的。**

修复后（同一 checkpoint、同一批种子、同一队列）：

| 指标 | 冻结值（bug 版） | 修复值 |
|---|---|---|
| J0a 平均 advantage（28 组件 / 9 族） | **0.12419** | **5.5532** |
| J0a 家族阳性 | 7/9（seq_1、seq_2 为负） | **9/9** |
| J0a 家族 bootstrap 95% CI | [0.051, 0.189] | **[5.313, 5.917]** |
| C1 native 排名（已跑 12/26 组件） | 0/26 第 1，平均百分位 0.767 | **12/12 组件 rank 1** |
| stripped NLL（native） | — | **与冻结值逐位相同（diff = 0.0）** |
| complex NLL（native） | — | 中位偏移 **2.27 nats** |

归因干净：`stripped` 上下文完全不变（28/28 组件 diff = 0.0），全部变化来自 complex 上下文的抗原角色。

---

## 2. Bug 定位

### 2.1 代码路径

`modules/bfn_loader.py:204-209` 的通用批次构建走 `mask_region` + `merge_protein` 两个 transform；
`disorderflow/utils/transforms/merge_protein.py:24` 写的是：

```python
data['fragment_type'] = torch.full_like(data['aa'], fill_value=ci)   # ci = 链序号 0,1,...
```

而残差编码器期望的是 `Fragment` 枚举：
`disorderflow/modules/encoders/residue.py:17`

```python
self.type_embed = nn.Embedding(10, feat_dim, padding_idx=0)   # 1: Heavy, 2: Light, 3: Ag
```

`Fragment.Heavy = 1, Light = 2, Antigen = 3`（`disorderflow/utils/protein/constants.py`）。

于是对 `[H, P]` 两链复合物：

| 链 | 应有 | 实际（bug） |
|---|---|---|
| H（抗体重链） | 1 = Heavy | **0 = padding** |
| P（肽抗原） | 3 = Antigen | **1 = Heavy** |

`mask_antigen = (fragment_type == 3)`（`disorderflow/modules/bfn/core.py:91`）→ **全空**，receiver 的抗原分支（抗原→CDR 交叉注意力、disorder 条件路由）从未启用。

`build_region_batch` 本身提供了正确的覆盖分支（`:210-226`），但只在调用方传 `antigen_chains=` 时才生效。
`benchmark_pae_revision_v4.py:48`、`pae_screening_next_v7.py:114`、`harden_pae_surrogate_v4.py:214` 都传了；
**FLEX 的三个打分脚本一个都没传**：

- `scripts/flex_j0_score_v1.py:65-68`
- `scripts/flex_j0b_score_v1.py:117-133`
- `scripts/flex_c1_score_and_rank_v1.py:68-79`

### 2.2 为什么契约测试没抓到

`tests/test_flex_j0_contracts.py:66-86` 只断言了「complex 比 stripped 多的残基数 ≤ 肽长」和 generate 区间长度，
**没有断言 `fragment_type` 的取值或 `mask_antigen` 非空**。四个契约全绿，bug 照过。

### 2.3 实证

```
component 2x6m_A_B | H:97-114 | peptide 6
current complex  len=136 frag_by_chain={'H':[0], 'P':[1], ' ': [0]} antigen_ft3=0
current stripped len=128 frag_by_chain={'H':[0], ' ': [0]}         antigen_ft3=0
FIXED   complex  len=136 frag_by_chain={'H':[1], 'P':[3], ' ': [2]} antigen_ft3=6
FIXED   stripped len=128 frag_by_chain={'H':[1], ' ': [2]}         antigen_ft3=0
```

---

## 3. 修复

新增两个脚本（不改原脚本、不改冻结产物）：

- `scripts/flex_j0_score_v2_antigen_roles.py`
  - complex：`context_chains=['P'], antigen_chains=['P']`
  - stripped：`context_chains=[], antigen_chains=[]`（H → Heavy）
  - 复用 `flex_j0_score_v1.candidate_nll` / 种子 / `FIXED_T`，只改批次构建
- `scripts/flex_c1_score_and_rank_v2_antigen_roles.py`
  - 同样修复；复用冻结的 MPNN / composition 缓存（与角色编码无关）
  - **顺带修掉 top1/top5 统计 bug**（原脚本 `:311-314` 统计的是 `percentile == 0`，即"从不垫底"）

产物：`results/flex_j0_v2_antigen_roles/`、`results/flex_c1_v2_antigen_roles/`（每个组件一份 scoring JSON + 汇总）。

---

## 4. 复跑结果

### 4.1 J0a（28 组件 / 9 族，全量）

| 族 | 冻结 advantage | 修复 advantage |
|---|---:|---:|
| seq_0 | +0.2283 | **+6.294** |
| seq_1 | −0.0553 | **+4.987** |
| seq_2 | −0.0285 | **+5.521** |
| seq_3 | +0.1068 | **+4.928** |
| seq_4 | +0.0446 | **+5.790** |
| seq_5 | +0.1946 | **+5.928** |
| seq_6 | +0.2522 | **+5.707** |
| seq_7 | +0.2122 | **+5.230** |
| seq_8 | +0.1629 | **+6.199** |

组件级：min 2.499 / max 6.738；家族等权均值 5.6205，家族 bootstrap 95% CI [5.3133, 5.9172]。

### 4.2 C1（26/26 组件，全量完成）

`results/flex_c1_v2_antigen_roles/c1_v2_summary.json`：

| 方法 | 平均百分位 | 家族 CI | top-1 | top-5 | 中位排名 |
|---|---:|---|---:|---:|---:|
| flex_ecls（修复后） | **1.0000** | [1.000, 1.000] | **26/26** | 26/26 | **1** |
| proteinmpnn_nll | 0.6374 | [0.587, 0.668] | 0/26 | 0/26 | 29.5 |
| composition_logistic | 0.5440 | [0.545, 0.630] | 6/26 | 11/26 | 7 |

- 预注册门：**pass**（赢组成 8/8 族、赢 MPNN 8/8 族、CI 下界 1.0 > 0.5）。
- native 相对最好 design 的判分优势：最小 0.067、中位 **1.770**、最大 3.596 nats。

**必须写在前面的解释**：这是"native 恢复"任务，不是结合任务。native 被放在**它自己的晶体骨架**上打分，而竞争者是 ProteinMPNN 在同一骨架上采样的设计。一个骨架条件序列模型在"自己的骨架"上偏好原生序列，测的是**序列-骨架自洽性**，不是亲和力。修复后它 26/26 第 1，恰好说明这个端点太容易，而不是说明它会设计。任何"排序能力"主张都必须先过异源骨架对照（见 §6）。

### 4.3 两个控制实验（2026-11-01，`results/flex_cross_backbone_v1/`）

**控制 1：异源骨架矩阵**（同 H3 长度的组内，每条 native 序列 × 每个骨架）

- 队列只有 26 个组件、**每个长度组最多 2 条不同序列**（len 11 的 7 个组件共用同一条序列），所以这是"两两比较"而非完整矩阵。
- 在有异源序列的 17 个骨架上，**自身序列 17/17 排第 1**，平均领先 5.11 nats（中位 4.87）。
- 解读：存在骨架特异性偏好，但独立对数只有 5–6 对，**不足以证明表位特异性**；它也完全可以由自洽性解释。

**控制 2：肽序列置换**（坐标不变，P 链序列换成组成匹配的打乱序列）

- **18/26 偏好自身肽序列**，平均 Δ = **0.076 nats**（自身更好）。
- 对照：native 相对最好 design 的优势中位 **1.770 nats**。
- 解读：**肽身份只贡献约 4%** 的判别幅度；裁判的主导项是 H3 序列本身"像不像天然 + 与骨架搭不搭"，不是"这条 H3 认不认这个表位"。

---

## 5. 这改变了什么 / 没改变什么

**改变**：
1. J0a 的效应量被低估约 45 倍；两个"阴性家族"（seq_1、seq_2）翻正。
2. C1 的"native 从未第 1"结论**至少部分**是 bug 造成的——修复后它在已跑的每个池里都第 1。
3. 任何引用 0.124 / 0.141 / 0.767 的下游文字都需要重算。

**没改变**：
1. 端点仍然是 **native 识别**，不是结合。修复让裁判更像"识别天然序列"，但仍然没有任何 binding 标签参与。
2. `mask_antigen` 启用后偏好天然序列，很可能同时包含"天然 H3 更常见"的真实统计与"见过同源复合物"的记忆成分——两者都指向"native-ness"，不是"binder-ness"。
3. J0 队列的 PDB **不在** BFN 训练/验证 split 里（train 2950 / val 327，PDB 级 0 重叠），但 H3 序列级的同源重叠尚未检查。

---

## 6. 结合端点回顾（AAYL / AbBiBench，2026-11-01）

`results/aayl_bfn_v2_antigen_roles/`：508 条 H3-only 变体（AAYL49 281 / AAYL51 227），
同一抗原、两个亲本谱系，AF3 预测复合物，端点为 `9 − median(log10 估计 Kd)`（越大越强）。
期望方向：Spearman(ECLS, endpoint) < 0。

| 谱系 | n | 修复后 BFN ECLS | 95% bootstrap CI | 冻结 ProteinMPNN ECLS | 负 complex NLL 基线 |
|---|---:|---:|---|---:|---:|
| AAYL49 | 281 | **+0.062**（方向错） | [−0.059, +0.184] | −0.121 | −0.068 |
| AAYL51 | 227 | **−0.049**（方向对，极弱） | [−0.175, +0.076] | −0.071 | +0.060 |
| 家族内中心化合并 | 508 | **+0.024** | — | — | — |

**结论：修复后的 BFN 裁判与实测（估计）亲和力之间没有可用相关。** 两个谱系符号相反、置信区间都跨 0，
合并后约 0.02。它和冻结的 ProteinMPNN ECLS 相关也只有 −0.11（两个模型互不一致）。

这一结果同时把 §4.2 的 C1 26/26 钉死为**天然感/自洽性**：在大幅刻意改变结合强度的真实变体集上，
它读不出任何方向。

---

## 7. 待完成

1. ~~J0b v2（系综边际化）需用同一修复重跑~~ **已完成（2026-11-07）**：
   `scripts/flex_j0b_score_v2_antigen_roles.py` → `results/flex_j0b_v2_antigen_roles/`。
   26/26 组件阳性（min +2.570），8/8 族，mean **+5.679** [5.336, 6.006]；
   系综边际化**无衰减**（vs J0a v2 Δ=−0.126，两种边际化次序差 0.045）。
   stripped 缓存等价校验 26/26 max diff = 0.0
   （`results/flex_j0b_v2_antigen_roles/stripped_equivalence_verification.json`）。
2. ~~给 `tests/test_flex_j0_contracts.py` 加断言~~ **已完成（2026-11-07）**：
   `test_antigen_roles_batch_marks_peptide_as_antigen`（antigen_count == 肽长、heavy > 0）。
3. ~~H3 序列级同源检查~~ **已完成（2026-11-07）**：
   `scripts/flex_h3_homology_screen_v1.py` → `results/flex_h3_homology_screen_v1/`。
   全量 SAbDab2 2026 快照（15,617 复合物，排除 24 个 J0 自条目）：
   H3 精确命中 **2/28**、肽 **5/17**，全部落在同表位家族复合物内；
   H3 平均共享 8-mer 比例 0.071。边界：基座 checkpoint 的确切 split
   （train 2950 / val 327）本地不可恢复，本筛查只是该宇宙上的上界。
4. ~~若要继续异源骨架矩阵，队列需扩到每个 H3 长度 ≥4 条不同序列~~ **已完成（2026-11-07，用户授权执行）**：
   `scripts/flex_cross_backbone_matrix_v2_expanded.py` →
   `results/flex_cross_backbone_v2_expanded/`。扩列池 = A1 坐标门全过组件
   （107 组件/36 结构，0 跳过）；入选长度组 **9/11/12/14**（各 ≥4 条不同沉积序列），
   **62 骨架 × 组内全候选 = 1,130 评分对**。结果：**self 严格最优 62/62**
   （交换零假设每骨架 1/4–1/5，联合概率 <10⁻³⁰），对最好异源序列边际
   **1.53–6.40 nats**（mean 4.35 / median 4.48）；与 v1 共享的 18 骨架
   self_ecls 逐位一致（max diff = 0.0）。结论：修正后 C1 26/26 = 序列-骨架
   自洽性，现在有真实统计功效；结合肽置换（~4%）与结合端点（≈0）对照，
   不是表位特异性。协议披露：扩列行绕过战役级排除门（暴露/隔离/先前面板/
   克隆碰撞），仅支持"同骨架序列偏好"对照，不支持 native 识别主张。

---

## 8. 下一步方向

裁判的"结合判别"能力已被回顾测试否定，因此**不能作为设计选择器**。剩余可走的路只有两条：

- **A（推荐）**：把模型线降级为边界研究；生成外包（RFdiffusion/ProteinMPNN），做一次真实测量
  （10–20 条设计 + BLI/SPR，或 Adaptyv 约 21 天）。项目目标（一个实测 Kd 的 IDP 结合体）仍可达，
  但它不依赖这个裁判。
- **B**：只保留一个未被检验的模型假设——"真构象系综鲁棒性"（mean − λ·std，20–50 构象）。
  它同样要先过结合标签才知道有没有用，且本次回顾已经降低了先验。
