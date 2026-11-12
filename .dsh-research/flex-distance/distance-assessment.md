# DisorderFlow 距离「柔性表位（无序蛋白）抗体设计」还有多远

- 评估日期：2026-11-01
- 评估对象：`D:\biological\DisorderFlow`（release 分支，HEAD `13160681`）
- 材料：
  - 仓库只读审计：`repo-audit.md`（457 行，数字带 文件:行号）
  - 外部文献对标：`lit-review.md`（26 篇，5 篇读过全文）
  - 本报告作者的独立复核：C1 排名重算、capability manifest、冻结件与 checkpoint 哈希、湿实验状态机
- 口径声明：下文「项目内部数字」= 仓库冻结产物；「文献数字」= 已发表/预印本；两者不混算。

> **问题解读**：我按「针对无序/柔性表位（IDP/IDR）的抗体或抗体片段（Fab/scFv/VHH）设计」来读。
> 若你指的是**非蛋白骨架的抗体替代物**（aptamer / peptoid / 分子印迹），它不是同一个问题，见 §7 一句话。

---

## 1. 一句话结论

**以「做出一个实测 Kd 的无序表位结合体」为完成定义：现在还差 6 道门里的 5.5 道；工程距离约 3–6 个月 + ¥1.5–4 万，但必须先换掉生成方法，且不能用项目自研打分器做选择器。**

**以「用项目自研的柔性感知打分/设计方法做成」为完成定义：目前没有任何一条已知可行路径**——判别器打不过 20 维组成基线、确认性数据门是数学死结、湿实验为零。这条口径下，「距离」不是几个月，而是「未知」。

项目自己的冻结能力清单已经写了同一句话（`publication/idp_platform_capability_v1.json`）：

> "This platform is a research tool for IDP-targeted antibody design exploration. It does not claim to produce validated IDP binders. No wet-lab validation has been performed."

---

## 2. 先把「实现」定义清楚：L0–L6 门槛

文献里没有任何一篇把「计算设计一个结合体」等同于「做出了结合体」。按证据强度分级：

| 级 | 门槛 | 文献代表（无序/柔性靶点） |
|---|---|---|
| L0 | 计算生成 + 计算排序，无湿实验 | AlphaProteo in silico、BindCraft 过滤链、本项目的 ECLS/PAE |
| L1 | 可表达、可纯化 | Jiang 2025 的失败首因之一就是低表达 |
| L2 | **体外实测 Kd**（BLI/SPR/ITC/FP） | Liu 2025：IDP/IDR binder **3–100 nM** |
| L3 | 特异性反筛 | Liu 2025：11 binder × 8 靶点 all-by-all |
| L4 | 结构或位点证据 | Liu 2025：G3BP1 共晶整体 Cα RMSD **0.8 Å**；或定点突变破坏结合 |
| L5 | 功能读出 | amybinder 抑制/解聚纤维、细胞内应激颗粒解体 |
| L6 | 体内 / 成药性 | 小鼠成像、脑穿透；de novo 无序 binder 的体内 PK 仍属空白 |

**关键锚点**：唯一系统的「无序靶点 de novo binder」论文是 [Liu et al., Nature 2025](https://doi.org/10.1038/s41586-025-09248-9)（seq-input RFdiffusion，靶点与 binder 构象同时自由采样），Kd 落在 **3–100 nM**，但命中率是 best-of-N：amylin 先测 96 个设计得 4 个命中（100–454 nM），双侧 partial diffusion 优化后 174 个设计里 107 个结合（最好 3.8 nM）；prion 48 个设计→14 nM；VP48 95 个设计最好 750 nM。

**反面对照**：[Jiang et al., bioRxiv 2025](https://doi.org/10.1101/2025.02.07.636769) 用 RFdiffusion 做 6 个靶点 × 5 个设计，除 Strep-Tag II 外基本全败（低表达/非特异/测不到亲和力）。**小规模战役在设计上没有统计功效**。

所以「实现」的最低法定门槛是 **L2**：一条实测的、带曲线与拟合的 Kd。没有它，任何「设计了抗体」的表述都不成立。

---

## 3. 项目现在站在哪：逐门记分卡

| 门 | 要求 | 项目现状（证据） | 状态 |
|---|---|---|---|
| **G0 判据** | 能把「会结合的候选」与「不会结合的候选」分开 | ① ECLS native-vs-shuffle 为正：31 结构/15 簇，均值 **0.172281**，CI [0.059156, 0.291920]，12/15 簇（`results/publication/h3_ecls_temporal_final_v1/final_decision.json`）；46 簇适配集 0.217481。② FLEX 复制（BFN 底模）：J0a +0.124 [0.051, 0.189]（9 族 7 正）、J0b +0.141 [0.069, 0.204]（8 族 7 正）。③ **排名门失败**：C1 里 native 平均百分位 0.767，但**从未第 1（0/26）**，前 5 名 **9/26**，中位第 15.5；修正方向后仍**输给 20 维组成基线 5/8 族**。④ Phase D（binder vs non-binder）只有草案：确认性需要 **≥24 家族**，现在 8–9 族，同效应功效 **~0.24**；精确组成匹配产率 **1.63%**（200 家族过 1 个），已被自己的数据证伪。 | **半通**：有「native 识别」，没有「结合判别」 |
| **G1 生成** | 能产出有机会折叠/表达的候选 | 只有**固定骨架 CDR-H3 单环重设计**（ProteinMPNN vanilla v_48_020 / BFN）；无 scaffold 级或 de novo binder 生成。BFN native recovery **0.080** vs ProteinMPNN **0.326**（且都是 AF2 端点，不是结合）。C1 的 1,300 个设计去重后 789 个，最差组件 50 个设计只有 7 个不同。 | **弱** |
| **G2 表达** | 拿到蛋白 | **0**。`outputs/non_abeta_idp_wetlab_handoff_v1/wetlab_status.json` 六个构建体 × 九阶段全部 `not_executed`，`results_present=false`；BLI 结果模板是空表。物料停在 `ready_for_protocol_review_and_execution`。 | **0** |
| **G3 Kd** | 实测结合 | **0**。无本项目 SPR/BLI/ITC/FP；唯一「实验端点」是借来的公开 AlphaSeq 估计（AAYL 线），且结论为负（ECLS 与亲和力 Spearman −0.121/−0.071）。 | **0** |
| **G4 特异性** | 交叉反筛 | **0**。验收判据已预注册（信号 ≥ 外部阴性 3.0×、打乱/BSA ≤ 目标 20%、两个独立批次），未执行。 | **0** |
| **G5 功能** | 细胞/聚集/毒性读出 | **0**。 | **0** |
| **G6 泛化/前沿** | 多家族、可复现、前瞻 | 开发队列 14 PDB / 9 新家族；界面 PAE 的绝对判别只在 **n=1 骨架**（6YXM，recycle 1 五模型 5/5 同向，差 2.0–4.9 Å）「待复现」，多骨架复制（33CI/5V6L/6MQE/6MQM/6P60）**PENDING**，单查询 4 小时超时。 | **未完成** |

**记分**：6 道门里通了 0.5 道（G0 的一半）。按「能否被称为设计成功」算，完成度约 **8–15%**。

---

## 4. 三道最硬的墙（为什么不是「再跑几个月」就行）

### 墙 1：组成混杂 —— 你的裁判和 20 维氨基酸组成打平

C1 的准确失败位置不是「输给 MPNN」，而是**连 20 维组成都赢不了**：修正方向后裁判赢 MPNN 6/8 族，却只赢组成基线 5/8 族（门要求 ≥70%）。分解显示裁判对「组成打乱」近乎完美（家族均值可达 1.00），对「近 native 的 MPNN 设计」崩塌（最低 0.00）。

这不是调参能解决的：**任何没有锁住组成的判别主张，都会被一个不用结构信息的基线解释掉**。项目自己已经把这条写成了下一条线的冻结禁令（Phase D 冻结条件 2）。

### 墙 2：数据 —— 精确组成匹配是数学死结

Phase D 需要「≥24 个抗原家族 × 每族 ≥2 binder × 每族 ≥1 个组成匹配的非结合体」。做法试过了：

- 全库阴性池 123.5 万条 → 709,890 独立序列 → 672,336 个不同组成（平均每组成 1.056 条）；
- v3 组装：200 家族**只过 1 个**（产率 0.96%）；
- v3b 扩池：产率升到 **1.63%**，通过家族仍是 **1/200**，339 个 binder 只有 8 个匹配。

结论是项目自己下的：**「精确组成匹配在任何可达池规模下不可行——组成空间稀疏度决定的数学事实，非工程问题。」**

同时公开数据也在收缩：配对 H/L 抗体–短肽空间**新增合格家族 = 0**；VHH 口径只剩 3 个暴露干净条目 / 1 个可确认组件。

### 墙 3：柔性 —— 现在的「柔性」比 IDP 的真实系综小 1–2 个数量级

项目的柔性建模是**沉积位姿邻域的受限 OpenMM 系综**：J0b 的构象散布 0.25–0.4 Å，T1 单姿态衰减 ~0.20 ECLS 的预言在这批队列上甚至没复现（差 −0.017）。

而 IDP 表位的真实构象异质性在 Å–nm 级。文献的第一性困难说得很直白（[Vázquez Torres et al., Nature 2023](https://doi.org/10.1038/s41586-023-06953-1)）：**把柔性肽折叠成特定构象要付熵代价**，这笔代价直接吃掉结合自由能。项目目前没有任何机制去给这笔熵代价定价或采样——打分器用的是「附近几个构象的平均」，不是「无序系综」。

### 墙 4（可解）：湿实验为零，但这是钱和时间问题，不是科学未知

三份计划（`WETLAB_VALIDATION_PLAN_V15.docx`、`QLU_NATBIOTECH_ROADMAP_V1.md`、Aβ 盲法 BLI 物料 + 预注册验收判据）+ 25 条 scFv 序列物料（S0 完成）+ 空结果模板。**一步都没执行**：路线图计划 2026-09 做 BLI，今天是 2026-11-01，`wetlab_status.json` 最后更新 2026-07-28，全项仍是 `not_executed`。

---

## 5. 外部坐标：别人做到哪

| 工作 | 靶点 | 做到什么 | 对本项目意味着 |
|---|---|---|---|
| [Liu et al., Nature 2025](https://doi.org/10.1038/s41586-025-09248-9) | 无序 IDP/IDR（amylin、C-peptide、VP48、G3BP1、IL-2RG、prion） | seq-input RFdiffusion，**3–100 nM**，共晶 0.8 Å，11×8 交叉反筛，细胞功能 | 「无序表位可设计」已不是新颖性；新颖性只能来自方法差异 |
| [Vázquez Torres et al., Nature 2023](https://doi.org/10.1038/s41586-023-06953-1) | 柔性螺旋倾向肽（PTH、Bim…） | partial diffusion 直接得 **pM 级**、无实验优化 | 柔性设计的成本项是熵罚；静态单构象打分原理上漏项 |
| [Pacesa et al., Nature 2025](https://doi.org/10.1038/s41586-025-09429-6) | **有序** 12 靶点（BindCraft） | 实验成功率 **10–100%，均值 46.3%**，nM | 这个数字**不能**外推到无序靶点 |
| [Jiang et al., bioRxiv 2025](https://doi.org/10.1101/2025.02.07.636769) | 6 靶点 × 5 设计 | 几乎全败（低表达/非特异） | 小规模战役没有统计功效，必须报分母 |
| [Islam et al., Protein Sci 2023](https://doi.org/10.1002/pro.4875) | α-syn 14 肽 | 共晶：**伸展主链 + β 补体**识别，特异性在 48–52 | 线性/柔性表位的几何应优先假设 β-pairing，而不是凹槽包裹 |
| [Sappington et al., Nat Commun 2026](https://doi.org/10.1038/s41467-025-67866-3) | 亲水 β 边缘链 | β-pairing 条件化 RFdiffusion，pM–中 nM，KIT 共晶 | 这是最接近的可比方法；但靶点是折叠蛋白边缘链 |
| [Chenna et al., bioRxiv 2025](https://doi.org/10.1101/2025.06.21.660895) | α-syn 无序 C 端 | pLM 排序：同源 CDR ρ=0.86，换新 CDR **ρ=0.18**，无湿实验 | 「只靠打分排序」的最强正例自己就崩了泛化 |
| [Bozkurt, bioRxiv 2026](https://doi.org/10.64898/2026.04.13.718094) | 公开基准复盘 | 叠加生物学先验把命中率 13.8%→38.6% | disorder 特征**方向会翻转**，不是单调有益 |

**结论**：外部世界已经把「柔性/无序表位可设计」做成了（2025 Nature ×2），所以这个项目面对的**不是未知的科学问题，而是一个已知答案的方法竞赛**。比的是：在同等实验预算下，你的方法与 Liu 2025 相比好在哪里。

---

## 6. 距离的三种量纲

### 6.1 工程量：到「第一个实测 Kd」≈ 3–6 个月，¥1.5–4 万

| 阶段 | 内容 | 时间 | 钱 |
|---|---|---|---|
| 生成 | RFdiffusion / ProteinMPNN（**不用 ECLS 选择**），50–100 条 | 1–2 周 | 0（GPU 已有） |
| 过滤 | AF2-MSA + 界面 PAE（项目唯一冻结部署的信号）；加表达/疏水性过滤 | 1–2 周 | 0 |
| 合成 | top 10–20 条基因，含 4HIX native 阳性对照 | 1–2 周 | ¥5k–14k |
| 表达纯化 | 校内大肠杆菌/HEK293 + Ni-NTA（或外包） | 1–2 月 | ¥3k–10k |
| BLI 初筛 | Octet，10–20 样 | 1–2 周 | ¥3k–5k |
| SPR 定量 | Biacore，hit 候选 6 浓度梯度 | 1–2 周 | ¥5k–10k |
| 反筛 | α-syn / tau / BSA / 打乱表位 | 并入上两步 | — |

这条路**不需要**项目自研的任何模型：可复用资产是 AF2/WSL 管道、界面 PAE 相对排序（跨骨架 Spearman 0.70–0.87，仅相对排序）、以及一整套预注册/审计纪律。

### 6.2 科学量：自研方法被证实 —— 没有已知路径

要证明「我们的柔性感知打分/设计方法能选出真正会结合的候选」，必须过三道门，而三道都没有已知可行路径：

1. **判别力**：必须在 ≥24 家族上，对「实验证实的 binder vs non-binder」赢过组成基线（Phase D）——数据不存在，且精确匹配数学不可行。
2. **排序力**：必须在「competent designs」池里把 native 排到前面——C1 已证伪（从未第 1）。
3. **前瞻效力**：设计 → 表达 → 测 Kd → 与随机/基线设计比较——零实验数据。

这不是「多跑几个月」。要么换任务定义（改用湿实验标签训练），要么承认这是方法边界研究。

### 6.3 论文量：现有资产能撑 2–3 篇边界研究，撑不起 1 篇设计成功

- ECLS 主稿（Bioinformatics，冻结，作者审阅中）
- PAE 界面边界稿（PLOS Comp Biol 路由，冻结部署合同）
- FLEX 边界稿（本地草案，未冻结）
- Phase D 草案：**倾向终止**（B 选项），论据从「未试」变成「已测得的数学事实」

iBEC / Nat Biotech 那种「AI 设计抗体 + SPR 验证」的叙事，需要 L2 以上；现在一条都没有。

---

## 7. 若「无蛋白质抗体」指的是非蛋白骨架（aptamer / peptoid / MIP）

那是换项目，不是走近路：

- **Aptamer**：亲和力不差（α-syn 纤维 2′F-Py RNA aptamer **Kd 6.6/8.8 nM**，[Bouvier-Müller et al., NAR 2024](https://doi.org/10.1093/nar/gkae544)），但 **SELEX 需要真实靶标材料，不能从一段序列「设计」出来**——与项目「从序列/结构生成候选」的核心资产完全不匹配。
- **大环肽/peptoid**：nM 级（29–53 nM），但路线是「库设计 + 亲和选择-MS」，需要合成与质谱设施。
- **MIP「塑料抗体」**：便宜稳定，但报告的是 LOD 不是 Kd，且必须先有模板——**是印迹，不是设计**。

一句话：**只有蛋白 binder 路线具备可编程性**；换到非蛋白骨架，亲和力不输，但「设计」这个词就没了。

---

## 8. 最短可执行路径（含判据与停止规则）

| 里程碑 | 动作 | 通过判据 | 失败处理 |
|---|---|---|---|
| **M1（2 周）** | 把 6YXM 的 recycle-1 界面 PAE 结果复制到其余 5 个骨架（脚本已有：`results/idp_rd_c1/run_msa_contrast_multiscaffold.py`；当前 PENDING 因 4h/查询超时，需分批/降模型数） | ≥5 骨架中 ≥4 个「native 更低 PAE」且方向一致 | 该信号降级为「单骨架先导」，不得用于候选选择 |
| **M2（1 月）** | 用 RFdiffusion/ProteinMPNN 生成 50–100 条针对 4HIX-DAEFRH 与一个 tau/α-syn 表位的候选；AF2 + PAE 过滤 | 至少 80% 通过表达可行性预筛（无 Cys 冲突、净电荷合理） | 生成器问题，回到方法定义 |
| **M3（2–4 月）** | 合成 top 10–20 + 4HIX native 阳性对照，表达纯化 | 至少 5 条可溶单体（SEC） | 报分母，不报 best-of-N |
| **M4（1–2 月）** | BLI 初筛 → SPR 定量 + 三类反筛 | ≥1 条 Kd ≤ 1 µM 且对 α-syn/tau/BSA 无结合 | 若全阴：按边界研究结题（诚实负面也是结果） |
| **M5（可选）** | 关键 CDR 位点突变破坏结合，或功能读出（纤维抑制/细胞） | 突变使 Kd 下降 ≥10× | — |

**预算**：最小 ¥12k–18k，含外包 ¥25k–47k（按 `QLU_NATBIOTECH_ROADMAP_V1.md`）。

**最重要的一条**：**M2 的候选选择不要用 ECLS**。项目自己的数据已经说明它在 competent designs 里没有排序力；用它选择会把「实验失败」与「选择器失败」混在一起，浪费唯一的实验预算。可以并行保留它作为**对照方法**（这正是边界研究的价值）。

---

## 9. 必须先修的证据问题（在任何写作/投稿之前）

这三条是我这轮复核出来的、可核验的问题。前两条会直接改变稿件措辞。

### 9.1 手稿「never top-5」与冻结数据冲突（已独立重算）

- `publication/flex_boundary_manuscript_draft_v1.md:123-124` 写："The native is never ranked first in any of the 26 pools by any method, and never top-5."
- **我的重算**（用 `results/flex_c1_v1/scoring/*.json` + `results/flex_j0_v1/scoring/*.json`，按 `scripts/flex_c1_score_and_rank_v1.py:216-220` 的公式；与冻结 `c1_results.json` 的 percentile 逐位一致，max diff = 0.0）：
  - native 从未第 1：**0/26**（最好第 2）✔
  - native 进前 5：**9/26 ✖**（6zwk_D_J、6zwk_E_K、8b8i_E_M、8op0_A_B、8opi_A_B、8pii_A_B 均第 2；6i2g_A_B、6zwk_B_H 第 3；9eiu_B_A 第 5）
  - 中位排名 15.5
- **根因**：`scripts/flex_c1_score_and_rank_v1.py:311-314` 的 `top1_rate`/`top5_rate` 统计的是 `percentile == 0` 与 `percentile ≤ 5/(pool_size-1)`，而在 `:169-179` 的约定里 1 = best——所以这两个字段实际统计的是「**从不垫底**」，被误读成「从不夺冠/从不进前五」。
- **波及**：`docs/FLEX_BINDER_CAMPAIGN_V1.md:307-309` 表格的 "Top-1 0/26 / Top-5 0/26" 同源。
- **正确表述**：native **从不是第 1（最好第 2），9/26 池进前 5，从不垫底**。
- **影响评估**：主结论（门失败、输给组成基线）不变；但「never top-5」这句会削弱边界陈述的准确性，且给审稿人一个可核的证据漏洞。

### 9.2 composition_logistic 的 CI 下界高于它自己的点估计

`c1_summary.json` 记录 mean 0.54399，CI [0.545, 0.630]；手稿照抄为 "0.544 ([0.545, 0.630])"。原因不是噪声：点估计用**组件等权**（`:304`），bootstrap 用**家族聚类**（`:291-306`），两个估计量不同。家族等权均值 = **0.58438**（我重算）。稿件必须同时给两种口径，否则读者会以为 CI 排除了点估计。

### 9.3 FLEX 整条线不在 git 里

28 个 `scripts/flex_*.py` + 3 个契约测试 + 2 篇文档 untracked；`results/` 被 `.gitignore:78` 忽略。冻结产物只有 JSON 内 sha256，没有版本历史——任何人 checkout 仓库都拿不到这条线。

---

## 10. 结论

> **这个项目已经把「做不到什么」测得很清楚，但还没开始做「做到」那一步。**

- **如果目标是一个真实的、带 Kd 的无序表位结合体**：距离 = **3–6 个月 + ¥1.5–4 万 + 一次方法换代**（生成端换成 RFdiffusion/ProteinMPNN，选择端不要再依赖自研打分器）。这是可达的。
- **如果目标是「用我们的柔性感知方法做成」**：距离 = **未知**。三道关键门（判别力、排序力、前瞻效力）没有一条有已知可行路径，其中数据门是数学死结。
- **最诚实的落点**：把现有三条线做成边界研究（ECLS 主稿 + PAE 边界 + FLEX 边界），这已经是不错的科研产出；把「设计成功」留给下一代方法或外部合作。

---

## 附：本报告引用的外部文献核验状态

8 条关键引用已过 `reference_verify`：**8/8 VERIFIED**（Crossref/DOI 注册中心返回记录）。
其余引用见 `lit-review.md` 的引用清单；该文件生成时 Crossref 曾不可达，作者已披露，本报告只使用其中被核验的 8 条作为论据。

1. Liu C, Wu K, Choi H, et al. Diffusing protein binders to intrinsically disordered proteins. *Nature*. 2025. doi:10.1038/s41586-025-09248-9
2. Vázquez Torres S, Leung PJY, Venkatesh P, et al. De novo design of high-affinity binders of bioactive helical peptides. *Nature*. 2023. doi:10.1038/s41586-023-06953-1
3. Pacesa M, Nickel L, Schellhaas C, et al. One-shot design of functional protein binders with BindCraft. *Nature*. 2025. doi:10.1038/s41586-025-09429-6
4. Jiang B, Li X, Guo A, et al. RFdiffusion Exhibits Low Success Rate in De Novo Design of Functional Protein Binders for Biochemical Detection. *bioRxiv*. 2025. doi:10.1101/2025.02.07.636769
5. Islam Z, Vaikath NN, Hmila I, et al. Structural insights into the unique recognition module between α-synuclein peptide and nanobody. *Protein Sci*. 2023. doi:10.1002/pro.4875
6. Sappington I, Toul M, Lee DS, et al. Improved protein binder design using β-pairing targeted RFdiffusion. *Nat Commun*. 2026. doi:10.1038/s41467-025-67866-3
7. Bouvier-Müller A, Fourmy D, Fenyi A, et al. Aptamer binding footprints discriminate α-synuclein fibrillar polymorphs from different synucleinopathies. *Nucleic Acids Res*. 2024. doi:10.1093/nar/gkae544
8. Hmila I, Vaikath NN, Majbour NK, et al. Novel engineered nanobodies specific for N-terminal region of alpha-synuclein recognize Lewy-body pathology and inhibit in-vitro seeded aggregation and toxicity. *FEBS J*. 2022. doi:10.1111/febs.16376
