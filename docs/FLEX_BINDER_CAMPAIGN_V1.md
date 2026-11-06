# FLEX-BINDER 战役 v1 —— 无序/柔性表位结合体方向的新线

日期：2026-10-29。这是**新线，不接 PAE v31 编号**：v4–v30 的教训已提炼为错误模式表
（§1），本战役的结构就是这些教训的防线。定位：**生成外包，自家模型只做柔性表位条件下
的打分/排序**。已冻结结果（ECLS、PAE v20/v30、T2.1）一律不动、不混写。

> **⚠️ 2026-11-01 复核：本文件 §8–§9 的数字作废。**
> §8（J0a +0.124）、§9（J0b +0.141、C1 native 0/26）产生于链角色编码 bug：
> `fragment_type` 用了链序号——H 链被编码成 padding、抗原被编码成 Heavy、`mask_antigen` 为空。
> 修复后重算：**J0a +5.553（9/9 族，CI [5.313, 5.917]）；J0b +5.679（8/8 族，CI [5.336, 6.006]，
> 26/26 组件阳性，系综边际化无衰减，Δ=−0.126，2026-11-07 完成，见
> `results/flex_j0b_v2_antigen_roles/`）；C1 native 在 26/26 池中均为 Top-1，
> 领先最好设计中位 1.77 nats**。下文表格保留为过程记录，**其数字不得再被引用**；线结论与关闭
> 决定见 [`DECISION_20261101_MODEL_LINE_CLOSURE.md`](DECISION_20261101_MODEL_LINE_CLOSURE.md)。
>
> 修复后的正确读法：裁判能识别"自己的骨架/自己的序列"（自洽性），但**没有超出组成的结合信息**
> ——肽序列置换 Δ 仅 0.076 nats（对设计优势 1.77）、AAYL 亲和力 Spearman +0.024（CI 跨 0）、
> 组内 pairwise 扣掉组成只剩 +0.0098。**native 恢复通过 ≠ 结合判别建立。**

---

## 1. 错误模式 → 本次防线

每条都以仓库内自己的审计文档为据，不是事后概括。

| # | 错误模式 | 证据出处 | 本次防线 |
|---|---|---|---|
| M1 | 先跑实验、后审数据。PAE 线从 v4 跑到 v20 才补证据审计，v21 数据核查结论是"新增合格家族 = 0"——数据墙在 17 个版本之后才撞上 | `docs/PAE_EVIDENCE_AUDIT_V20.md`、`docs/PAE_NEW_FAMILY_FEASIBILITY_V21.md` | Phase A 数据先行：任何模型接触之前完成资格审计并冻结数字。首轮扫描已执行（§2） |
| M2 | 裁判未经验证就当裁判。AF2 单姿态 ipTM 对 native/shuffle 臂差仅 0.0004–0.013；PRODIGY 种子方差巨大——裁判本身对柔性肽没有区分力，之后所有"筛选收益"都建立在这个裁判上 | `INTEGRATIVE_ANALYSIS.md` §6、`docs/PAE_ANALOG_TRANSFER_V30.md` 边界节 | Gate J0：裁判的 native/shuffle 序列敏感度必须先在 held-out 上单独验证并冻结，才有资格当终点 |
| M3 | 基建 bug 污染科学结论：`VH+VL` 链分隔符使 ipTM 低估 4.5 倍、antigen 枚举掩码为空、11 残基 H3 掩码无效、`num_residues` NameError | `INTEGRATIVE_ANALYSIS.md` §6.3、`HANDOFF_V5.md` §三 | 契约测试先行：解析、掩码、链编号、坐标门控的用例在 Phase A 结束前全部进 `tests/` 并通过；带 bug 的代码路径不许产出科学数字 |
| M4 | 看了结果改协议，把确认性证据降级成 development evidence：T2.1 v2 事后改 QC、CAID2 用于选目标后降为回归集、被迫启用未接触的 CAID3 | `INTEGRATIVE_ANALYSIS.md` §4、`docs/DISORDER_SUPERVISION_PROTOCOL.md` | 预注册 + sha256 在任何候选生成/评分之前冻结（沿用 v27 先例）；凡接触过数据的选项只能写 development evidence，外部基准终身只用一次 |
| M5 | 弱标签烧训练周期：charge-hydropathy 启发式 2950 条 → AUC 0.509；143 条 DisProt → 0.500。两个周期废在标签上 | `INTEGRATIVE_ANALYSIS.md` §8 | 标签源分级表先冻结（MD/NMR 实测 > 结构代理 > AF2-seed 代理），训练前先过 NMR 校准审计；代理标签只给低权重或排序目标 |
| M6 | 版本链漂移：v4→v30 每轮从上一轮残骸出发，claim 逐轮回撤（组成驱动、无 pre-AF2 主张……） | `docs/PAE_ANALOG_TRANSFER_V30.md` 边界节、v4 纠错账本 | 新线新名；每阶段结束写"产物 + 允许表述 + 仍缺什么"三栏，claim 边界先写后测，测完不改 |
| M7 | 小样本上跑确认性检验：确认性总体 6 组件/6 家族；最终端 15 簇；n=6 的 70% 门 | v23–v27 门控链、ECLS 面板 | 预注册时先算最小可检测效应；n 低于阈值就把终点声明为描述性，不做显著性表述 |
| M8 | 生成器/裁判能力错配：BFN 生成弱于 ProteinMPNN（native recovery 0.08 vs 0.33）却承担生成+排序双重角色；复杂模型从未稳定优于 Ridge（0.839） | `INTEGRATIVE_ANALYSIS.md` §6.2、`docs/PAE_SURROGATE_REVISION_V4.md` | 生成外包（ProteinMPNN / 公开抗体 LM）；自家模型只做排序；**Ridge、组成模型、ProteinMPNN NLL 必须进主对比表**，复杂模型赢不了基线就按边界研究重新定位 |

v4 整改时写下的科研条件至今有效，本次直接作为前提："取得真正独立、跨抗原家族的数据，
并证明方法稳定优于 Ridge 等简单监督基线；否则按基准/方法边界研究重新定位。"

## 2. 数据先行结果（2026-10-29 冻结）

### 2.1 单域骨架 × 短肽（v21 明确未覆盖的类）

本地 SAbDab all-summary 快照（21,142 行，`data/sabdab_summary_all.tsv`）只读扫描：

| 统计 | 数 |
|---|---:|
| 无轻链行（nanobody/sdAb 类） | 5,010 |
| 无轻链 × 肽抗原行 | 132 |
| **对应唯一 PDB** | **57** |
| 其中 X-ray | 122 行（57 PDB 中绝大多数） |
| 无轻链 × 短蛋白抗原行 | 3,472（粗筛，未核链长） |
| 肽抗原 × 带亲和力注释行（全库） | 249 |
| 无轻链 × 肽 × 亲和力行 | 9 |

脚本：`scripts/oneoff/flex_campaign_v1/scan_sabdab_scaffolds.py`；
产物：`results/flex_campaign_v1_scaffold_scan.json`。

**边界**：这是摘要行级粗筛。未做链身份核验（ANARCII + 生物学身份复核——v21 证明
"抗体模式编号成功 ≠ 确为抗体"）、未做坐标/H3 接触/家族隔离/同源审计。57 PDB 是
**候选池上界**，不是合格数；数字只能用于"够不够启动审计"的判断，不得写进任何手稿。

对照：当前 ECLS 面板为 31 结构/15 簇。57 PDB 的原始池意味着该类审计**值得做**——
即使资格门槛砍半，也有望把独立复合物规模翻倍。

### 2.2 MD-RMSF 标签源（修 M5 的标签根因）

- ATLAS（dsimb.inserm.fr/ATLAS）：1,938 个蛋白、标准化 MD、逐残基 RMSF 可下载
  （2025 起统一为 Å）、`download_ATLAS.py` 批量脚本、CC-BY-NC 4.0、维护活跃
  （最近变更日志 2026-10-27）。备选：mdCATH。
- 用途：**替换/加权降级** v5 的 AF2-seed RMSF 合成标签（AF2 种子间 per-res std ≈ 0.09，
  是代理的代理）。验收：对本地 159 个 NMR ensemble（`data/calibration_nmr_pdb`）做
  秩相关审计，通过 `pass_physical_label` 同级门槛后才允许进入训练。

## 3. 阶段结构与停止规则

```
Phase A 数据与基建（当前）
  A1 单域骨架资格审计（复用 v21 五轴门控脚本，骨架类改为无轻链）→ 冻结合格数
  A2 ATLAS 下载 + NMR 校准审计 → 冻结标签分级表
  A3 契约测试补齐（链分隔/掩码/编号/坐标门控）→ 全绿才算 A 完成
  A4 亲和力/阴性池盘点（249 行的资格审计；IEDB/ProtaBank 另评估）
  停止规则：A1 合格 PDB < 12 或 A2 审计不过 → 本战役降级为描述性研究，不训练

Phase B 裁判验证（Gate J0，先于一切候选生成）
  在 Phase A 冻结的复合物上验证裁判对 native vs 组成 shuffle 的序列敏感度
  （ensemble 化条件：NMR/MD 多构象，不用单姿态）
  停止规则：裁判分不开 native/shuffle → 回到裁判设计，禁止进入 Phase C

Phase C 预注册排序器实验
  生成外包；自家打分 + Ridge/组成/ProteinMPNN-NLL 同台主对比
  预注册（claim 边界、终点、预算、最小可检测效应）sha256 冻结后才开始
  结果（2026-10-30）：C1 门失败——裁判未取得对组成基线与 MPNN-NLL 的家族
  一致优势（3/8 与 2/8），native 在 26 个池中均为非 Top-1；M8 条款触发，
  本线按基准/边界研究重新定位（见 §9）
```

## 4. 红线继承

- 不触碰 `publication/ECLS_SCOPE_FREEZE.yml`、PAE 冻结部署、一次性 holdout 脚本。
- 本战役的任何结论不得与已冻结结果混写或互证。
- 外部基准（CAID3 已用）之外，新基准一旦接触即冻结用途。

## 5. Phase A 执行结果（2026-10-29，数字以下列冻结产物为准）

### A1 单域骨架 × 短肽资格审计——**通过停止规则**

`scripts/flex_a1_scaffold_audit_v1.py`（复用 v26 坐标门控机械：严格残基解析、
ANARCII Chothia 编号、CDR-H3 93–102 区域提取、4.5 Å 接触度量）。

| 阶段 | 数 |
|---|---:|
| 候选 PDB（来自 2026-10-29 扫描） | 57 |
| mmCIF 下载成功 | 57 |
| ANARCII 编号序列 | 90 |
| 结构级排除 | 17（15 无确认重链——其中 14 个为 1mc\* Bence-Jones 轻链二聚体，被正确拦下；2 无合格肽链） |
| 构建组件（重链×肽组合） | 340 |
| **通过全部门组件** | **107** |
| **通过 PDB** | **36**（≥12 停止规则达标） |
| 序列去重 | 32 条互不相同的重链序列；跨 PDB 重链零重复 |
| 方法分布 | 101 X-ray / 6 EM |
| 接触与编号质量 | 最小接触距离 2.5–3.0 Å；ANARCII 分数 ≈29 |

产物：`results/flex_a1_scaffold_audit_v1/{protocol,download_receipt,manifest,summary,completion}.json`。

**边界（必须保留）**：这是坐标/身份/接触门，不是结合证据。暴露洁净度、与 ECLS
面板及训练数据的同源隔离、抗原家族拆分（A1b）均未做；36 个 PDB ≠ 36 个独立家族。
第一次运行曾产出全零假阴性（M3 类解析 bug：三字母残基名直接比对单字母集合），
被空编号池硬报错拦下后修复重跑——该 fail-closed 守卫现在是脚本的一部分。

### A2 柔性标签源审计——NMR 侧完成，ATLAS 侧进行中

- NMR 侧（完成）：`scripts/flex_a2_nmr_rmsf_v1.py`；159 个校准 PDB 解析出
  **129 个多模型 ensemble**（中位 20 模型/条目、135 残基/条目），
  逐残基 Cα RMSF 已冻结于 `results/flex_a2_nmr_rmsf_v1/rmsf.json`。
- ATLAS 侧（首轮完成，2026-10-29）：`scripts/flex_a2_atlas_nmr_audit_v1.py`。
  管道 = 官方 sequence-search 定位 ≥90% 同源 ATLAS 蛋白 → RCSB FASTA 全局比对配对
  残基 → tracks 端点逐残基 RMSF（3×100 ns 重复均值，Å）→ 蛋白内 Spearman。
  **配对数不足（2/129），按预注册处理方式记为"配对不足"，不得引用为验证**：
  2 个匹配对（2gpq→5gw6_A 98.4%、2m5b→2ims_A 100%）方向一致——Spearman 0.585 与
  0.727、NMR top-20% 召回 0.62/0.59，均高于 0.30 门，但 n=2 只允许作为先导信号。
  自动产物 `summary.json` 的 `pass_physical_label` 字段必须连带本边界一起读。
  下一步（按序）：改道 Chameleon（16 蛋白）与 DPF（100 蛋白）数据集扩配对；以及
  直接在 ATLAS 全库 1,938 蛋白上做 MD-RMSF vs AF2 pLDDT 的大样本一致性审计
  （tracks 端点同文件自带 plddtTrack，无需新管道）。运行事实：96/129 时 ATLAS
  服务器曾无响应（WinError 10060），恢复后断点续跑完成；搜索/FASTA/track 三级
  缓存齐备，重跑零重复下载。

### A3 契约测试——完成

- 新补 `tests/test_flex_a1_scaffold_audit_contract.py`：6 条测试全绿。
  覆盖两类曾静默失败、耗费数月的契约：(a) AF2 multimer 链分隔符（当年 VH+VL 连写
  使 ipTM 低估 4.5 倍，原无任何回归测试）；(b) 非标准残基计数的单/三字母转换
  （A1 假阴性事故的同款 bug 模式）。
- 盘点确认已在位：antigen 枚举掩码、H3 掩码、ipTM/ptM 替换
  （`test_bfn_loader_antigen_roles.py`、`test_scientific_contracts.py`）。
- 已知不可覆盖项：`num_residues` NameError 属 WSL venv 内 alphafold-colabfold 的
  外部 patch，契约测试无法触及，记为环境风险（重装 alphafold 后必须重新打 patch）。

### A4 亲和力注释盘点——完成

`scripts/flex_a4_affinity_inventory_v1.py`：

| 统计 | 数 |
|---|---:|
| 肽抗原 × 带亲和力注释行 | 249 |
| 唯一 PDB | 138 |
| 配对 H/L 行 | 248 |
| 有明确方法注释 | 69（SPR 51、ITC 9、Other 9；其余 180 Unknown） |
| 与 A1 合格集交集 | 1 个 PDB |

**边界**：SAbDab 摘要的亲和力是未核验元数据，不得直接引用数值；只作为"哪些复合物
存在一手结合测量"的线索池，使用前必须回读原始文献。

## 6. 停止规则核对（2026-10-29，A1b 后更新）

- A1：合格 PDB 36 ≥ 12 → **继续**，进入 A1b（已完成，见 §7）。
- A1b：独立单元逐级收紧后 **14 PDB / 9 抗原家族**（§7 级联表），仍 ≥12 但贴近下限
  → 开发队列够用（构建与校准裁判）；**确认性终点不可只靠这 9 个新家族**（M7），
  Phase B/C 预注册前必须扩展宇宙（monobody/FN3 类、TCR 管道扩家族、回顾性家族）
  或把终点声明为描述性。
- A2：配对 2/129，不足 → **按预案改道**。腿 1（Chameleon/DPF 扩配对）已完成：
  **0 对**——129 个 NMR 序列在两个数据集里均无 ≥90% 匹配，公共数据上"MD vs NMR"
  配对路线的天花板即 ATLAS 的 2 对（Spearman 0.585/0.727，仅先导）。
  腿 2（ATLAS 全库 MD-RMSF vs pLDDT）已完成并经方向修正（v1→v2，见下）：
  **1,938/1,938 蛋白全部审计，中位每蛋白 Spearman(原始 MD RMSF, −原始 pLDDT) =
  0.730**（IQR 0.615–0.808；10–90 分位 0.468–0.859），96.9% 蛋白 ≥0.30、87.5%
  ≥0.50。**AF2 置信度是 MD 柔性的强排序预测子**——这同时说明：(a) pLDDT 族信号
  真有柔性信息，v5 的 rank-only 标签约定方向正确；(b) 换 MD 标签的排序信息增量
  可能有限，无序头对 metapredict 的差距更可能出在校准/架构而非标签信息量。
  偏差披露：v1 误用展示层归一化值（pLDDT 轨道按站点约定反转展示），得到物理上
  不可能的中位 −0.71；v2 用 featureId 原始值重算（rank 保持性使 RMSF 侧等价，
  反转只影响 pLDDT 侧），v1 产物按原样保留。边界：这是折叠蛋白为主的库内排序
  一致性，不是无序金标准验证，不涉及绝对值校准、结合或设计。

## 7. Phase A1b 隔离与家族拆分（2026-10-29 冻结）

脚本：`scripts/flex_a1b_isolation_families_v1.py`（v1，含一次误设）与
`scripts/flex_a1b_isolation_families_v2.py`（修正解读）。产物：
`results/flex_a1b_isolation_families_v1|v2/`。

### 级联表（36 → 14）

| 关卡 | 剩余 PDB | 说明 |
|---|---:|---|
| A1 坐标/身份/接触门 | 36 | §5 |
| 同 PDB 排除（与 ECLS 线宇宙重叠） | 30 | 7a48、8dtu、8jjv、8jly、9egn、9nk9 直接是 ECLS 线看过的结构，非独立数据 |
| 重链克隆级撞车（VH 同一性 ≥0.90） | （并集） | 9eg6、9nk9 等 4 例，多为同 PDB 或姊妹部署 |
| × 暴露洁净（至少一个组件的肽只接触确认重链） | **14** | 2x6m、5hm1、5ivn、6i2g、6j7w、6s0y、6zwk、8b8i、8op0、8pii、8opi、8skj、9eiu、9lds |

### 家族拆分（30 个隔离干净 PDB 上）

- **18 个肽抗原家族**（≥50% 同一性/80% 覆盖合并；最大家族 13 序列）；其中
  **9 个家族完全无 ECLS 抗原重叠（13 个 PDB）**，9 个家族扩展 ECLS 已知抗原
  （17 个 PDB——可用于开发，不算新家族）。
- **27 个重链谱系**（克隆级 ≥0.90 合并）。
- 最严苛 14-PDB 开发队列跨 **9 个抗原家族**。

### 偏差披露（M4 纪律）

1. v1 误把 v5 无序标签的 30%/80% 同源门用在 Ig 域上——任何两个 VH 靠框架区保守性
   就过 30%（被标记重链的同一性中位数 0.796），单连锁把 56 条重链连成一个"家族"。
   v2 的域感知阈值（重链克隆级 0.90、肽家族 0.50/0.80、同 PDB 硬排除）是在查看
   v1 冻结同一性分布**之后**、重算之前声明的；v1 产物按原样保留作误设记录。
2. 身份复核汇总打印有一处真值性 bug（全 False 的旗标字典被当真值导致 36 个全列）；
   冻结数据本身干净：**0 个内容旗标**（无 TCR/MHC/Bence-Jones/非 Ig），32/36 标题
   直接确认抗体，其余 4 个标题过简但 ANARCII+文献上下文确认（如 1mvf 为抗 MazE VHH）。

### 队列任务对口性（标题证据）

2x6m = α-synuclein C 端肽 VHH；8op0 = PHF6 Tau 肽 VHH；9lds = SARS-CoV-2 spike
无序柄螺旋 nanobody——单域 × 柔性表位正是本战役的目标类。

### Phase B 的输入（从 A1b 推出）

- 开发/校准队列：14 PDB（双干净）；扩到隔离干净 30 PDB（暴露不洁者仅用于
  非暴露敏感的校准）。
- 确认性预算：9 个全新抗原家族不足以支撑独立确认终点（对照 ECLS 15 簇、TCR
  类似物 6 评估家族的先例），扩宇宙清单见 §6。

## 8. Phase B，Gate J0a（2026-10-29 冻结，门已按预注册规则执行）〔⚠️ 本节数字已作废，见文首横幅〕

预注册：`results/flex_j0_v1/preregistration.json`（打分前冻结，含功效陈述）；
队列：`results/flex_j0_v1/cohort.json`；契约测试：`tests/test_flex_j0_contracts.py`
（打乱组成保持、RNG 可复现、组件 PDB 链与区域、上下文 batch 语义，4 条全绿）；
打分：`results/flex_j0_v1/scoring/`、`j0a_results.json`、`j0a_summary.json`。

### 结果

| 统计 | 值 |
|---|---:|
| 组件 | 28（14 PDB） |
| 抗原家族 | 9 |
| 家族阳性（advantage > 0） | 7/9（77.8%） |
| 组件级阳性 | 25/28 |
| 家族均值 advantage | **+0.124** |
| 家族聚类 bootstrap 95% CI | **[0.051, 0.189]，排零** |
| 预注册门（≥70% 家族阳性 且 CI 排零） | **通过** |
| checkpoint | CHECKPOINT_TRUTH s2032 best.pt（sha256 记录于 summary） |
| NLL 字段 | `seq`（checkpoint 无 mismatch 权重，`factual_nll_per_sample` 缺席，单样本下二者等价；披露于 summary） |
| 种子 | 3 个打分种子，未坍缩（非逐位一致） |

### 边界（按预注册逐条保留）

1. **J0a 是单姿态门**：T1 已证明单姿态高估信号（约 0.20 ECLS）。J0a 通过只解锁
   "裁判可用于开发校准"；进 Phase C 需 J0b（ensemble 边际化）同门通过。
2. **纯符号检验单独看不过 α=0.05**：7/9 家族阳性的单侧 p ≈ 0.090，而预注册功效
   陈述早已写明本门只能检出 ≥8/9 家族阳性的强裁判（临界数 8，实现 α=0.0195，
   80% 功效对应 p≥0.91）。门通过靠的是预注册的复合规则（70% 阈值 + CI 排零），
   不是家族计数显著性——这正是"开发门，非确认性终点"的活例子。
3. 这是 ECLS 信号在**同源隔离的新骨架类**（单域 × 柔性肽，A1b 级联后）上的
   开发级复制，不是对 ECLS 冻结结果的确认，也不是任何结合或设计有效性声明。
4. 打乱为组成匹配多重集排列；个别 shuffle 可能与原生重合（短区域自然现象，
   预注册未排除，原样计入零差质量）。

### J0b（已执行，2026-10-30 冻结）——**门通过，Phase C 准入**

构建：`results/flex_j0b_v1/build_summary.json`（T1 原版设置逐值复用，链重映射
H/P；有效规则 = ≥2 接受构象 且 mean pairwise RMSD ≥0.25）；打分与判定：
`results/flex_j0b_v1/{scoring/,j0b_results.json,j0b_summary.json,completion.json}`；
契约测试：`tests/test_flex_j0b_contracts.py`（3 条全绿，运行前抓到实例命名 bug）。

| 统计 | 值 |
|---|---:|
| 有效 ensemble 组件 | 26/28（5ivn_A_B、6zwk_F_L 构象散布 <0.25，按预注册剔除并披露，无单姿态回退） |
| 家族阳性 | **7/8（87.5%）**，组件级 24/26 |
| 家族均值 advantage | **+0.141** |
| 家族聚类 bootstrap 95% CI | **[0.069, 0.204]，排零** |
| 预注册门（≥70% 且 CI 排零） | **通过** |
| 两种边际化顺序 | 先边际后对比 +0.141 / 先对比后边际 +0.138（一致） |

**关键观察——T1 衰减预言未复现**：ensemble 边际化后的均值（0.141）与单姿态
J0a（0.124）相比无衰减（差 −0.017，方向反而略正）。T1 在旧配对 H/L 面板上测得
的 ~0.20 衰减在本队列（单域骨架、短肽、~0.3 Å 受限散布）上不成立。可能解释：
散布幅度小（多数构象 0.25–0.4 Å）、队列与裁判定义不同；本结果只说明"该裁判对
本队列的受限 ensemble 不敏感"，不推翻 T1 的原始观察。

**实现披露**：stripped 上下文按构造与构象无关且与 J0a 逐位一致，打分复用 J0a
缓存（复杂上下文全新计算）；首个组件做了 fresh 重算等价验证（max_abs_diff = 0.0，
bit 级一致），验证记录在 summary。中途一次崩溃（初始化 bug）发生在任何终点计算
之前，缓存续跑完成，无数据损失。

**边界**：J0b 仍是开发门（26 组件/8 家族，符号检验单独 p≈0.035 但家族数少）。
Phase C 准入指"裁判可用于开发排序"，任何确认性主张仍需 §6 的宇宙扩展。
唯一阴性家族（seq_1，均值 −0.052）在 J0a/J0b 两个裁判下方向一致——同一抗原
家族在两个条件下都被裁判读负，值得在 Phase C 里单独看是什么抗原。

## 9. Phase C1（2026-10-30 冻结）——**门失败，M8 重新定位条款触发**〔⚠️ 本节为 bug 修复前读数，已作废；修复后读法见文首横幅〕

预注册：`results/flex_c1_v1/preregistration.json`（生成前冻结）；候选：
`generation/`（1,300 个 ProteinMPNN 设计、789 不同、0 个等于 native）；
打分与判定：`results/flex_c1_v1/{scoring/,c1_results.json,c1_summary.json}`。
任务：26 个组件（8 家族）上，在 101 候选池（native + 50 MPNN 设计 + 50 组成
打乱）中定位 deposited native。

> ⚠️ 下表与本节判定均为链角色编码 bug 修复前的读数（native 从未排到第 1 是 bug 造成的），
> 仅保留作过程记录，**不得引用**；修复后 native 为 26/26 池 Top-1。

### 四方法主表（native 平均百分位，1.0 = native 全池最优）

| 方法 | 均值 | 家族 bootstrap 95% CI | Top-1 | Top-5 |
|---|---:|---|---:|---:|
| flex_ecls_marginal（自家裁判） | **0.767** | [0.676, 0.809] | 0/26 | 0/26 |
| proteinmpnn_nll | 0.637 | [0.587, 0.668] | 0/26 | 0/26 |
| composition_logistic | 0.544 | [0.545, 0.630] | 0/26 | 0/26 |

### 门判定：**fail**（按预注册复合规则；对两处方向 bug 稳健）

- 原方向读法：赢组成基线 3/8、赢 MPNN-NLL 2/8（要求 ≥70%）→ 双条款失败。
- **方向修正披露（v2，`results/flex_c1_v1/direction_correction.json`）**：实现的
  percentile 是"池中比 native 差的比例"（越高 = native 越好），冻结门条款却按
  "配对差 < 0 算赢"打分——方向相反；另有 random 条款 "CI upper < 0.5" 的同类
  方向笔误（按定义应为 CI lower > 0.5，实测 0.676，通过）。按修正方向重算：
  **赢 MPNN-NLL 6/8（75%，条款通过）；赢组成基线 5/8（<70%，条款仍失败）**。
  两种读法下门均为 **fail**——判定稳健，但失败的具体条款不同：修正后真正的
  失败点是**组成基线**，而非 MPNN 自似然。

### 结果的正确读法

1. **边界精确化（`boundary_decomposition.json`，按竞争对手类别拆开）**：
   native 在 50 个打乱中的裁判百分位家族均值最高达 1.00/0.99/0.97
   （seq_7/seq_6/seq_0），但在 50 个 MPNN 设计中两个家族跌到 0.00（seq_4=
   8skj V5 肽 nanobody，native 在设计池中垫底）和 0.33（seq_0）。
   **裁判识别的是"非 native 的组成"，不是"众多近 native 竞争者中的 native"**。
2. **修正方向后的真实格局**：裁判以 6/8 家族胜过 ProteinMPNN 自条件似然
   （其设计池、其评分器），但 5/8 不及组成基线——**连 20 维氨基酸组成都要
   赢不了，"超出组成的信息"主张不成立**，这是 M8 失败的准确位置。
3. **信息来源**：裁判与 MPNN-NLL 的逐池 Spearman 家族均值大多为正
   （seq_0 0.77、seq_4 0.69、seq_7 0.66），seq_3 为负（−0.16）——裁判大部分
   排名信息与 MPNN 共享，独立增量集中在个别家族。
4. **家族身份（`identity_review.json`）**：seq_4 = 8skj（V5 肽 nanobody，判读
   最反常：J0 名义通过但设计池垫底）；seq_1 = 2x6m（α-synuclein C 端肽）+
   9lds（SARS-CoV-2 spike 无序柄）——两个生物学上无关的抗原被 ≥50% 肽同一性
   规则并成一个"家族"，这本身暴露了短肽家族合并规则的粗糙性。
5. **M8 条款生效**：Phase C 开发主张停止，本线按"基准/边界研究"重新定位。
   修正后的边界陈述比原版更强也更有用：ECLS 类似然差携带真实但组成可替代的
   排序信息，对近 native 竞争者无分辨力。

### 过程披露

四次启动失败均为基建层（MPNN 参数名/链分隔/fasta 格式、glob 污染、design_mask
字段缺席、增量缓存误用 freeze），无一次触碰科学输入；每处修复与披露在案。
native 与 shuffle 的分数逐位复用 J0a/J0b 冻结缓存，只有 1,300 个设计是新增计算。

### Phase C 后续

按 M8：不再做"裁判优于基线"的开发主张。可选后续（另行预注册）：
(a) 把 26 组件 × 4 方法的完整排名表作为边界研究手稿的核心结果；
(b) 查 seq_4（裁判最强）与 seq_1（J0 阴性但 C1 反转）的抗原身份，理解家族结构；
(c) 任何新的排名能力主张必须换任务定义（例如 binder-vs-nonbinder 而非
native-vs-designs）并先扩宇宙。

### 10. 产出与后续（2026-10-30）

- **边界研究手稿草案已立**：`publication/flex_boundary_manuscript_draft_v1.md`
  （本地作者评审稿；无作者声明、无投稿主张；相关手稿重叠披露在文内——ECLS
  与 PAE 两线）。数据核全部来自冻结产物。
- **Phase D（binder-vs-nonbinder）预注册草案已立**：
  `docs/FLEX_PHASE_D_BINDER_TASK_PREREG_DRAFT_V1.md`——**未冻结**，被数据
  可行性门阻塞（≥24 家族、组成匹配的非结合体、实验阴性记录）；功效表已冻结
  （`results/flex_phase_d_v1/power_table.json`：24 家族检出 75% 家族阳性功效
  0.77，30 家族 0.89；当前 8–9 家族同类功效 ~0.24）。
- **Phase D 终结（2026-11-02，决定 B）**：严格路线已测尽——A' 全库扩池证伪
  （709,890 独立序列、平均每组成类 1.056 个，产率 0.96%→1.63%，1/200；
  results/flex_phase_d_v1/{negative_pool_summary,pilot_v3b_summary,
  termination}.json）。终止产物含重启条件（需改变组成碰撞数学的新实验阴性
  数据集，且重启须先冻结新预注册）。**ECLS 线以边界研究手稿收尾**
  （publication/flex_boundary_manuscript_draft_v1.md）。

