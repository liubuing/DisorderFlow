# CSN 战役 v1 —— 神经蛋白无序表位的构象态识别（计算归因线）

日期：2026-11-08。**新线，不接 FLEX/PAE/ESEL 编号。** 定位一句话：把「免疫识别能否
读取神经蛋白（Aβ / tau / α-syn / Syt1）无序表位的构象态」变成可预注册、可证伪、
可阴性终结的问题；计算做**归因**（哪个构象态被读），实验裁决仍归 ESEL 壳
（呈现条件 × 富集读出）。

## 0. 授权链

- 用户指令（2026-11-08）：执行 2026-11-06/07 提案讨论中确定的 CSN 方向，
  负结果不回避、如实冻结；
- 继承 [`DECISION_20261101_MODEL_LINE_CLOSURE.md`](DECISION_20261101_MODEL_LINE_CLOSURE.md)
  全部约束：不自建亲和力判别器、设计选择不用模型分、AF2/PAE 不作门禁；
- 继承 [`ESEL_CAMPAIGN_V1.md`](ESEL_CAMPAIGN_V1.md) 的实验壳定位：ESEL RFQ 短名单
  中的 α-syn C 端（DYEPEA）与 Syt1 行即本战役的神经行；CSN 不发订单、不碰湿实验；
- 与已冻结结果（ECLS 主稿、PAE v30、J0a/J0b v2 门）的关系：**只读引用，不混写、
  不互证、不重跑**。

## 1. 科学问题（三条，均可阴性终结）

- **H1（富集，C1）**：IEDB 有先例的神经 IDP 表位窗口，是否富集于自由态系综中
  具有瞬态结构偏置的区段（局部 RMSF 谷 / MoRF 样段 / chameleon 核心）？
- **H2（归因，C0/C2 的延长线）**：已知结合构象能否从构象源中被似然召回？
  前置问题（C2）：J0b v2「系综不衰减」与 T1「MD 系综衰减 ~0.20 ECLS」的张力，
  首先要归结为系综幅度不匹配还是真实的机制差异。
- **H3（实验裁决，挂 ESEL）**：同一 CDR-H3 文库分布对同一表位肽的不同呈现态
  （单体 / 寡聚 / 纤维）是否产生可区分的富集谱。CSN 只负责给 ESEL 的呈现矩阵
  **加列建议**，不执行任何湿实验。

## 2. 错误模式 → 本战役防线（全部继承 M1–M8，新增两条）

| # | 错误模式 | 防线 |
|---|---|---|
| M1–M8 | （同 ESEL 战役表，全文适用） | 预注册先行、契约测试先行、write-once、claim 先写后测 |
| N1 | 跨战役引用造成"互证" | 每个产物头部带 `classification`；引用冻结数字必须写"只读引用" |
| N2 | 把描述性构象观察说成选择机制 | C0 的 verdict 只允许三种封闭取值（见 C0 预注册），禁止叙事性升级 |

## 3. 阶段结构与停止规则

```
C0 跨结合者构象共享（纯离线，本地坐标簿记，无模型无网络）
  问题：独立沉积的结合者（不同 PDB 条目）对同一条无序表位肽是否给出
        相同主链构象（构象选择信号）还是不同构象（诱导契合信号）？
  宇宙：flex_j0 28 组件 + sabdab2 池 1,146 CIF 的机械子串扫描扩充
  停止规则：0 个有效跨条目族 → 「构象共享坐标饥饿」，C0 终结；
            1–2 个族 → 完成，但 verdict 分类为「单/少数族描述性」，
                        明示不可推广；
            底板对照失效（同条目同结合者对 RMSD > 1.0 Å）→ 同条目多样性
                        解读整体降级为「测量底板未验证」。
C2 系综幅度诊断（纯离线，只读 J0b conformer PDB）
  问题：J0b v2 生成 conformer 的肽段运动幅度是否达到 T1 MD 系综参考
        （平均成对肽 RMSD 0.63 Å，7 开发族）？
  红线：不重跑、不修正、不重解释已冻结的 J0a/J0b v2 门数字。
  判读（预注册封闭规则）：中位幅度 < 0.3 Å → 「边缘化平凡解释」；
        ≥ 0.6 Å → 「幅度对等成立，张力真实，需 judge/队列层深查」。
C1 瞬态结构富集图谱（网络标签源：ATLAS RMSF + IEDB metadata 控制窗口）
  阻塞：C0/C2 完成后才设计；IEDB 访问保持 metadata 级；E0b 阶梯复用。
C3 → 实验层（不属本战役）：向 ESEL RFQ 提交呈现加列建议（单体 vs PFF、
  α-syn C 端截短 1-121 对照），由用户与伙伴决定是否采纳。
```

## 4. 输入宇宙（全部只读，或本地冻结文件）

- 队列：`results/flex_j0_v1/preregistration.json` `components[]`（28 组件 / 14 PDB）；
- 坐标（解析顺序，fail-closed）：① `data/sabdab2_abag_project_pool_v2/cif/
  pdb_0000{pid}_{heavy}_+.cif`；② `data/multiscaffold_confirmatory_v1/cif/
  pdb_0000{pid}-{heavy}.cif`；③ `data/candidate_interface_external_calibration_v1/
  structures_v3b/cif/{PID}.cif`（全条目，原始链名）；
- conformer：`results/flex_j0b_v1/conformer_component_pdbs/*.pdb`（26 组件 × 4 种子）；
- 参考：T1 平均成对肽 RMSD 0.63 Å（`INTEGRATIVE_ANALYSIS.md` §2.2，只读引用）；
- 已知缺口（如实记录，不补抓）：5ivn、6zwk×4、8skj_G_H 本地无坐标。

## 5. 红线继承

- 不触碰 `publication/ECLS_SCOPE_FREEZE.yml`、`publication/candidate_interface_pae_deployment_v1.json`、
  一次性 holdout 评估脚本；
- 本战役任何结论不得与已冻结结果混写或互证；
- 冻结产物目录 write-once：修正开 `*_v2` 目录 + 披露文件；
- 任何模型分数不得进入 C0/C1 的任何排序或选择步骤（N1 纪律；
  C0/C2 是坐标簿记，无模型推理）。

## 6. 执行记录

### 基建发现（M3 级，2026-11-08，先于任何 C0/C2 结果）

**legacy kabsch 转置 bug**：`scripts/flex_a2_nmr_rmsf_v1.py` 的 `kabsch` 把正确旋转的
**转置**以行约定应用（`mc @ rotation`），完美刚体运动的往返 RMSD 为 5–9 Å 而非 0。
契约测试 `test_corrected_kabsch_beats_legacy_on_random_rigid_motions` 把该发现钉死为
canary。修正实现为 `scripts/csn_c0_conformer_sharing_v1.kabsch_row`（行约定，
`u diag(1,1,d) vt`），随机刚体运动全部机器精度归零。

影响面（逐一验伤后的修正结论，2026-11-08 更新）：

| 产物 | 状态 |
|---|---|
| `results/flex_a2_nmr_rmsf_v1/rmsf.json`（NMR 系综 RMSF 校准参考） | **受影响，已 v2 重算**（见下） |
| `results/flex_a2_atlas_nmr_audit_v1`（Spearman 审计，NMR 侧用 v1） | **已离线重审**（见下）；ATLAS 缓存复用，无网络 |
| IDP 系综姿态多样性审计（`audit_idp_ensemble_v3_pose_diversity` 等 4 脚本） | **不受影响**——逐一读码确认它们各自本地定义的 kabsch 均为正确的行约定（`u @ vt` + 反射处理），未继承 bug。此前基于 grep 的"需重算"初判**撤回** |
| T1 参考 0.63 Å（`INTEGRATIVE_ANALYSIS.md` §2.2，7 开发族） | 生成脚本的对齐约定无法从冻结记录核实——C2 协议已按"双口径桥接"处理并披露 |

按仓库纪律，冻结产物不做就地修正；重算开 `*_v2` 目录 + 披露。

### 修复执行（2026-11-08）

- **`flex_a2_nmr_rmsf_v2`**（`scripts/flex_a2_nmr_rmsf_v2.py`，自带修正 kabsch）：
  129 条目与 v1 一致；NMR RMSF 均值 3.195 → **2.244 Å**（错位噪声占 ~42%），
  条目级 v1/v2 比值中位 1.115、p90 1.76、最差 5.2×（1mwn）；**残基级 Pearson
  中位 0.981**——排序信息基本幸存，绝对校准失效。披露：
  `results/flex_a2_nmr_rmsf_v2/v1_disclosure.json`；
- **`flex_a2_atlas_nmr_audit_v2`**（`scripts/flex_a2_atlas_nmr_audit_v2.py`，
  完全离线，v1 缓存只读复用，缓存缺失即 fail-closed）：2 个受审蛋白全部重审，
  **rho 中位 0.656 → 0.725**（2gpq +0.152，2m5b −0.014），v1 的
  `pass_physical_label` 判定在修正标签下**幸存且更强**。披露：
  `results/flex_a2_atlas_nmr_audit_v2/summary.json`；
- IDP 线无重算需求（见影响面表修正）。

### C0（2026-11-08，v1 → v2）

- **v1**：`results/csn_c0_v1/`——预注册冻结（d042d0d5…）、契约测试全绿后执行；
  `components.json` 与 `sweep.json` 成功冻结（提取环节无 bug），随后在
  `pair_record` 崩溃（二面角一致率误用父坐标系索引组件本地残基），
  pairs/summary/completion 未产出；
- **v2**：按 write-once 纪律开 `results/csn_c0_v2/` + 预注册 `revision_note`
  披露（93fd5ae9…）；科学内容不变，仅修 bug 与换目录；v1 部分产物原样保留；
- 契约测试：`tests/test_csn_c0_contracts.py` 13/13 全绿（跑前 8 项，冻结后 13 项）；
- 脚本：`scripts/csn_c0_conformer_sharing_v1.py`；
- **结果**（`results/csn_c0_v2/summary.json`）：
  - 队列 22/28 组件有效（`5ivn_A_B`、`6zwk×4`、`8skj_G_H` 本地无坐标，如实记录）；
  - 池扫描 1,146 文件全解析、14 命中、0 个新肽级扩充组件（肽级命中全部被队列覆盖）、
    0 解析错误；
  - 底板对照通过：9 个 NCS 拷贝对均值 0.211 Å（≤0.5 预注册线）；
  - **主检验（跨条目族，tau PHF6* GSVQIVYKPK，8op0/8opi/8pii 三个独立沉积的
    VHH Z70 突变体）：成对 RMSD 0.098 / 0.137 / 0.157 Å，全部
    `same_bound_conformer`，二面角类别一致率 0.75–1.0——三个独立结合者给出
    同一主链构象，一致性达到晶体学 NCS 拷贝量级**；
  - 8b8i（Syt1，同晶体 8 拷贝）：G_O 拷贝对其余拷贝 0.64–0.74 Å、二面角一致率
    0.45——同结合者同晶体内的微多态，descriptive 记录；
  - α-syn C 端 DYEPEA（2x6m）：单复合物，无跨结合者证据可用（如实记录）；
  - **全局判定：`conformer_sharing`，分类 `single_or_few_family_descriptive`
    （1 个跨条目族，明示不可推广）**。

### C2（2026-11-08）

- 协议先冻结（a6ce1fb5…），双口径（修正 complex-frame 主口径 / legacy 披露口径）；
- 脚本：`scripts/csn_c2_ensemble_magnitude_v1.py`；契约测试并入
  `tests/test_csn_c0_contracts.py`；
- **结果**（`results/csn_c2_v1/summary.json`）：26 组件（缺 `5ivn_A_B`、`6zwk_F_L`
  conformer，如实记录），26/26 框架验证通过（跨种子重链 CA RMSD ≤0.1 Å）；
  修正口径中位成对肽 RMSD **0.445 Å**（legacy 口径 0.564 Å，被错位噪声抬高，
  方向与披露一致）；tier 分布 2 near_rigid / 22 modest / 2 t1_like；
- **判定：`intermediate_requires_deeper_resolution`**——J0b conformer **不是**
  近刚性（0.445 Å ≈ T1 参考 0.63 Å 的 70%），"系综边缘化是平凡的"这一解释被
  **否证**；J0b v2"无衰减"与 T1"衰减 ~0.20"的张力不能归结为系综幅度不匹配，
  必须在 judge（BFN vs MPNN）/队列（26 组件 vs 7 开发族）/系综构造（生成式 vs
  MD 约束采样）层面继续分解；
- 红线遵守：J0a/J0b v2 冻结门数字只读引用，未重跑未修正。

### C1 数据与预注册（2026-11-08）

- **数据采集**（`scripts/csn_c1_data_v1.py` → `results/csn_c1_v1/`）：
  E0b 8 行只读复用锚定；6 个未解析行走 CSN 本地缓存阶梯——RCSB entity 直接
  给出 tau 候选 P10636 但包含验证**正确拒绝**（deposited 肽 GSVQIVYKPK 非字
  canonical tau 子串：canonical 同源段为 VQIVYKPVL，PDB 肽末位 K）；
  UniProt 肽搜索 API 提交正常（202）但轮询 ≥90 s 恒空（已知阳性对照肽
  KEDAFSKLKEKFMNEL⊂P21579 亦空）——**服务降级记为环境事实**，非零证据；
  6i2g/8skj/6zwk 排除出宇宙；
- **首次运行的坑**：run-1 用了我手写的同步肽搜索（错误参数、无轮询），产物
  已按纪律移入 `results/csn_c1_v1/superseded_buggy_peptidesearch/` + 披露；
  run-2 镜像 E0b 异步轮询后重冻结；
- **关键标签事实**：α-syn DYEPEA 窗口被 **2kkw**（游离态 NMR 系综）直接覆盖
  ——头号神经行获主标签；ATLAS 仅覆盖 gp160/spike（2/8）；主层 = 3 亲本，
  其余 6 亲本 AFDB 降级层（仅描述性）；
- **预注册冻结**（`scripts/csn_c1_prereg_v1.py` →
  `results/csn_c1_v1/preregistration.json`，e8b19110…）：含 control 定义修正
  （窗口级 IEDB 先例在 metadata-only 下不可得——草案缺陷已披露修正）、
  anchored_with_mismatch 规则、标签分层、封闭判读、停止规则；
- 契约测试 13/13 全绿。

### C1 执行（2026-11-08 晚，v1–v6 迭代）

- **肽匹配服务问题真正解决**（回应"回避"批评）：UniProt asyncrest 轮询失效
  与 PIR 503 后，改为**本地 Swiss-Prot+异构体流式精确扫描**
  （`scripts/csn_peptidematch_local_v1.py`，~57 万条）；**三个阳性对照全过**
  （Syt1 肽→P21579、DYEPEA→P37840、canonical tau 段→P10636），管线自证可用；
- **四个肽定性定论**（`results/csn_c1_v1/peptidematch/`）：
  - tau deposited `GSVQIVYKPK`：**全库验证零命中**——与数据库差 1 个替换
    （canonical 同源段 GSVQIVYKPV）从"API 失败"升格为"验证事实"；
  - PK tag `GKPIPNPLLGLDST` → P11207/P11208（PIV5）→ 8skj 亲本解决；
  - `TQAQEY` → Q08346 → 6zwk 亲本解决；
  - `SRLEEELRRRLTE` 验证零 → 6i2g 保守排除。C1 宇宙 9 → 11 亲本；
- **执行迭代**（`csn_c1_enrich_v1..v6`，全部披露于各目录
  SUPERSEDED_DISCLOSURE / protocol revision_note）：v1 CIF/PDB 解析器、
  v2/v3 未绑定变量、v4 AFDB v4 URL 404 + 沉积链暴露度无对照窗、v5 nmr_v2
  元组未转 map + RCSB GET + 大文件超时——**全部在读取任何标签结论之前修复**；
  v6（protocol 3113af4a…）跑通；
- **v6 结果**（`results/csn_c1_enrich_v6/`）：6/11 亲本评分
  （1 主层 + 5 AFDB 降级层；5 个未评分各有记录原因：AFDB 404×3、
  对照不足×2）；
- **科学结果（诚实版）**：
  - 主层（α-syn，2kkw NMR RMSF）：case DYEPEA 21.81 Å vs 匹配对照
    8.55–10.01 Å，**配对差 +12.37——反方向**（case 更柔性）；
    附标签局限：末端尾巴的全局摆动主导 RMSF，"局部柔性"解释受限；
  - AFDB 降级层（pLDDT，高=更有序）：Syt1 −0.262、tau −0.153、
    Q6NZI2 −0.009、Q08346 −0.007、G9GAG7 +0.038——**4/5 反方向**
    （case 窗口比匹配对照更无序）；
  - 封闭判读：主层 n=1 < 3 → `primary_stratum_underpowered`（冻结停止规则
    生效，CI 终点不可发）；描述层两方向证据一致指向——
  - **H1（表位=瞬态结构岛）在现有数据下不成立；反向图景成立：抗体先例
    窗口落在亲本最无序/最柔性的区段**。这与免疫学常识（可及性=免疫原性）
    和临床记录（prasinezumab 靶 C 尾）一致，并与 C0 呼应：构象选择发生在
    结合时刻（跨结合者共享结合构象），而非表位固有结构；
- **叙事更新（CSN v2 的方向）**：问题从"哪些窗口被读"变为"**被读的最无序
    窗口如何在结合时刻选择构象**"——C0（共享结合构象）+ C1（最无序窗口）
    + ESEL（态×呈现富集谱）构成完整闭环。

### 下一步（2026-11-08 深夜更新）

1. **C1 v7（可选）**：清 3 个 RCSB 搜索坏缓存 + AFDB 404 亲本换 pLDDT 来源，
   补齐降级层行数；标签 v2 用二面角稳定性替代 RMSF（消除末端摆动混淆）；
2. **C3 三列已并入 ESEL RFQ 附录**（§9，用户批准）；RFQ 发送在用户手中；
3. T1 vs J0b v2 张力分解、ECLS/PAE 稿件流程照旧。
