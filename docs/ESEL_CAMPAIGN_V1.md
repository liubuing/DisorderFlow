# ESEL 战役 v1 —— 表位指导的结合体选择线（实验裁决，计算执教）

日期：2026-11-04。**新线，不接 FLEX/PAE 编号。** 定位一句话：判别器从模型搬到实验，
计算只做三件有证据支持的事——**选表位、设计文库分布、事后读数**。已冻结结果
（ECLS、PAE 部署合同、FLEX J0/C1 缓存）一律不动、不混写、不互证。

## 0. 授权链

本战役是 [`DECISION_20261101_MODEL_LINE_CLOSURE.md`](DECISION_20261101_MODEL_LINE_CLOSURE.md)
第 3/4/5 条的执行载体：

- 第 3 条（逐句继承）：**设计候选的选择只允许用可开发性与生成器自身似然，不得使用
  自研亲和力分数**。ESEL 的候选排序者是实验富集，不是任何模型；
- 第 4 条：项目目标（一个实测 Kd 的无序表位结合体）由「外部生成 + 实验测量」承担
  ——ESEL 就是这条的外壳；
- 第 5 条：AF2/界面 PAE 不作门禁；E2/E3 不依赖任何 AF2 运行。

## 1. 错误模式 → 本战役防线（全部引用仓库内证据，不是事后概括）

| # | 错误模式 | 证据出处 | ESEL 防线 |
|---|---|---|---|
| M1 | 先跑实验后审数据 | `docs/PAE_EVIDENCE_AUDIT_V20.md` | E0 数据先行：呈现矩阵先于任何湿实验接触 |
| M2 | 裁判未验证就当裁判 | `INTEGRATIVE_ANALYSIS.md` §6 | 本战役**没有计算裁判**；选择压力 = 展示 panning 富集，构造性消解该模式 |
| M3 | 基建 bug 污染科学结论 | `HANDOFF_V5.md` §三 | 契约测试先行（`tests/test_esel_e0_contracts.py`），fail-closed 守卫进脚本 |
| M4 | 看结果改协议 | `INTEGRATIVE_ANALYSIS.md` §4 | E1 文库规格先冻结后合成；凡接触过数据的选项只能写 development evidence |
| M5 | 弱标签烧训练周期 | `INTEGRATIVE_ANALYSIS.md` §8 | 不训练；E2 富集数据是**产出**不是输入 |
| M6 | 版本链漂移 | `docs/PAE_ANALOG_TRANSFER_V30.md` 边界节 | 新线新名；每阶段"产物+允许表述+仍缺什么"三栏，claim 先写后测 |
| M7 | 小样本确认性检验 | v23–v27 门控链 | E2/E3 的终点是能力判定（≥1 克隆实测 Kd），不是显著性主张；功效陈述只对富集一致性作描述 |
| M8 | 生成器/裁判能力错配 | `INTEGRATIVE_ANALYSIS.md` §6.2 | 库存的角色是**原料**不是主张：文库分布设计继承 C1 修复后教训（native 26/26 但结合判别未建立） |

## 2. 阶段结构与停止规则

```
E0 表位选择与呈现矩阵（数据先行，纯离线冻结）
  停止规则：双条件表位（IEDB 先例 AND 可机械呈现）< 3 个
            → 战役停在「表位饥饿」，上报用户增购目标或改疾病方向
E1 文库设计规格（预注册草案 v1 已立，冻结被 E0 完成 + 湿实验伙伴确认阻塞）
  停止规则：伙伴确认失败（预算/通量不可行）→ 库存改购现成合成库，规格降级为备注
E2 展示选择（外包 panning × 呈现矩阵，逐轮 NGS 富集）
  停止规则：连续两轮无富集（enrichment fold < 预注册阈值）→ 换呈现条件；
            三轮仍无 → 该表位判「不可呈现」，矩阵内降级，不 concluded
E3 合成 top 克隆 + BLI/SPR Kd（并入 design_v1_20261101 的 50 条候选实测数据集）
  能力判定：≥1 克隆实测 Kd ≤ 冻结阈值（E1 冻结时定数值）
            → 目标达成；否则按轮换表位回到 E0 扩充宇宙
```

## 3. E0 输入宇宙（全部只读冻结产物）

- 28 组件队列：`results/flex_j0_v1/preregistration.json` `components[]`（14 PDB / 9 家族）；
- 暴露洁净：`results/flex_a1b_isolation_families_v1/exposure.json`；
- 锚点：`results/design_v1_20261101/package.json`（DAEFRH / 4HIX / APP-P05067）；
- IEDB 元数据计数：`results/flex_phase_d_v1/iedb_feasibility_audit.json` +
  `feasibility_addendum_v3.json` + `pilot_families_v3/*.json`（只取计数与 identity）；
- 柔性证据：`results/flex_a2_atlas_plddt_audit_v1/tracks/` 离线缓存；
- 亲和力线索：`results/flex_a4_affinity_inventory_v1/inventory.json`（仅存在性标注）。

## 4. Phase D 数据使用边界（M6 纪律的显式声明）

Phase D 于 2026-11-02 终结（`results/flex_phase_d_v1/termination.json`）。ESEL 只把
v3 结构桥清单（200 accession + IEDB 计数 + identity 块）当作**扩充线索**读取：

1. 不继承其判别器目的，不做 binder-vs-nonbinder 建模；
2. 不下载 IEDB 记录级数据（那是 ESEL 自己预注册之后的动作）；
3. 身份以各文件 `identity` 块为准（`feasibility_addendum_v3.json` 的
   `head_antigens_note` 把 P05877 误标为 α-synuclein；冻结记录里 P05877.1 是
   HIV gp160，α-syn 是 P37840.1）；
4. 若 ESEL 未来要用 v3 引线生成候选，须先冻结 ESEL 自己的预注册。

## 5. 红线继承

- 不触碰 `publication/ECLS_SCOPE_FREEZE.yml`、`publication/candidate_interface_pae_deployment_v1.json`、
  `scripts/evaluate_candidate_interface_final_holdout.py`；
- 本战役任何结论不得与已冻结结果混写或互证；
- 外部基准一接触即冻结用途；
- 冻结产物目录 write-once：修正开 `*_v2` 目录 + 披露文件。

## 6. 执行记录

### E0（2026-11-04 冻结）

- 脚本：`scripts/esel_e0_epitope_selection_v1.py`；
- 契约测试：`tests/test_esel_e0_contracts.py`（跑前全绿）；
- 产物：`results/esel_e0_v1/{protocol,epitope_presentation_matrix,summary,completion}.json`；
- 边界（必须保留）：见产物内 protocol 的 `pending_separate_gates`——ATLAS 同源补查、
  湿实验伙伴确认、病原来源肽段的生物安全审阅均未完成，缺什么列什么，不静默。
