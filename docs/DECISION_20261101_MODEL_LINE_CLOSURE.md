# 决定记录：关闭「自研结合判别器」线（2026-11-01）

## 背景

本仓库此前有两条与「打分/判别」相关的主张：

1. **ECLS 主稿**（ProteinMPNN 底模）：native vs 组成打乱为正，已冻结；
2. **FLEX 线**（BFN 底模）：J0a/J0b 裁判门通过，C1 排序门失败后转边界研究。

2026-11-01 的复核发现 FLEX 打分路径存在**链角色编码 bug**（`fragment_type` 用了链序号，
H 被编码成 padding、抗原被编码成 Heavy、`mask_antigen` 为空），修复后：
J0a 从 0.124（修复前，7/9 族）变为 5.55（9/9 族）；C1 从 0/26 第 1（修复前）变为 26/26 第 1。
随后做了四项判别测试，全部指向同一结论。

## 证据

| 测试 | 结果 | 结论 |
|---|---|---|
| 修复后 J0a（28 组件/9 族） | +5.553，CI [5.313, 5.917]，9/9 族 | native 识别极强 |
| 修复后 C1（26 池） | native 26/26 第 1，领先最好设计中位 1.77 nats | 但端点是"自己的骨架" |
| 异源骨架矩阵 | 有异源序列的 17 个骨架：自身 17/17 第 1，但每个长度组只有 2 条不同序列 | 证据不足，无法区分自洽性 |
| 肽序列置换（同坐标） | 18/26 偏好自身肽，平均 Δ 仅 **0.076 nats**（对设计优势 1.77） | 肽身份只占约 4% |
| AAYL 结合端点回顾（508 变体） | Spearman **+0.062 / −0.049**，合并 **+0.024**，CI 跨 0 | 与亲和力无关 |
| 组内 pairwise 监督（5 复合物/1,284 变体） | 组成单独 0.26–0.62；BFN+组成 − 组成 = **+0.0098**，2/5 组为正 | 无超出组成的信息 |

## 决定

1. **关闭「用现有数据训练/微调结合判别器」这条线。** 不再以 BFN/ECLS 作为候选选择器；
   任何"排序能力"主张需以本记录中的测试为反例先行反驳。
2. **保留 BFN 作为工具与边界研究对象**（生成器、诊断、教学），不作为设计有效性的证据。
3. **设计候选的选择只允许用可开发性与生成器自身似然**，不得使用自研亲和力分数；
   该约束写入候选包 README（`results/design_v1_20261101/README.md`）。
4. **项目目标（一个实测 Kd 的无序表位结合体）转由「外部生成 + 实验测量」承担**，
   与模型线解耦。
5. **AF2/界面 PAE 不再作为门禁**：本机单条 444 残基复合物预测超过 2 小时，
   吞吐不可用；PAE 仅在外部资源可用时作相对排序参考（冻结部署合同不变）。

## 后果

- FLEX 线的 C1「失败」需按修复后数据重写为「native 恢复通过、结合判别未建立」；
  J0a/J0b 的旧数字（0.124/0.141）作废。
- 候选包（50 条 4HIX H3 设计 + native + scramble）继续推进到合成/BLI；
  选择依据为 MPNN 似然 + 可开发性过滤，与模型判别无关。
- 若未来重启判别器方向，起点必须是**组成受控的实验数据**（同骨架、同表位、
  组成相近、结合差异大），而不是继续在公开变体库上换损失函数。

## 复现

```bash
python scripts/flex_j0_score_v2_antigen_roles.py --all --out results/flex_j0_v2_antigen_roles
python scripts/flex_c1_score_and_rank_v2_antigen_roles.py --all --out results/flex_c1_v2_antigen_roles
python scripts/flex_c1_cross_backbone_matrix_v1.py --out results/flex_cross_backbone_v1
python scripts/score_aayl_bfn_v2_antigen_roles.py --out results/aayl_bfn_v2_antigen_roles
python scripts/build_pairwise_dataset.py
python scripts/score_pairwise_bfn_features.py
python scripts/train_pairwise_head_v1.py
python scripts/eval_within_group_v1.py
```

证据汇总：`.dsh-research/flex-distance/code-fix-report.md`、`supervision-report.md`。

## 传播（2026-11-03 核对）

本记录「后果」一节要求把旧数字与旧叙事清掉；以下文档已逐一核对/修订：

| 文档 | 动作 |
|---|---|
| `docs/FLEX_BINDER_CAMPAIGN_V1.md` §8–§9 | 已加作废横幅：J0a +0.124 / J0b +0.141 / C1 native 0/26 不得再引用；§9 主表加注 |
| `docs/FLEX_PHASE_D_BINDER_TASK_PREREG_DRAFT_V1.md` | "C1 门失败"前置事实已更正为"native 恢复通过、结合判别未建立" |
| `publication/flex_boundary_manuscript_draft_v1.md` | 已加"数字作废、按修复后数据重写"横幅；未投稿 |
| `results/design_v1_20261101/README.md` | 已含"本包没有任何模型分数参与排序"约束（§3 要求），无需改动 |
