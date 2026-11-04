# ESEL Phase E1：文库设计规格（预注册草案，**未冻结**）

日期：2026-11-04。**状态：草案——冻结被两件事阻塞**：(a) E0b 手工 IEDB 注录完成
（`results/esel_e0_v1/protocol.json` 的 `stop_rule_evaluation` 已触发升级）；
(b) 湿实验伙伴确认（`docs/ESEL_PARTNER_BRIEF_V1.md` 的三件事）。冻结动作沿用
FLEX 战役先例：本文档定稿 → sha256 记入 `results/esel_e1_v1/preregistration.json`
→ 之后才允许任何文库合成订单或序列级输出。

## 1. 任务定义

**不是候选序列设计。** 本阶段唯一产出是**分布规格**：CDR-H3 长度/电荷/疏水/芳香
位置的条件分布 + 按表位桶的差异化参数，交给文库合成商或展示平台实现。
单序列选择是实验（E2 panning）的职责——这是 DECISION_20261101 第 3 条在本战役的
构造性落实。

## 2. 分布来源（全部可从冻结产物机械导出）

| 分布维度 | 来源 | 导出方式 |
|---|---|---|
| H3 长度分布 | 14-PDB 队列（`flex_j0_v1/preregistration.json` components 的 region_sequence 长度）+ ECLS 面板 H3 长度 | 经验分布，记录直方图 |
| 净电荷窗口 | `results/design_v1_20261101/package.json` filter_rules（已有 [−4,+6] 先例） | 按表位电荷分桶平移 |
| 芳香/Pro 位置偏好 | design 包 filter_rules + 队列 H3 序列统计 | 位置级频率表 |
| 表位桶 | E0 矩阵的 family + flexibility_evidence 桶（酸性/疏水/Pro 富集） | 每桶一个参数集 |

## 3. 终点与门（冻结时定数值，现在定结构）

- **能力终点（E3）**：≥1 克隆实测 Kd ≤ 阈值 T*（数值在 E1 冻结时写死；依据 BLI
  通量与项目目标量级，候选区间 10–100 nM）；
- **E2 过程门**：逐轮 NGS 富集 fold ≥ 预注册阈值（写死）；连续两轮不达 → 换呈现
  条件；三轮不达 → 该表位矩阵内降级；
- **强制对照**（继承 design 包 README 的对照角色表）：无关 scFv 阴性（README 已
  指出 scramble 不构成实验阴性）+ native 阳性对照 + 每表位组成扰动肽（E2 富集的
  序列特异性检查）；
- **功效陈述**：E2 是选择过程，无显著性主张；E3 的 Kd 判定是能力判定，不做群体
  推断——预注册里写明"descriptive capability endpoint"。

## 4. 冻结前置条件（全部满足才执行冻结）

1. E0b 完成：28 队列行的 IEDB accession 手工注录 + ATLAS 同源补查（离线缓存
   1,938 蛋白之外的部分）；
2. 伙伴三件事确认（文库来源/呈现通量/报价）；
3. 表位呈现矩阵的 E0b 修订版冻结（`results/esel_e0b_v1/`）；
4. 生物安全审阅：spike/HIV gp160 来源表位的处理决定（跳过或保留）。

## 5. 红线继承

ESEL_CAMPAIGN_V1.md §5 全文适用。冻结后本文件不可改；修订走 `*_v2` + 披露。
