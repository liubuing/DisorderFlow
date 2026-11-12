# 4HIX AntiFold H3 候选臂（第二生成器）

日期：2026-11-12。状态：**可合成交付物；本环境未执行湿实验。**

## 1. 为什么加这一臂

现有 4HIX 候选包来自 ProteinMPNN。2026-11-12 用 AntiFold（ESM-IF1 抗体微调）
在同一个 4HIX Fab–DAEFRH 复合物上重设计 CDR-H3，作为**第二生成器对照**：

- AntiFold 200 条采样（T=0.2 / 0.5），125 个唯一 H3；
- 可开发性过滤 100% 通过；
- 平均 5.25 个 H3 突变（ProteinMPNN shortlist 平均 7.06）；
- AntiFold 只采样 6 个 H3 位置（IMGT 107/109/110/112/114/115），比 MPNN 保守；
- **没有任何 ECLS/BFN 亲和力分数参与选择**；排序依据是 AntiFold score + 可开发性 + 多样性。

## 2. 首批 10 条

| ID | H3 | AntiFold score | 相对 native 突变数 |
|---|---|---:|---:|
| 4HIX-native | VRYDHYSGSSDY | — | 0 |
| 4HIX-AF001 | VREDYSYGVFDY | 1.1296 | 6 |
| 4HIX-AF002 | VRYDEFYGAMDY | 1.0714 | 5 |
| 4HIX-AF003 | VRHDRYYGTADY | 1.0707 | 5 |
| 4HIX-AF004 | VRMDMYYGAIDY | 1.0488 | 5 |
| 4HIX-AF005 | VREDVSFGTMDY | 1.0350 | 6 |
| 4HIX-AF006 | VRHDARYGSFDY | 1.0253 | 5 |
| 4HIX-AF007 | VRMDLSSGAPDY | 1.0232 | 5 |
| 4HIX-AF008 | VRWDLRYGSMDY | 0.9952 | 5 |
| 4HIX-AF009 | VRYDIRFGAFDY | 0.9801 | 5 |
| 4HIX-AF010 | VRQDGSYGHMDY | 0.9801 | 6 |
| 4HIX-H3scramble | YDVRSGHYSSDY | — | 反事实对照 |

完整 H3 见 `antifold_candidates.csv`；可合成序列见 `antifold_arm_order_list.csv`
（scFv 266 aa / Fab VH 219 / VL 219）。

## 3. 文件

| 文件 | 内容 |
|---|---|
| `antifold_arm_order_list.csv` | 12 条 × scFv/VH/VL |
| `antifold_arm_order_list.fasta` | 36 条序列（每条 3 种格式） |
| `antifold_arm_manifest.csv` | 72 行：12 样本 × 2 浓度 × 3 复孔 |
| `antifold_plate_layout_96well.csv` | 96 孔布局（列 1–12：native + AF001–AF010 + scramble） |
| `../results/hf_aux_benchmark_v1/antifold_4hix/antifold_candidates.csv` | 10 条选中 + 分数/突变数 |
| `../results/hf_aux_benchmark_v1/antifold_4hix/antifold_all_scores.csv` | 200 条全量 |

## 4. BLI 执行

与主臂相同：

- 生物素-Aβ42（或 DAEFRH）固定 SA tips；
- 100 nM / 10 nM，三复孔；
- 结合 120 s，解离 180 s，glycine pH 1.5 再生；
- 验收：native ≥3× BSA；设计 hit ≥3× BSA；scramble ≤20% native；复孔 CV ≤20%；
- hit 后 8 点浓度 1:1 Langmuir，报告 kon/koff/KD；
- 分母：合成 → 表达 → 可溶单体 → 可测 → hit。

## 5. 边界

- AntiFold score 是序列-结构似然，不是亲和力；
- 只采样 6 个 H3 位置，多样性低于 ProteinMPNN；
- 1N8Z/4HIX 骨架、DAEFRH 表位；
- 没有湿实验，不能写“结合更好”；
- 这一臂是**生成器对照**，不是 ECLS/BFN 的验证。
