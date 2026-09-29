# Checkpoint 真相表

> 生成：2026-09-27。本文件是"哪个模型是当前真相"的唯一权威索引。
> 原则：改 checkpoint 指向时必须同步更新本表并留原值注释。

## 当前主力（日常设计/评分用）

| 用途 | 路径 | 指向它的配置 |
|---|---|---|
| App + CLI 通用 BFN 设计 | `logs/bfn_disorder_v4_afdb_supervised_2026_08_05__17_22_45_disorder_v4_balanced_long_s2032/checkpoints/best.pt` | `app_config.yaml → models.bfn.checkpoint`；`configs/demo_design.yml`（2026-09-27 纠偏，原值 v6_phase2_2026_05_18） |

覆盖方式：环境变量 `DISORDERFLOW_CHECKPOINT=/绝对路径/best.pt`（`get_bfn_ckpt()` 最高优先级）。

## 论文冻结资产（不可改动，只读）

| 资产 | 路径 | 说明 |
|---|---|---|
| PAE 代理三种子部署 | `logs/bfn_candidate_interface_multiscaffold_dual_sem_v2_2026_09_01__*/checkpoints/{1000,800,800}.pt` | 冻结校准映射在 `publication/candidate_interface_pae_deployment_v1.json`；对应 Bioinformatics 投稿稿 |
| ECLS 冻结结果 | `publication/ECLS_SCOPE_FREEZE.yml` + 结果文件 sha256 | 禁止 temporal-final 重跑 |

## 历史版本（不删，但不是"当前"）

| 模型 | 状态 |
|---|---|
| `logs/bfn_confidence_combined_v6_phase2_2026_05_18__14_05_03/` | 2026-05 置信度头早期版本；曾是 demo_design.yml 默认（2026-05-18 ~ 2026-09-27），已让位 |
| `logs/` 下其余 `bfn_*` 时间戳目录 | 演化档案；新任务一律从"当前主力"出发，不回退 |

## 规则

1. 换主力模型 = 改 `app_config.yaml` + 本表记一行 + 原值注释留在被改的配置里。
2. 任何 config 里的 checkpoint 字段只允许指向"当前主力"或"论文冻结资产"，不允许第三种状态。
3. 疑惑时以本表为准；本表与实际不符时，以修表为第一动作。
