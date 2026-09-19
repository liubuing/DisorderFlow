# 独立数据可行性审计

日期：2026-09-16。状态：完成本轮检索与元数据核查；尚未获得新的独立验证结果。

后续已完成结构接入：见 `docs/ABBIBENCH_STRUCTURAL_AUDIT.md`。12 个结构文件中包含 4 个 AAYL 肽结构，后者共享同一 14 aa 抗原；不能将本来源整体归为蛋白抗原。原始 MMseqs 剩余 3 个结构 / 1 个组件，新增短序列敏感性检查后没有通过者。实验回顾性评估规则已制定，评分仍需端点及亲本映射。

## 原任务结构检索

固定范围为 2025-01-01 至 2026-09-16 发布的实验结构，沿用配对重轻链抗体、5–50 aa 肽及既有结构筛选规则，包含病毒条目。结果如下：

|步骤|数量|
|---|---:|
|RCSB 返回条目|1,223|
|初步元数据候选|111（非病毒 55、病毒 56）|
|按既有历史元数据暴露规则排除|111|
|剩余结构候选|0|
|新增独立组件|0|

111 条均在项目之前的检索中出现。此处“元数据暴露即排除”是项目沿用的保守设计规则，不表示只见过元数据在所有研究中都等于训练泄漏。不能在看到本轮零结果后改规则并将旧样本重命名为独立验证。

隔离流程完整输出了失败报告；因没有剩余候选，未执行非空序列同源检索，也没有产生新的模型分数或 AF2 标签。VH/VL/配对 CDR/H3/抗原阈值仍为 0.90/0.90/0.70/0.50/0.30，覆盖率 0.80。12 组件是预设可行性门槛，不是统计功效证明。检索范围和描述词有局限，本结果不能解释为全球不存在可用数据。

产物均在 `data/ecls_independent_feasibility_v1/`：`query.json`、`protocol.json`、`reference_union.json`、`snapshot/`、`discovery_unexposed.json`、`structures/structural_manifest.json`、`isolation_audit.json`。快照、引用来源及审计输入带哈希。复现入口为 `scripts/discover_ecls_independent_v1.py --stage materialize`；既有快照只读复用。

## 实验结合数据的扩展入口

[AbBiBench 论文](https://arxiv.org/abs/2506.04235)汇集抗体突变体实验结合测量，主要覆盖蛋白抗原。它适合评估评分与实验结合的关系，但属于扩展任务，不能直接替代原肽任务确认。

已保存[官方数据集](https://huggingface.co/datasets/AbBibench/Antibody_Binding_Benchmark_Dataset)版本 `556fd6913aa231c0d342a8658818be8f963fd582` 的仓库清单和链映射元数据，未下载个体亲和力表、未进行模型评分。已看过论文和代码首页的汇总结果，不能声称从未接触其公开表现。

- 仓库列出 17 个亲和力 CSV、13 个结构文件；文件数不是独立抗原数。
- 文件名可暂时识别 8 个不同 PDB ID，均未与本项目 exact PDB 排除集合命中；另有 4 个 AAYL 命名结构，身份尚待核实。文件名检查不等于序列、结构或预训练独立性。
- 元数据引用的 `1mlc_LC_benchmarking_data.csv` 和 `1mhp_benchmarking_data.csv` 在固定版本清单中缺失。
- `aayl50_bca.pdb` 引用与实际 `AAYL50_bca.pdb` 大小写不一致；必须显式记录修正映射，避免跨平台静默差异。
- 产物：`abbibench_metadata/repository.json`、`metadata.json`、`audit.json`、`path_consistency.json`。采集脚本：`scripts/acquire_abbibench_metadata.py`。

[AIntibody 前瞻性研究](https://www.nature.com/articles/s41587-026-03238-6)亦公开实验数据，可列为次选来源。本轮仅核查文章与数据可用性说明，未下载个体测量。原研究的盲测性质不自动传递给本项目；RBD 单抗原不能替代多抗原独立组件。

## 下一执行阶段

优先处理 AbBiBench 的结构与测量映射，建立单独的“实验功能外部回顾性评估”方案：

1. 固定版本下载结构，验证 H/L/抗原链及 H3 编号；核实 AAYL 来源与缺失表项，原样记录不支持或缺失项目，不按已公布成绩挑数据。
2. 对全部可解析结构执行五轴历史同源审计及候选间聚类；单列预训练重叠未知项。蛋白抗原与原肽任务分层，不混入冻结主结果。
3. 在读取个体实验值前冻结测量端点、方向、重复/删失值规则、主要比较及失败处理。H3-only 评分应预先限定适用突变区域；LC 或 H3 外突变不能被错误地当作 H3 方法的同一有效性检验。若扩展全可变区评分，需单独命名为新方法。
4. 确定真实组件数及功效计划后再评分，按抗原/谱系报告，不将大量相关突变行当独立样本。优先比较 ECLS 与原始 complex/apo NLL；不继续用旧 6 组件调融合权重。

当前可声称的是已完成数据来源可行性审计；尚不能声称找到了独立亲和力证据或达到了 JCIM 接收标准。非 OA 投稿意向保持。

验证：`python -m pytest tests/test_successor_v3_isolation.py tests/test_discover_rcsb_candidate_interface_extension.py -q`，11 项通过。首次执行因系统临时目录权限失败，授权后原命令通过。
