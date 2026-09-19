# AbBiBench 结构接入与实验评估准备

2026-09-16。本阶段已完成固定版本结构下载、链与 H3 编号、历史序列重叠筛查及回顾性评估规则制定。未读取个体实验测量值，未运行 ECLS/PAE 评分。

后续状态更新：以上是结构阶段的历史状态。亲本/端点核查及原始 AAYL 队列功能评分已完成，见 `docs/AAYL_FUNCTIONAL_RETROSPECTIVE_RESULTS.md`。已确认 AAYL 为 AF3 预测结构；从原始数据重建 508 条变体，ECLS 未优于简单基线。原始 curated 端点发现与重复取对数一致的转换异常，未直接用于最终评分。

## 实际接入结果

固定数据版本：`556fd6913aa231c0d342a8658818be8f963fd582`。17 个元数据条目映射到 12 个结构文件，其中 11 个文件内容互不相同。全部结构的 H/L 编号角色与声明一致，已观测残基的 N/CA/C/O 主链原子齐全；这不代表实验结构不存在未观测残基，也不代表已经完成实验序列到坐标的匹配。

|结构编号|数据来源|H3 序列|抗原链长度|MMseqs 历史重叠轴|
|---|---|---|---|---|
|ABB001|1mhp|TRGFGDGGYFDV|184|配对 CDR、抗原|
|ABB002|1mlc|ARGDGNYGY|129|VL、配对 CDR、抗原|
|ABB003|1n8z / 4d5_her2|SRWGGDGFYAMDY|581|VL、配对 CDR、抗原|
|ABB004|2fjg / g6_LC|ARFVFFLPYAMDY|95|VL、配对 CDR、抗原|
|ABB005|3gbn|AKHMGYQVRETMDV|328 + 173|全部五轴|
|ABB006|4fqi|ARHGNYYYYSGMDV|324 + 176|全部五轴|
|ABB007|5a12_vegf|ARFVFFLPYAMDY|99|VL、配对 CDR、抗原|
|ABB008|5a12_ang2|ARFVFFLPYAMDY|220|VL、配对 CDR、抗原|
|ABB009|aayl49 / aayl49_ML|AKGRAAGTFDS|14|无命中|
|ABB010|aayl50|AKGRAAGTFDS|14|无命中|
|ABB011|aayl51|AKVGRGGGYFDY|14|VH、VL、配对 CDR|
|ABB012|aayl52|ARVGRGVIDH|14|无命中|

沿用 VH/VL/配对 CDR/H3/抗原一致性阈值 0.90/0.90/0.70/0.50/0.30，双向覆盖率 0.80。多链抗原逐链检索，任意链命中均记录。原始 MMseqs 筛查为 9/12 结构命中；其余 3 个结构只构成 1 个组件。全部 12 个候选形成 5 个组件，不能把 12 个文件视作 12 个独立任务。

## 短序列敏感性检查

MMseqs 的启发式搜索可能漏掉很短的 H3/肽序列。新增独立输出，对长度不超过 50 的查询逐偏移枚举无缺口比对，仍使用相同的一致性和双向覆盖率门槛，保留原始 MMseqs 结果不覆盖。

合并此保守检查后，12/12 结构均触发至少一条排除规则，剩余 0 个。AAYL 共同抗原的示例匹配为 12 个对齐位置中 4 个相同，一致性 1/3，查询覆盖率 12/14，参考覆盖率 1.0。**这只说明触发当前操作性门槛，不能据此证明短肽存在进化同源性或实际数据泄漏**；低阈值短肽匹配可能偶然出现。该补充检查不是原始 MMseqs 预设流程，必须并列报告，不能将其包装成原流程结果。

据此，本阶段不授予这批数据“严格独立确认集”的资格。即使仅采纳原始 MMseqs 结果，也只有一个剩余组件，仍不足以支撑跨抗原泛化主张。无缺口枚举也不穷尽所有带缺口比对。

## 数据问题与解释

- 缺失表项仍为 `1mlc_LC_benchmarking_data.csv`、`1mhp_benchmarking_data.csv`；不猜测替代文件。
- `aayl50_bca.pdb` 按仓库唯一匹配显式修正为 `AAYL50_bca.pdb`，原始元数据未改。
- AAYL49 与 AAYL50 文件 SHA-256 完全相同。原研究设计包括同一亲本的重链/轻链突变库，因此共享结构可能合理；不能仅因文件相同认定上游错误。具体亲本匹配尚待序列核实。
- 四个 AAYL 抗原均为 `PDVDLGDISGINAS`。已从[原始实验论文](https://www.nature.com/articles/s41597-022-01779-4)核实这是 SARS-CoV-2 HR2 靶肽。因此应修正上一阶段“主要为蛋白抗原”的粗分：本数据源包含可单独分析的肽子集，但其结构来源仍未核实为实验解析或计算模型。
- AAYL 实验测量由 AlphaSeq 酵母配对信号经标准曲线估计亲和力，不能写成逐变体 SPR/ITC 直接 Kd。原论文支持该测量类别；当前未解析个体值。
- `5a12_vegf` 的辅助 epitope/paratope 字段与其 H/L/antigen_chains 声明冲突。按声明链读取坐标后，H3 到抗原最小 CA 距离约 8.644 Å，8 Å 内 H3 位置数为 0；CA 结果不能等同于不存在侧链接触。保持原结构，不按预期分数挑另一条链。

## 已制定的评估规则与剩余条件

机器可读方案：`configs/benchmarks/abbibench_functional_retrospective_v1.json`，当前明确为 `ready_for_scoring=false`。

主要评估限定为 H3 内替换且 H/L 其他位置不变的变体，沿用固定骨架 H3 ECLS，与负 complex NLL 比较；负 apo NLL 为次要基线。LC-only 或其他位置突变只计入适用范围核查，不通过改变评分区域来适应本批数据。每个测量条件内计算 Spearman 及配对差值，按谱系/抗原组件汇总，不将大量突变行当作独立抗原。

直接 Kd、AlphaSeq 估计及富集分数分别报告。禁止重复对已变换的分数取对数；删失值不伪装为精确读数；重复技术测量按已确认的变换尺度取中位数。少数组件只支持描述性分析，不宣称确认性优势。

在评分前还需完成：逐表端点字典及单位/方向/删失规则核实、亲本序列到坐标映射、AAYL 结构来源审计、5a12_vegf 链冲突核实、H3-only 变体数量和实际组件可行性统计。以上是缺失证据和数据依赖，不是额外授权请求。

## 产物与验证

`data/abbibench_structural_audit_v1/` 保存结构、哈希、映射、编号、完整结构清单、MMseqs 命令与命中、原始隔离报告及短序列补充报告。

复现：

```powershell
python scripts/prepare_abbibench_structural_audit.py --stage prepare
python scripts/prepare_abbibench_structural_audit.py --stage audit
python scripts/audit_abbibench_short_sequences.py
python -m pytest tests/test_abbibench_structural_audit.py -q
python scripts/validate_publication_alignment.py
```

3 项新增测试通过，覆盖大小写解析歧义、双向覆盖率及短序列阈值边界。冻结论文材料一致性检查通过。首次结构下载遇到网络限制，授权执行后成功。
