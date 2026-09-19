# PAE 加密面板 v6：启动记录

日期：2026-09-17。状态以 `results/pae_dense_v6/pipeline_status.json` 为准。本文件是启动说明，不表示运行已完成。

## 已固定并启动

- 保留现有六个 EXT 骨架；不读取或使用历史 GP2 最终测试集。
- 面板目标为每个骨架、每种生成方法20条非原生唯一序列，4种方法，共480条候选，另有12条对照。
- 复用原576次生成记录；为ProteinMPNN增加4个固定种子×32次×6骨架，共768次生成，温度仍为0.1。
- 按固定哈希顺序选择；不读取生成评分、PAE或其他置信分数。某组不足20条时停止并报告，不临时改变阈值或数量。
- 旧规范化骨架已核对原CIF和PDB哈希，按完全相同字节复用，避免重复执行昂贵的Bio.PDB层级深拷贝。
- 教师协议维持AF2-Multimer v3，模型1/2，各3种子，3次recycle；全矩阵留档。同序列、骨架和协议的旧结果校验后复用，跨生成方法的相同序列共享教师计算。
- 全部492个实体对应名义2952个模型×种子槽位；真正新增运行数在选取完成及缓存匹配后写入 `progress.json`，不能把全部槽位称为新增计算。
- 全部教师标签齐备后，自动评价原Ridge和冻结的v5组成、位置、二肽模型；不使用加密面板标签重新训练或选参。

## 外部独立性检查

已将原RCSB元数据检索窗口刷新至2026-09-17，仍为45个条目，相对前次新增0个ID。因此此次没有新增独立抗原组件。更密集的同骨架候选只能支持开发评估，不能改称独立外部验证。

快照及差异：`data/pae_external_refresh_v6/delta.json`。这只是原时间窗口的增量检索，不代表穷尽所有潜在外部来源。

## 运行与恢复

生成由 `scripts/generate_pae_dense_v6.py` 执行；`scripts/continue_pae_dense_v6.py` 是本次有限任务的后台串行衔接进程。它等待生成完成，然后进行面板选取、AF2评分和评价，不是定时任务。

- 实时阶段：[pipeline_status.json](../results/pae_dense_v6/pipeline_status.json)
- AF2开始后的进度：`results/pae_dense_v6/progress.json`
- 总日志：`results/pae_dense_v6/pipeline.stdout.log`、`pipeline.stderr.log`
- 每个AF2模型的错误日志：`worker_model1.stderr.log`、`worker_model2.stderr.log`
- 完成后的标签：`teacher_labels.json`；评价：`evaluation.json`；报告：`docs/PAE_DENSE_V6_RESULTS.md`

生成阶段最多等待两小时，超时记录阻塞。AF2连续三个槽位失败时停止并保留记录；没有完整六重复时不自动输出缩小样本后的性能。程序可利用已校验结果恢复，不需重新运行全部槽位。

本轮会持续占用本机GPU；需保持电脑和WSL可运行。尚未产生本轮最终性能，也没有实测筛选流程相对完整AF2的端到端加速。
