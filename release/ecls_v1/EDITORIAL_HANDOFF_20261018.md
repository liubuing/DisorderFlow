# ECLS 交接变更：通讯作者改为胡岳、作者名单调整（2026-10-18）

## 变更内容

依据用户 2026-10-18 的明确指示，`release/ecls_v1/submission_metadata.json` 内部台账调整如下：

- 新增作者：胡岳（Yue Hu / Hu, Yue），置于末位（5 号），通讯作者。
- 移除此前名单中的一位作者：依据用户"全部移除其信息"的指示，其姓名、顺序号、邮箱不再保留于任何当前记录（含 2026-09-29 交接记录，该处已改为移除说明）；2026-09-28 确认的旧五人顺序随之作废。现行名单为：陈昊阳、胡静、戴谭宇、孙宇超、胡岳（通讯，末位）。
- 胡岳的邮箱与单位当前为空，已加入 pending 清单（"Corresponding author (Hu Yue) email and affiliation"）。

## 未变事项

- 作者信息在渲染产物（正文、附信、PDF 元数据、Zenodo creators）中继续隐去，`authors_withheld_from_rendered_artifacts` 保持 true；本变更只改内部台账，无需重新渲染稿件。
- 上传清单（UPLOAD_MANIFEST.json）不覆盖 submission_metadata.json 的哈希，故无需重建清单。
- 科学内容、冻结结果、目标期刊（Bioinformatics）与两线关联披露计划均不变。

## 校验

2026-10-18 运行 `scripts/validate_publication_alignment.py`：作者相关检查在 withheld 模式下按设计跳过，结果与基线一致。既有错误 "Stale upload/source hash: docs/PUBLICATION_MAP.md" 在本次变更前即存在（2026-10-03 后对该文档的编辑晚于清单构建），与本次作者变更无关，待下次重建上传清单时消除。

## 历史遗留说明

上述被移除作者的信息仍存在于 git 历史提交中（submission_metadata.json 与 EDITORIAL_HANDOFF_20260929.md 的旧版本）。若需在公开仓库或归档前彻底清除，需要另行改写历史（如 git filter-repo），该操作影响所有协作者，须经明确决定后执行。
