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

## 历史清理记录（2026-10-18，经用户明确指示执行）

经用户指示，已用 `git filter-repo --refs refs/heads/release --replace-text` 完成历史改写：

- **范围**：仅 release 分支线。经全引用勘测，敏感信息只存在于 2026-09-29 的检查点提交 59afad54 引入的两个文件旧版本中；master、main、feature 分支、两个 tag、codex 检查点引用均不受影响、原哈希不变。
- **改写**：release 线上共 10 个提交被重写（59afad54 及其后代），三种姓名/邮箱形式在全部历史中替换为占位文本，`git log -S` pickaxe 复扫为零残留；替换后旧对象已 reflog 过期 + gc 物理清除（cat-file 验证不存在）。
- **哈希变化**：release 端点 4ab2ae6f → 7010c975；预留 DOI 的出处提交 04581bb0 → **278ff97c**（submission_metadata.json 新增 reserved_doi_source_commit 字段记录此映射；2026-10-03 状态检查文件为带日期历史快照，保持原样）。
- **stash**：filter-repo 移除了 refs/stash 上的两条临时记录；其内容（train.py）经核对早已提交进 feature/idp-aware，非未合并工作，补丁存档于仓库外备查。
- **备份**：改写前全量 bundle（含旧历史，注意其中仍含被移除作者的信息，勿外传/勿推送到任何远程）位于 `D:/biological/df-history-backup-prescrub-20261018.bundle`；替换规则、stash 补丁与引用快照同目录。
- **遗留风险（无法在本地消除）**：GitHub 端旧提交在 force-push 后变为不可达，但仍可按 SHA 直达，直到 GitHub 侧 gc 或联系 support 清除；任何协作者的旧克隆仍保留完整旧历史，需重新克隆或硬重置。
