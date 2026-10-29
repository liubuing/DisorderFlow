# 代码注释与冻结脚本注解索引

更新：2026-10-24。本目录是"全仓代码注释工程"的配套文档。

## 覆盖格局

项目自有代码共 **855 个 .py**（git 追踪 931 减 vendored 的 diffab/ 73 与 ProteinMPNN/ 12，另含本会话新建脚本），按溯源约束分成两类处理：

| 类别 | 数量 | 处理方式 | 校验 |
|---|---:|---|---|
| 非冻结文件 | 606 | 就地中文注释（只新增 `#` 注释行与缺失的模块 docstring） | 全量 `ast.parse` 零失败；AST 剥离注释/docstring 后与 HEAD **语义等价（0 差异）**；纯插入核验通过 |
| 冻结文件（sha256 被协议记录） | 249 | **不可就地修改**；配套只读注解文档（本目录三个 FROZEN_ANNOTATIONS_*.md） | 注解条目与 frozen_list 1:1 比对覆盖；冻结文件 git 状态零改动 |

排除说明：`diffab/`、`ProteinMPNN/` 为 vendored 第三方仓库，不作注释；`venv*/`、`Lib/`、`lib/`、`.venv*/` 为环境目录，不属项目代码。

## 冻结机制与为何不能改冻结脚本

大量协议 JSON（`results/**/protocol.json`、`release/ecls_v1/source_manifest_frozen_v1.json`、`release/zenodo_v1/upload/source_manifest.json` 等）记录了脚本的 sha256，且部分脚本运行时自我断言哈希（如 `scripts/guard_pae_batch_v19.py`）。修改这些文件会使：已发布证据链的哈希校验失败、`tests/test_release_lineage.py` 的机械校验失败、论文"脚本带哈希回执"的声明失真。因此注释以配套文档形式提供。

## 文档清单

- `FROZEN_ANNOTATIONS_scripts_a.md` — PAE 类似物迁移证据链 v23–v30 及运行时基础设施（14 个，注解作者第一手撰写）
- `FROZEN_ANNOTATIONS_scripts_b.md` — 其余全部 scripts/ 冻结脚本（114 条；含 3 个 0 字节占位文件的如实标注；43 个定位到具体冻结协议目录）
- `FROZEN_ANNOTATIONS_packages.md` — disorderflow/ 94 + modules/ 4 + tests/ 25（123 条；109 个定位到 ECLS 发布清单，其余定位到 PAE/H3/StateContrast 协议）

## 执行记录

- 执行方式：分批并行代理 + 主会话内联；限流（1302/1308）期间降级为主会话逐文件处理。
- 损伤修复：过程中发现并修复 2 处代理引入的缩进损伤（`scripts/ablation_disorder_conditioning.py`、`scripts/crosswalk_abbibench_aayl_endpoints.py`），修复后均通过 AST 等价检验。
- 终验命令口径：`git status --porcelain` 对照冻结清单（`M/A` 为真改动）；`ast.parse`（utf-8-sig 容错）；`ast.dump` 剥离注释与 docstring 后与 `git show HEAD:` 比较。
