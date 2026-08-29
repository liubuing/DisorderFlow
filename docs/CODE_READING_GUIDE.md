# DisorderFlow 代码阅读指南

本文档说明项目的核心执行链、重要数据结构和科学边界。建议先按下列顺序阅读，不要从历史实验脚本或 `app.py` 的 UI 定义开始。

## 1. 最短阅读路径

1. `README.md`
   - 了解项目能做什么、证据等级和已知限制。
2. `app_config.yaml`
   - 查看当前 checkpoint、WSL、ColabFold、输出目录和默认阈值。
3. `modules/runtime_environment.py`
   - 理解 Windows Python 如何调用 WSL2 ColabFold。
4. `modules/bfn_loader.py`
   - BFN 加载、区域 mask、候选注入、生成和固定评分的统一入口。
5. `scripts/generate_idp_ensemble_v3_candidates.py`
   - 多构象和单构象 H3 候选如何在相同预算下生成。
6. `scripts/score_idp_ensemble_v3_source_sensitivity.py`
   - 固定候选如何在 experimental、sampled、mixed pose panel 上重评分。
7. `scripts/score_idp_ensemble_v3_leave_one_pose_out.py`
   - 删除单个 pose 后如何检查候选排序和方向稳定性。
8. `scripts/check_idp_level2_computational_validation.py`
   - 多个证据轴如何合并成 fail-closed 的二级判定。

## 2. 运行时结构

```text
Windows Python
  -> BFN / ProteinMPNN / 数据处理 / Web UI
  -> modules/runtime_environment.py
  -> wsl.exe -d Ubuntu-24.04-D
  -> WSL JAX / ColabFold
  -> RTX 3080
```

推荐入口：

```powershell
.\run_disorderflow_test.ps1
.\run_disorderflow.ps1 start
```

`run_disorderflow.ps1` 显式设置 D 盘 Python、模型缓存、临时目录和 WSL 发行版，避免重新引入 C 盘用户缓存依赖。

## 3. BFN 核心数据流

```text
PDB
  -> preprocess_protein_structure
  -> mask_region
  -> merge_protein
  -> patch_protein
  -> PaddingCollate
  -> model.sample 或 model.score
```

关键接口位于 `modules/bfn_loader.py`：

| 接口 | 用途 |
|---|---|
| `parse_region_spec` | 将用户的 1-based chain-local 位置转换为 0-based transform 索引 |
| `build_region_batch` | 生成和固定评分共用的结构 batch |
| `inject_candidate_sequence` | 仅替换 `generate_flag` 对应的设计位置 |
| `score_bfn_candidate` | 不采样，直接评分外部候选 |
| `run_bfn_design` | 采样候选并输出 PPL、entropy 和置信度特征 |

注意：`A:10-20` 表示解析后 A 链序列的第 10-20 个位置，不一定等同于 PDB 文件中的 `resseq 10-20`。带 insertion code 或缺失残基的出版级任务必须先完成编号映射。

## 4. Context Chain 语义

`build_region_batch` 和 `run_bfn_design` 对 `context_chains` 的约定：

| 值 | 含义 |
|---|---|
| `None` | 包含所有非设计链，属于 complex-conditioned 模式 |
| `[]` | 不包含额外链，属于 fixed-backbone 模式 |
| `['A']` | 只包含指定的可见上下文链 |

`antigen_chains` 必须是 `context_chains` 的子集。它用于设置 fragment role，不会自动添加缺失的上下文链。

## 5. 多构象候选生成

`scripts/generate_idp_ensemble_v3_candidates.py` 中有两个匹配设计臂：

```text
ensemble
  -> 所有冻结 pose
  -> 每个 H3 位置按 pose 等权平均 log probability

single_state
  -> 仅冻结的第一个实验 pose
```

两条臂保持相同：

- ProteinMPNN checkpoint；
- 随机种子；
- mutation bucket；
- 每个 bucket 的候选数量；
- H3 设计区域；
- framework、light chain 和 antigen sequence。

`BatchedPoseAdapter.next_log_probs(prefix, position)` 是真正的自回归接口。模型只接收已经采样的 H3 prefix，不能读取未来设计残基。

`HeterogeneousPoseAdapter` 用于处理缺失残基或 chain layout 不同的 pose。它分别 featurize 后再平均输出，避免错误地把不兼容布局 padding 到同一索引语义中。

## 6. 固定候选评分

固定评分的原则：候选序列生成完成后不再改变，只更换 pose panel 或评分器。

相关入口：

| 脚本 | 作用 |
|---|---|
| `score_idp_ensemble_v3_source_sensitivity.py` | 比较 experimental、sampled、mixed pose source |
| `score_idp_ensemble_v3_leave_one_pose_out.py` | 删除每一个 pose 后重评分 |
| `score_idp_ensemble_candidates_bfn.py` | 使用 BFN 固定评分输出进行独立开发诊断 |
| `score_idp_ensemble_expanded_independent_v2.py` | 使用 contact-chemistry heuristic 做正交评分 |

不要把 context-only ipTM 当成同一结构内的候选级 ranking metric。该项目的开发审计显示其组件内变化很小。`state_compatibility` 更敏感，但开发门槛同样没有通过，因此也不能直接作为正式终点。

## 7. Robustness 与 Abstention

Source sensitivity 检查：

```text
experimental_only
sampled_only
mixed
```

Leave-one-pose-out 检查：

```text
完整 mixed panel 排名
  vs
每次删除一个 pose 后的排名
```

`scripts/analyze_idp_level2_stability_abstention.py` 要求 source direction 和 leave-one-pose direction 同时稳定。

Abstention 的含义是“不给出结论”，不是“效应等于零”。统计汇总时必须从条件效应集合中排除 abstained pair，并单独报告 coverage。

## 8. 二级计算判定

配置：

```text
configs/benchmarks/idp_level2_computational_validation_v1.yml
```

执行：

```powershell
D:\DisorderFlowRuntime\Python314\python.exe `
  scripts\check_idp_level2_computational_validation.py
```

输出：

```text
reviewer_outputs/idp_level2_computational_validation_v1/status.json
reviewer_outputs/idp_level2_computational_validation_v1/report.md
```

状态语义：

| 状态 | 含义 |
|---|---|
| `level2_computational_validation_passed` | 所有现有证据和缺失证据门均通过 |
| `level2_failed_existing_evidence` | 已经观察到方法失败，不能通过新增 cohort 消除 |
| `level2_blocked_missing_evidence` | 现有方法门通过，但仍缺预注册数据 |

已有负证据优先于缺失证据。即使以后获得 untouched IDP cohort，只要当前方法失败项未通过，也不能升级二级结论。

## 9. AF2 批处理与 Resume

`scripts/run_multiscaffold_v2_af2.py` 采用 JSONL 将任务传给一个持久 WSL worker：

```text
Windows orchestrator
  -> JSONL stdin
  -> scripts/utils/af2_wsl_batch.py
  -> JAX model load/JIT 一次
  -> 多个 prediction slot
```

每个 slot 都必须记录 success 或 failed。批次失败不能通过缩小分母静默删除。

`--resume` 会核对 config hash、selection hash、entity 列表和 seed 列表。任一项改变都会拒绝续跑。

## 10. 科学边界

代码中的指标只能支持计算候选优先级：

- BFN/ProteinMPNN likelihood 不是结合自由能；
- AF2 ipTM 和 interface PAE 不是实验亲和力；
- contact-chemistry 是 heuristic；
- developability 是序列代理；
- MD 稳定不等于真实结合或特异性。

阅读结果文件时先查看：

```text
status
classification
provenance
gate_results
decision
claim_boundary
```

不要只读取均值或 Top-1 候选。

## 11. 测试入口

快速检查本轮核心逻辑：

```powershell
D:\DisorderFlowRuntime\Python314\python.exe -m pytest `
  tests\test_idp_level2_computational_validation.py `
  tests\test_idp_level2_stability_abstention.py `
  tests\test_idp_ensemble_bfn_state_compatibility.py -q
```

全量测试：

```powershell
D:\DisorderFlowRuntime\Python314\python.exe -m pytest -q
```

新增核心模块必须至少满足：

- import/compile 成功；
- 输入 hash 不匹配时 fail closed；
- 不静默删除失败样本；
- 不覆盖 write-once 冻结产物；
- 输出包含 claim boundary。

## 12. StateContrast-v2 训练线

StateContrast-v2 是独立开发线，不修改冻结的 `antibody_bfn` 和 ECLS 发布源。

核心文件：

| 文件 | 作用 |
|---|---|
| `disorderflow/modules/statecontrast_v2.py` | target/apo/off-target loss、source-constrained pose weight、independent teacher consistency |
| `disorderflow/models/statecontrast_v2.py` | 新模型类型 `antibody_bfn_statecontrast_v2` 和可学习 state/pose head |
| `disorderflow/datasets/statecontrast_v2_structural.py` | 将显式 state/source/teacher metadata 加到结构 batch |
| `disorderflow/utils/source_balanced_sampler.py` | 在不拆 evidence group 的前提下轮换 dataset source |
| `disorderflow/statecontrast_v2_contract.py` | 训练 manifest 的 fail-closed schema 验证 |
| `train_statecontrast_v2.py` | 专用训练入口，不改变默认 `train.py` 行为 |

训练目标：

```text
target state score
  > apo state score + apo margin
  > worst off-target state score + off-target margin
```

Pose weighting 先在 source 之间分配质量，再在同 source 的 pose 之间分配质量。这样增加更多 sampled pose 不会自动压过一个 experimental pose。`minimum_source_mass`、`maximum_pose_weight` 和 KL regularization 防止权重塌缩到单个 pose。

Independent teacher score 必须离线生成并在模型 loss 中 `detach`。它只监督同 group 内的 pair order，不允许模型通过教师计算图反向传播，也不能使用 validation/test teacher 标签训练。

当前 expanded IDP readiness：

```powershell
D:\DisorderFlowRuntime\Python314\python.exe `
  scripts\audit_statecontrast_v2_expanded_readiness.py
```

输出：

```text
reviewer_outputs/statecontrast_v2_expanded_readiness_v1/status.json
```

当前 expanded artifacts 只有 target-bound pose。缺少真实 apo/off-target 和足够 teacher coverage 时，训练入口不应启动。特别禁止把 restrained sampled target pose 重命名为 apo 或 off-target。

显式状态数据构建：

```powershell
D:\DisorderFlowRuntime\Python314\python.exe `
  scripts\build_statecontrast_v2_manifest.py `
  --input <frozen_explicit_state_input.json> `
  --output data\statecontrast_v2_expanded_idp\manifest.json
```

Split 使用全局 antibody-lineage 与 sequence cluster 的组合 hash，不包含 dataset source 名称，避免同一谱系从不同 source 进入不同 split。

ProteinMPNN prefix adapter 只将 H3 标记为 design region。固定 heavy framework、light chain 和 antigen sequence 都属于已知上下文；未来 H3 suffix 仍由 decoding order 隐藏。修复后的候选必须写入新的版本化目录，不得覆盖旧候选结果。

## 13. StateContrast-v2 完整开发流水线

### 13.1 物化显式状态

```powershell
D:\DisorderFlowRuntime\Python314\python.exe `
  scripts\prepare_statecontrast_v2_explicit_states.py
```

该步骤生成：

- target：冻结 bound pose；
- apo：删除 antigen chain，但保留同一 antibody frame；
- off-target：在固定几何上使用跨 target antigen-sequence mismatch，标记为 `synthetic_weak_not_experimental_nonbinder`，训练权重为 0.25。

Off-target mismatch 不是实验 non-binder。它只提供序列特异性弱监督。

### 13.2 构建全局 cluster split 和五折编号

```powershell
D:\DisorderFlowRuntime\Python314\python.exe `
  scripts\build_statecontrast_v2_manifest.py `
  --input reviewer_outputs\statecontrast_v2_expanded_states_v1\explicit_states.json `
  --output reviewer_outputs\statecontrast_v2_expanded_states_v1\training_manifest.json `
  --folds 5
```

同一 antibody lineage 与 sequence cluster 的所有 candidate、pose 和 state 都进入同一 fold。

### 13.3 冻结五折配置

```powershell
D:\DisorderFlowRuntime\Python314\python.exe `
  scripts\prepare_statecontrast_v2_cross_validation.py
```

输出 `reviewer_outputs/statecontrast_v2_cross_validation_v1/contract.json` 和五个 fold config。

### 13.4 只生成执行计划

```powershell
D:\DisorderFlowRuntime\Python314\python.exe `
  scripts\run_statecontrast_v2_cross_validation.py `
  --device cuda
```

没有 `--execute` 时不会启动 GPU 训练。确认显卡空闲后才运行：

```powershell
D:\DisorderFlowRuntime\Python314\python.exe `
  scripts\run_statecontrast_v2_cross_validation.py `
  --device cuda --execute
```

每折必须从配置指定的相同 initializer 启动，并只评估该折的 heldout components。

### 13.5 OOF 汇总

```powershell
D:\DisorderFlowRuntime\Python314\python.exe `
  scripts\aggregate_statecontrast_v2_oof.py `
  --evaluation 0=<fold0-evaluation.json> `
  --evaluation 1=<fold1-evaluation.json> `
  --evaluation 2=<fold2-evaluation.json> `
  --evaluation 3=<fold3-evaluation.json> `
  --evaluation 4=<fold4-evaluation.json> `
  --output reviewer_outputs\statecontrast_v2_oof_v1\analysis.json
```

OOF 汇总器拒绝漏折和 component 跨折重复。只有合计覆盖至少 20 components 和 4 targets，且全部冻结门槛通过，才允许冻结方法去接触一个新 untouched IDP panel。

### 13.6 训练后候选生成

```powershell
D:\DisorderFlowRuntime\Python314\python.exe `
  scripts\generate_statecontrast_v2_candidates.py `
  --config <fold-or-final-config.yml> `
  --checkpoint <best.pt> `
  --output <versioned-candidates.json>
```

两个 arm 使用相同采样 attempt 数：

- ensemble arm 在冻结的多个 target pose 间分配采样；
- single-state arm 只使用第一个 experimental pose；
- 所有固定候选随后在 target/apo/off-target states 上重评分；
- 每个 2/4/6/8 substitution bucket 只选择实际生成且精确满足 bucket 的候选；
- 缺失 bucket 记录为失败，不插值、不复制、不降低 mutation budget。

StateContrast-v2 的主 state score 是设计区域 `-NLL`，梯度直接进入 sequence decoder。Pose-quality head 只负责 source-constrained pose aggregation，不承担旁路分类任务。

## 14. 更广 IDP Ensemble 覆盖

执行：

```powershell
D:\DisorderFlowRuntime\Python314\python.exe `
  scripts\statecontrast_v2_cli.py coverage
```

覆盖审计按 antibody framework 对齐后计算 antigen CA RMSD，并使用 near-duplicate 连通分量计算 effective poses。通过要求至少两个非局部 pose；一个 deposited pose 加多个 restrained local replicas 不算广 ensemble。

支持的 broad source 类别：

- independent experimental structure；
- NMR/deposited model；
- 预声明 predicted ensemble；
- local sampling 只能补充，不能单独满足覆盖。

## 15. 实测 Developability 接口

生成测量模板：

```powershell
D:\DisorderFlowRuntime\Python314\python.exe `
  scripts\statecontrast_v2_cli.py developability `
  --template-from reviewer_outputs\statecontrast_v2_expanded_states_v1\explicit_states.json
```

审计测量结果：

```powershell
D:\DisorderFlowRuntime\Python314\python.exe `
  scripts\statecontrast_v2_cli.py developability `
  --measurements <measurements.json>
```

必需字段包括 replicate 数、表达量、SEC monomer fraction、aggregate fraction 和 normalized polyspecificity。缺失值 fail closed；sequence proxy 或模型预测不能代替测量字段。

## 16. 仓库边界

参见 `docs/REPOSITORY_BOUNDARIES.md` 和 `configs/repository_layout.yml`。

```powershell
D:\DisorderFlowRuntime\Python314\python.exe `
  scripts\audit_repository_boundaries.py
```

冻结 evidence 不移动、不覆盖；现有根目录历史脚本保留以维持 provenance。新工作流必须进入 `scripts/`，可复用模型/数据/loss 进入 `disorderflow/`。
