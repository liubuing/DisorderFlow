# 冻结包代码配套注解（disorderflow/ · modules/ · tests/，共 123 个文件）

> **为什么本文件存在**：下列文件的 sha256 被冻结协议记录——109 个记录于 `release/ecls_v1/source_manifest_frozen_v1.json` 与 `release/zenodo_v1/upload/source_manifest.json`（release 证据链清单，files[].sha256），StateContrast-v2 相关 7 个记录于 zenodo 清单，`modules/af2_jax_runner.py` 记录于 `results/pae_guarded_abba_v16/protocol.json` 等 PAE 协议，peptide 模块与对应测试记录于 `results/publication/h3_submission_package_v1/evidence_manifest.json` 与 iBEC 提交包 `ecls_evidence_manifest.json`。**修改这些文件本身会使已发布证据链的哈希校验失效**，因此不可就地加注释，本文件是只读配套注解。清单来源：`C:\Users\34342\AppData\Local\Temp\comment_batches\frozen_list.txt`。
>
> 条目格式：路径（行数）→ 角色 → 关键类/函数/常量 → 不可就地修改原因。行数为清单生成时的 wc -l。

---

## 一、disorderflow/ 包（94 个文件）

### `disorderflow/__init__.py`（0 行）
**角色**：Django 项目顶层包的空标记文件，无任何代码。不可就地修改：sha256 记录于 release 冻结清单（`release/ecls_v1/source_manifest_frozen_v1.json`、`release/zenodo_v1/upload/source_manifest.json`，下同），改动即哈希失配。

### `disorderflow/data_factory.py`（604 行）
**角色**：state-contrast 设计的确定性、带溯源数据工厂（docstring 自述）。规范化序列（20 aa + X 字母表）、生成 observed/counterfactual 记录并以确定分片写出。
**关键项**：`SCHEMA_VERSION="statecontrast_factory_v1"`、`CDR_NAMES`（H1–L3）、`AA_ALPHABET`、`EXTERNAL_EVALUATION_TOKENS`（"caid"/"external_evaluation" 用于外评隔离）；`canonical_json`/`sha256_bytes`/`sha256_file`/`stable_id`（哈希溯源与 24 位稳定 ID）；`make_observed_record`（组装 observed_bound 父记录，映射六条 CDR 跨度）；`generate_counterfactuals`（打乱/单点突变/CDR 替换三类反事实）；`DeterministicShardWriter`（write 分片 + manifest）；`iter_phase3_source`/`iter_sabdab2_abag_source` 等来源迭代器。
**不可就地修改**：sha256 记录于 release 冻结清单；工厂记录的 record_id/provenance 哈希均由本文件逻辑生成，改动会破坏下游所有 lineage 校验。

### `disorderflow/datasets/_base.py`（40 行）
**角色**：数据集注册表（DiffAb 风格）。`register_dataset(name)` 装饰器把类登记进 `_DATASET_DICT`；`get_dataset(cfg)` 按 `cfg.type` 实例化并接 transform；`get_concat_dataset` 拼接多个数据集；`BalancedConcatDataset` 以最大子集长度做跨数据集均衡采样。不可就地修改：sha256 记录于 release 冻结清单，且所有 `@register_dataset` 入口依赖此契约。

### `disorderflow/datasets/lmdb_dataset.py`（49 行）
**角色**：`LMDBDataset`——只读、按需惰性连接的 LMDB 数据集，键为 8 位十进制序号（`f"{index:08d}"`），越界键回绕重试；`get_lmdb_dataset` 注册为 `lmdb_preprocessed`，明确忽略 transform（数据已在预处理时变换完毕）。不可就地修改：sha256 记录于 release 冻结清单，重排序或改键格式会使已构建 LMDB 的内容不可复现。

### `disorderflow/datasets/phase3_dataset.py`（99 行）
**角色**：Phase 3 跨链微调数据集——抗体-抗原复合物 LMDB（由 preprocess_phase3_dataset.py 构建），掩蔽全部 CDR（H1+H2+H3 或 L1+L2+L3）共设计、保留抗原结构作上下文、支持 `generate_flag` 跨链特征流。**关键项**：`Phase3Dataset.MAP_SIZE=4GB`；每 worker 独立开/关 LMDB 连接（readahead 关闭，注释说明是为 DataLoader 多进程正确性）；transform 失败时随机换样本重试。不可就地修改：sha256 记录于 release 冻结清单，改动会破坏 phase3 训练产物的可复现声明。

### `disorderflow/datasets/protein.py`（71 行）
**角色**：通用（非抗体概念）蛋白结构预处理。`preprocess_protein_structure(pdb_path, chain_ids)` 用 Bio.PDB 解析 PDB，逐链调 `parsers.parse_biopython_structure`，`cdr_flag` 全零（通用蛋白无 CDR 概念）；解析异常（PDBConstructionException/ParsingException 等）返回 None 而非抛出。供 PPI/通用设计 runner 使用。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/datasets/statecontrast_metadata.py`（33 行）
**角色**：`StateContrastMetadataDataset`——把已对齐的结构数据集与 StateContrast 标量记录逐条打包：长度必须一致，否则 ValueError；为每样本附加 `contrastive_group_id`（group_id 排序后索引）、`contrastive_rank`、`contrastive_weight` 张量。metadata 侧的轻量包装，与 structural 侧互补。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/datasets/confidence_dataset.py`（330 行）
**角色**：BFN 置信度头微调数据集。`ConfidenceRegressionDataset` 从 LMDB（或 pkl 目录回退）加载"BFN 预处理 batch + AF2 真值（pLDDT/ipTM/PAE 矩阵）"条目。**关键项**：`_acquire_lmdb`/`_release_lmdb` 引用计数环境缓存；`candidate_interface_v1` 模式强制校验 schema=`candidate_interface_confidence_v1`、非空 generate_flag、显式 Antigen 残基、pLDDT/PAE 覆盖全复合体（fail-closed）；`max_residues` 防 OOM 过滤、`min_af2_plddt` 质量过滤、`idp_sample_weight` IDP 加权采样、`scaffold_id` 供分组批采样器（V14+ 分组对比损失）；legacy 路径把全复合体 AF2 目标截断到抗体段并把 generate_flag 清零；`disorder_label` 优先 MobiDB-lite 逐残基标签，回退 pLDDT<50 / is_idp 广播；`set_max_residues` 支持训练中长度课程。不可就地修改：sha256 记录于 release 冻结清单；置信度监督的样本过滤与加权语义被已发布训练/评估产物引用。

### `disorderflow/datasets/custom.py`（216 行）
**角色**：自定义抗体结构目录数据集（`custom`）。`preprocess_antibody_structure` 解析 H/L/抗原链：重链按 Chothia Fv 截到 max_resseq=113、轻链到 106，再走 `_label_*_chain_cdr` 打 CDR 标；`CustomDataset` 用 32GB LMDB（`structure_cache.lmdb`）做并行预处理缓存（joblib 半核并行），增量处理新 PDB。不可就地修改：sha256 记录于 release 冻结清单；设计 runner（design_for_pdb 等）与推理缓存都复用该预处理语义。

### `disorderflow/datasets/custom_ppi.py`（153 行）
**角色**：单样本过拟合测试用的 PPI 数据集（`custom_ppi`）。`INTERFACE_DISTANCE_THRESHOLD=8.0Å`；`detect_interface_residues` 以 CA-CA 最小距离判定双链界面残基；`preprocess_ppi_structure` 解析 receptor（上下文）/ligand（待设计）两链并要求双侧界面非空；`CustomPPIDataset` 逐 PDB 即时解析、失败随机换样本。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/datasets/dips.py`（349 行）
**角色**：DIPS-Plus 通用 PPI 界面设计数据集（`dips`）。`parse_dips_dill_file` 解析 atom3 Pair 对象（df0/df1 双链 DataFrame），只保留 CA（pos_heavyatom 仅索引 1 有效，14 原子张量其余占位零）、`AA_NAME_TO_INDEX` 20 标准残基、未知记 20；8Å CA 阈值 `detect_interface_residues`，无界面则丢弃；`DIPSDataset` 遍历 `*.dill` 并走 LMDB 缓存。不可就地修改：sha256 记录于 release 冻结清单；CA-only 张量布局是下游模型输入契约。

### `disorderflow/datasets/sabdab.py`（489 行）
**角色**：SAbDab 抗体-抗原主数据集（DiffAb 血统）。**关键项**：`ALLOWED_AG_TYPES`（仅蛋白型抗原组合）、`RESOLUTION_THRESHOLD=4.0Å`、`TEST_ANTIGENS`（SARS-CoV-2 RBD、HIV-1 gp160、MERS S、流感 A、CD27 五个测试抗原名）；`_label_heavy_chain_cdr`/`_label_light_chain_cdr` 按 `ChothiaCDRRange` 打 cdr_flag 并写 H1–H3/L1–L3 序列，CDR3 长度 >max_cdr3_length（30）或缺失即整链判无效；`preprocess_sabdab_structure` 解析 H/L/抗原链（Fv 截断同 custom.py）；`SAbDabDataset` 32GB LMDB 缓存并按 split/抗原划分任务。不可就地修改：sha256 记录于 release 冻结清单；CDR 剔除与测试抗原划分规则被所有已发布训练集引用。

### `disorderflow/datasets/statecontrast_structural.py`（224 行）
**角色**：以权威父坐标（父 LMDB）+ data_factory 记录为准的 StateContrast 结构数据集（`statecontrast_structural`）。**关键项**：`exact_alignment_map`——无端点间隙全局比对（PairwiseAligner，match+2/mismatch−2/内部 gap −5/−0.5、端点 gap 0），仅保留等长且逐残基相同的比对块；`assign_official_imgt_cdrs` 强制官方 CDR 跨度可完整重构、在坐标中完全解析、映射连续且互不重叠（任一不满足即 ValueError，fail-closed）；`require_full_antigen_mapping` 要求解析抗原序列与官方序列完全一致；`load_factory_records` 按 manifest 校验 `schema_version`（默认 statecontrast_factory_v2_1）读 gzip JSONL 分片；`_ParentStore` 只读父结构 LMDB；每组必须恰有一个 `observed_bound` 父记录；`_replace_sequence` 反事实序列替换时同步清零侧链坐标/扭转。不可就地修改：sha256 记录于 release 冻结清单；该文件是 StateContrast 结构侧哈希校验与 fail-closed 语义的载体。

### `disorderflow/datasets/statecontrast_v2_structural.py`（64 行）
**角色**：StateContrast-v2 结构数据集（`statecontrast_v2_structural`）——继承 v1 结构集的 fail-closed 扩展。**关键项**：`STATE_TYPES = {"target":0, "apo":1, "off_target":2}`（状态枚举常量，模型与损失共用）；构造期要求每条记录 `state.source` 存在（出现 None 即 ValueError），并按"target 成员的 provenance.source 唯一"建 `group_source_ids` 供批均衡；`__getitem__` 附加 `state_type`/`pose_source_id`/`pose_prior_weight`/`state_sample_weight`/`state_valid`/`independent_teacher_score(+valid)` 等字段。不可就地修改：sha256 记录于 `release/zenodo_v1/upload/source_manifest.json`（release 清单链）。

### `disorderflow/datasets/statecontrast_v2_pose_manifest.py`（217 行）
**角色**：显式 pose manifest 直接读 PDB 的 StateContrast-v2 数据集（`statecontrast_v2_pose_manifest`），不需要 legacy 父 LMDB。**关键项**：`_parse_pose`（lru_cache 256）解析抗体（H 可选 L）+ 多抗原链合并 `_merge_chains`；`parse_record_structure` 支持抗原 `antigen_sequence_override`（线性映射替换并清零变化残基侧链/扭转）、按 `native_h3` 在重链序列中定位并重打 H3 cdr_flag、注入 `candidate_sequence`；构造期按 `teacher_group_id` 把 teacher 超组整组打包（保证 pair 比较同批）、要求每组同时含 target 与 apo/off_target（缺即 ValueError）、组内数据源必须唯一；支持 heldout_fold/fold_role 划分。不可就地修改：sha256 记录于 `release/zenodo_v1/upload/source_manifest.json`；分组完整性检查是 v2 对比评估不泄漏的守门逻辑。

### `disorderflow/disorder_calibration.py`（115 行）
**角色**：无序（disorder）logits 的事后 Platt 校准与操作点选择。**关键项**：`apply_platt`（sigmoid(scale·logit+bias)，float64）；`load_disorder_calibration` 校验 schema_version=1、scale>0、threshold 存在（不完整即拒绝）；`fit_platt` LBFGS+strong_wolfe 拟合加权 BCE（要求两类齐备、权重非负）；`calibration_metrics` 加权 Brier/ECE（10 bins）；`mcc_at_threshold`/`select_mcc_threshold` 遍历所有候选阈值（含相邻分数中点），同 MCC 时取最接近 0.5 者（确定性 tie-break）。不可就地修改：sha256 记录于 release 冻结清单；校准产物 JSON 的 schema 与确定性阈值规则被已发布评估引用。

### `disorderflow/disorder_metrics.py`（65 行）
**角色**：零依赖（仅 math）的残基级无序预测池化指标。**关键项**：`auc_roc`（并列分数取平均秩的 Mann-Whitney 实现，单类返回 NaN）；`auc_pr`（按分数降序的 Average Precision）；`pooled_disorder_metrics` 汇总 auc-roc/auc-pr/MCC/positive_rate/n_residues，并强制分数为 [0,1] 有限概率、长度一致（违者 ValueError）。不可就地修改：sha256 记录于 release 冻结清单；发布结果中的无序指标数值由该实现产出。

### `disorderflow/disorder_supervision.py`（114 行）
**角色**：残基级无序监督的验证与合并。**关键项**：`SOURCE_CONFIDENCE` 置信度表（disprot_experimental/mobidb_experimental=1.0，missing_density_proxy/nmr_ensemble_proxy=0.65，rmsf_proxy=0.55，external_predictor/afdb_plddt_ordered_proxy=0.25，charge_hydropathy_heuristic_v1=0.10）；`DisorderEvidence` frozen dataclass（values/mask/confidence/sources/cluster_id）；`normalize_evidence` 强制一维、[0,1]、mask/置信度形状一致、置信度 ∈(0,1]、cluster_id 非空（"leakage-safe supervision"）；`merge_evidence` 置信度加权平均合并、同序列必须同 cluster；`audit_cluster_splits` 检查 cluster 与精确序列不得跨 split，违例列入 errors（fail-closed 拆分审计）。不可就地修改：sha256 记录于 release 冻结清单；来源置信度权重与拆分泄漏审计是已发布无序监督的证据链组成部分。

### `disorderflow/models/__init__.py`（4 行）
**角色**：模型包子导出——`DiffusionAntibodyDesign`（扩散模型）、`get_model`（注册表工厂）、`AntibodyBFN`（BFN 生成模型）。不可就地修改：sha256 记录于 release 冻结清单，配置中 model 名解析依赖这些导出。

### `disorderflow/models/_base.py`（13 行）
**角色**：模型注册表。`register_model(name)` 装饰器 + `get_model(cfg)` 按 `cfg.type` 实例化；当前注册键有 `disorderflow`（扩散）与 `antibody_bfn`/`antibody_bfn_statecontrast_v2`。不可就地修改：sha256 记录于 release 冻结清单。
### `disorderflow/models/bfn_model.py`（191 行）
**角色**：BFN（Bayesian Flow Network）抗体设计主模型，注册名 `antibody_bfn`。**关键项**：`AntibodyBFN` 组合 `ResidueEmbedding`+`PairEmbedding`+`AntibodyBFN_Core`；`encode()` 用 context_mask（CA 可见且非 generate_flag）屏蔽待生成区，防结构/序列信息泄漏（structure_mask/sequence_mask）；`build_antigen_sequence_mismatch` 把批内抗原序列跨样本循环交换但保留目标几何——优先用预计算的 `hard_antigen_mismatch_aa`（来自 dev 数据集），否则批内动态构造并要求真正不同（valid 标志），供 `antigen_mismatch_rank` 损失构造"序列换、几何同"的困难负例；`loss_weight` 可配置各损失权重与 `confidence_head_kind`（默认 legacy_v12）；`forward` 训练、`sample` 采样、`score` 固定 t=0.5 评分、序列按 sqrt 长度重加权。不可就地修改：sha256 记录于 release 冻结清单；抗原失配预计算优先级与防泄漏编码语义是 successor-v3 系列 dev/评估结论的模型侧根基。

### `disorderflow/models/disorderflow.py`（143 行）
**角色**：扩散基线模型 `DiffusionAntibodyDesign`（注册名 `disorderflow`，DiffAb 血统）：Residue/Pair 嵌入 + `FullDPM` 扩散（SO(3) 旋转 + 位置 + 类别序列三通道去噪）。`encode` 返回 res_feat/pair_feat/帧 R/位置 p，`forward` 加噪训练、`sample` 从随机初值去噪、`optimize` 从固定 opt_step 加噪再退火（局部优化语义）。context_mask 同样屏蔽生成区防泄漏。不可就地修改：sha256 记录于 release 冻结清单；作为 BFN 的对照基线，其行为被消融/基线结果引用。

### `disorderflow/models/statecontrast_v2.py`（184 行）
**角色**：StateContrast-v2 模型扩展（注册名 `antibody_bfn_statecontrast_v2`），docstring 明言"不修改冻结的 BFN core"。**关键项**：`StateContrastV2Objective` 组合三损失——`grouped_state_contrast_loss`（目标态得分须超 apo/off_target，margin 可配）、`grouped_source_constrained_pose_weights`（pose 质量权重，按数据源设下限 minimum_source_mass=0.10、单 pose 上限 maximum_pose_weight=0.60，附 KL 正则）、`grouped_teacher_consistency_loss`（匹配独立冻结教师排序，tie_epsilon=1e-4 处理平局）；`AntibodyBFNStateContrastV2` 用 forward-hook 捕获 receiver 的事 factual logits，序列恢复 CE 只在 target 态记录上监督（注释说明：否则 apo/off_target 上下文会与对比损失互相打架）；`score_states` 提供评估用固定序列打分。off-target 状态强制参与（注释：仅 target-vs-apo 会被"检测抗原是否存在"作弊）。不可就地修改：sha256 记录于 `release/zenodo_v1/upload/source_manifest.json`；hook 捕获与 target-only 监督是 v2 结论有效性的关键不变量。

### `disorderflow/modules/__init__.py`（0 行）
**角色**：modules 子包空标记。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/modules/bfn/__init__.py`（0 行）
**角色**：BFN 模块子包空标记。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/modules/bfn/contrastive_losses.py`（53 行）
**角色**：匹配 observed/counterfactual 组的分组对比损失。`grouped_contrastive_margin(scores, group_ids, ranks, sample_weights, valid_samples, margin=0.5, min_rank_gap=1e-6)`：在每个完整组内对所有有序证据对做排序——二值秩保持旧行为，连续秩支持有序构象态；秩差 <min_rank_gap 的对（平局/近平局）被忽略，要求 margin 随目标秩差缩放（softplus(margin·Δrank − Δscore)），样本权重按对乘积归一。空组返回 0 梯度张量。不可就地修改：sha256 记录于 release 冻结清单；平局忽略与秩差缩放语义被已发布校准结论引用。

### `disorderflow/modules/bfn/core.py`（1146 行）
**角色**：`AntibodyBFN_Core`——BFN 生成核心与训练损失编排（仓库最大单文件之一）。**关键项**：`sequence_cross_entropy_20`（仅 20 个有效氨基酸类的逐残基 CE，ignore_index=-100，被 v2 模型复用）；`num_classes=22`、`beta`（BFN 精度）、`schedule`（linear/cosine）；组合 `CategoricalBFN`/`PositionBFN`/`OrientationBFN`/`SidechainBFN` 四个流 + `AntibodyBFN_Receiver` + `LieOTAlign`；seq-only 模式判定（无结构损失时 receiver 用真实骨架、预测置信度）；`_mask_antigen` 定义抗原上下文掩码（优先 fragment_type==Antigen，否则按 chain_nb 推断；fixbb 单链时为空）；置信度头系列（pLDDT/ipTM/PAE）与 confidence_head_kind 分派（legacy_v12 与 candidate_interface_v1–v3）；disorder/对比/失配排序等损失的加权执行。不可就地修改：sha256 记录于 release 冻结清单；损失键名与权重语义被全部已发布训练配置与结果引用。

### `disorderflow/modules/bfn/core_losses.py`（185 行）
**角色**：V14+ 方向 B 的组感知置信度损失（从 core.py 抽出以保持前向聚焦）。**关键项**：`candidate_antigen_pair_mask`（候选-抗原对掩码，可双向）；`within_group_pair_mask`（同 scaffold 组对掩码，None 时回退全批旧行为）；`grouped_std`（组内标准差均值——方差损失，把共享骨架的设计推向不同置信度，对抗常数输出失败模式）；`grouped_margin_ranking`（预测 ipTM 对真实 AF2 ipTM 的组内 margin 排序，带 SEM 噪声阈值 target_noise 过滤不可靠对——锚定到真实设计质量的监督信号）；`grouped_pair_difference`（组内分差回归，SEM 感知可靠性权重、smooth_l1、可选按目标 RMS 归一）；`within_protein_disorder_ranking`（蛋白内残基对排序，目标只决定方向不回归幅值，min_target_gap=0.05 排除近平局）。不可就地修改：sha256 记录于 release 冻结清单；这些损失的平局/噪声处理规则是置信度头评估结论的前提。

### `disorderflow/modules/bfn/orientation.py`（70 行）
**角色**：旋转（SO(3)）通道 BFN 流。**关键项**：`svd_project_so3`——SVD 投影到 SO(3)（加 1e-6 噪声破退化；MPS/XPU/CUDA 上强制 float32 并关 autocast，因 SVD/det 不支持 fp16）；`OrientationBFN.sample_theta`——theta(t) ~ N(α·R_clean, α)，F(t) 累积噪声旋转（均值 α·R_clean、方差 ∝α）；`get_rotation` 即 SVD 投影。不可就地修改：sha256 记录于 release 冻结清单；fp16 兼容处理是跨设备可复现性的组成部分。

### `disorderflow/modules/bfn/position.py`（56 行）
**角色**：位置通道 BFN 流。`PositionBFN`：prior 为零（N(0,I) 视角）；`sample_theta` 取 theta ~ N(α·x_clean, α)（"参数空间"信息累积的简化实现，注释保留推导）；`get_mean` 后验均值 theta/(α+1)。α=β·t 带多维广播适配。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/modules/bfn/receiver.py`（784 行）
**角色**：`AntibodyBFN_Receiver`——BFN 接收器（去噪网络主干）与置信度头编排。**关键项**：`aggregate_pair_contact_logits`——把残基对概率以 noisy-OR 聚合成"任一抗原接触"logits（log 空间稳定实现，无有效抗原时输出 -20）；`candidate_antigen_geometry`——候选-抗原距离的无坐标系摘要（计数、最小距离均值/标准差/四分位、<6/8/10Å 比例，各带归一化常数）；构造器校验 `confidence_head_kind` ∈ {legacy_v12, candidate_interface_v1/v2/v2_2/v2_3/v3} 且 candidate 系必须配 confidence_version v12（fail-closed）；序列/位置/角/骨架/时间五路嵌入 + 共享 `GAEncoder`；注释记录 Direction D（抗原→CDR 交叉注意力）与 Direction E（池化拼接条件化）的动机——probe 显示抗原链经 pair_feat 到达编码器但从未到达逐残基序列头，复杂物 vs fixbb 的 CDR 恢复率无差（Δ≈0 pp）；seq_only、disorder_head、contrastive_order_sensitive、disorder_condition_scale 等开关。不可就地修改：sha256 记录于 release 冻结清单；置信度头分派与接触聚合语义被候选界面置信度产物引用。

### `disorderflow/modules/bfn/receiver_heads.py`（78 行）
**角色**：可复用置信度/判别头模块（V18 从 receiver.py 抽出）。**关键项**：`ResidualMLP`（输入投影 → N 个残差块[Linear+LayerNorm+ReLU+Dropout] → 输出投影）；`AttentionPAEHead`——轻量 Q/K 外积注意力 PAE 头，pair_feat 经线性投影作注意力偏置，输出 (N,L,L) 逐对误差预测。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/modules/bfn/sequence.py`（76 行）
**角色**：序列（类别）通道 BFN 流。`CategoricalBFN`：`_alpha(t)` 支持 linear（β·t）与 cosine（β·(1−cos πt)/2，注释：早期 t 更平滑）两种调度；`sample_theta`——theta(t) ~ N(α·onehot(x), α·I)（高斯参数化类别分布）；`update` 累积 y 步；`probabilities` softmax(theta)。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/modules/bfn/sidechain.py`（86 行）
**角色**：侧链扭转角通道 BFN 流（GMM 参数化占位实现）。**关键项**：`SidechainBFN(num_angles=4, num_components=6)`——4 个 χ 角、6 分量高斯混合；prior 用均匀铺开的近似 rotamer 峰（注释：真实实现应从统计加载）；`sample_theta` 是简化占位（注释明言"placeholder for the complex GMM BFN forward process…to ensure the pipeline runs"），把均值按 PositionBFN 方式加噪；`get_angle` 取最大权重分量的均值。不可就地修改：sha256 记录于 release 冻结清单；其占位语义本身是被冻结的历史事实。

### `disorderflow/modules/common/__init__.py`（0 行）
**角色**：common 子包空标记。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/modules/common/geometry.py`（529 行）
**角色**：几何工具库（DiffAb 血统 + 本项目扩展）。**关键项**：`safe_norm`/`pairwise_distances`/`normalize_vector`/`project_v2v`；`construct_3d_basis`（CA-C-N 建局部标架）、`local_to_global`/`global_to_local`/`compose_rotation_and_translation`/`compose_chain`；`quaternion_to_rotation_matrix`/`quaternion_1ijk_to_rotation_matrix`/`repr_6d_to_rotation_matrix`（旋转表示互转）；`dihedral_from_four_points`、`knn_gather`/`knn_points`、`angstrom_to_nm`/`nm_to_angstrom`；`get_backbone_dihedral_angles`（骨架 φ/ψ/ω，末端由 `get_terminus_flag` 判定）；`reconstruct_backbone`（由帧/序列重建 N,CA,C,O，含肽平面几何）与 `reconstruct_backbone_partially`（只重建指定子集）；`compute_fape`（帧对齐误差）。不可就地修改：sha256 记录于 release 冻结清单；重建几何直接决定输出 PDB 的坐标合法性，被全部结构产物引用。

### `disorderflow/modules/common/layers.py`（160 行）
**角色**：基础网络层。**关键项**：`mask_zero`、`clampped_one_hot`（越界索引零向量）；`DistanceToBins`（高斯模糊距离分箱 0–20Å/64 bins，或 one-hot 模式，含溢出标志位）；`PositionalEncoding`（NeRF 风格 sin/cos 多频编码）；`AngularEncoding`（正负频率 sin/cos 角度编码，用于二面角）；自定义 `LayerNorm`（epsilon=1e-10）。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/modules/common/ot_align.py`（120 行）
**角色**：Lie 氏最优传输对齐（BFN 训练的 OT 损失组件）。**关键项**：`svd_project_so3_ot`——超稳健 SVD 投影：NaN/Inf 立即返回单位阵、范数缩放、三级扰动（1e-4/1e-2/1e-1）逐级重试、CPU float32 上做 SVD、全部失败回退单位阵；`LieOTAlign.sinkhorn`——log 空间稳定 Sinkhorn（等长时加 −0.1 对角偏置利于序列对齐、强掩码值 1e4 防 logsumexp 溢出）；`forward`——OT 传输计划 π 软对齐预测/真实坐标，Kabsch 式协方差 + SVD 求最优旋转，按 TM-score 公式（d0=1.24·(L−15)^{1/3}−1.8，下限 0.5）在对接空间计平均 TM。不可就地修改：sha256 记录于 release 冻结清单；数值稳定性回退层级是训练可复现性的一部分。

### `disorderflow/modules/common/so3.py`（146 行）
**角色**：SO(3) 旋转群工具（DiffAb 血统）。**关键项**：`log_rotation`（矩阵对数，梯度启用时 clamp 到 −0.999 稳定 acos 导数）、`skewsym_to_so3vec`/`so3vec_to_skewsym`/`exp_skewsym`/`so3vec_to_rotation`/`rotation_to_so3vec`（旋转向量⇄旋转矩阵）；`random_uniform_so3`（四元数均匀采样）；`ApproxAngularDistribution`——IPAB 血统的角分布近似：8192 bins 直方图预计算（球谐级数 1024 项），std ≤0.1 时切换高斯近似采样；`random_normal_so3`（随机方向 × 采样角度）。不可就地修改：sha256 记录于 release 冻结清单；加噪/去噪角分布决定扩散轨迹语义。

### `disorderflow/modules/common/structure.py`（77 行）
**角色**：帧级结构更新模块（供 graph 层用）。**关键项**：`FrameRotationTranslationPrediction`——由节点特征预测帧增量（旋转支持 quaternion 或 6d 表示 + 平移 3 维）；`FrameUpdate`——transition MLP 残差更新特征后预测 R_delta/t_delta，经 `compose_rotation_and_translation` 合成新帧，且仅在 mask_generate 处更新（`torch.where` 保留上下文帧）。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/modules/common/topology.py`（24 行）
**角色**：链拓扑标志工具。**关键项**：`get_consecutive_flag(chain_nb, res_nb, mask)`——残基 i 与 i+1 同链且 res_nb 差为 1 时为真（(B,L-1) BoolTensor，断链/边界处为假）；`get_terminus_flag`——由连续性标志推 N 端/C 端标志（padding 后每段首/尾为真），geometry.get_backbone_dihedral_angles 用它判定末端二面角有效性。不可就地修改：sha256 记录于 release 冻结清单。
### `disorderflow/modules/diffusion/__init__.py`（0 行）
**角色**：diffusion 子包空标记。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/modules/diffusion/dpm_full.py`（319 行）
**角色**：完整扩散去噪管线（`disorderflow` 扩散基线模型的核心）。**关键项**：`rotation_matrix_cosine_loss`（逐列余弦嵌入损失的旋转损失）；`EpsilonNet`——序列当前步嵌入（Embedding(25)）+ res_feat_mixer + GAEncoder，三头输出位置 ε、旋转增量（四元数→旋转矩阵）、20 类去噪序列分布（已 softmax），仅 mask_generate 处更新；`FullDPM`——组合 `RotationTransition`/`PositionTransition`/`AminoacidCategoricalTransition`，`forward` 随机 t 加噪并算 rot/pos/seq 三损失（序列用后验 KL）；`sample` 从均匀 SO(3)+标准正态+随机序列起步逐步去噪（traj 逐步搬 CPU 省显存）；`optimize` 从固定 opt_step 加噪后回退（局部优化）。位置统一 mean/scale 缓冲归一化。不可就地修改：sha256 记录于 release 冻结清单；作为基线的采样语义被消融结论引用。

### `disorderflow/modules/diffusion/transition.py`（223 行）
**角色**：扩散三通道的加噪/去噪转移核。**关键项**：`VarianceSchedule`——余弦调度（improved-DDPM 风格：f(t)=cos²(π/2·((t/T)+s)/(1+s))，s=0.01）派生 betas/alpha_bars/alphas/sigmas（上限 clamp 0.999）；`PositionTransition`——标准 DDPM 加噪与后验均值去噪（注释：clamp α 修首步不稳定问题，末步不加噪）；`RotationTransition`——SO(3) 上按 ApproxAngularDistribution 采角加噪（E·R 复合），去噪同理；`AminoacidCategoricalTransition`——均匀混合类别扩散（c_t = α·onehot + (1−α)/K），posterior 与多项式采样。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/modules/encoders/__init__.py`（0 行）
**角色**：encoders 子包空标记。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/modules/encoders/ga.py`（220 行）
**角色**：几何感知注意力编码器（Invariant Point Attention 血统，DiffAb）。**关键项**：`GABlock`——节点 Q/K 点积 + pair 偏置 + 空间 logit（局部坐标系查询/关键点经 local_to_global 后的距离平方，softplus 系数 γ）三类 logit 相加（×√(1/3)）后 softmax 成注意力；聚合 pair 特征、节点值、空间点特征（帧内距离/方向）后经 out_transform + 双 LayerNorm 残差；`_alpha_from_logits` 掩码外对数 logits；`GAEncoder`——堆叠 num_layers 个 GABlock，训练且需要梯度时用 `torch.utils.checkpoint` 省显存。不可就地修改：sha256 记录于 release 冻结清单；注意力归一化常数参与 checkpoint 兼容权重形状。

### `disorderflow/modules/encoders/pair.py`（109 行）
**角色**：残基对（L×L）特征嵌入。**关键项**：`PairEmbedding`——氨基酸对 Embedding（22×22）、同链相对位置 Embedding（±32 截断）、原子对高斯距离特征（softplus 的 aa-pair 距离系数 aapair_to_distcoef，零初始化）、骨架二面角 AngularEncoding，拼接后 MLP 输出；`sequence_mask` 把待生成残基换成 UNK 防序列泄漏；`structure_mask` 乘性屏蔽生成区距离/二面角特征，但 `generate_flag` 提供时放行跨链对（cross_chain，供表位条件化——注释说明）。不可就地修改：sha256 记录于 release 冻结清单；跨链放行与泄漏屏蔽语义影响所有已训练权重。

### `disorderflow/modules/encoders/residue.py`（92 行）
**角色**：残基级特征嵌入。**关键项**：`ResidueEmbedding`——氨基酸类型 Embedding（22 类）、按氨基酸位放置的局部坐标 one-hot 展开特征（max_aa_types×max_num_atoms×3）、骨架二面角 AngularEncoding（structure_mask 与相邻残基同时可见才保留——注释：防锚定残基二面角轻微泄漏）、`type_embed`（fragment_type：1=Heavy/2=Light/3=Ag 等 10 类）拼接 MLP。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/modules/statecontrast_v2.py`（184 行）
**角色**：分组构象数据的可微状态对比目标（纯函数层，无 nn 参数）。**关键项**：`TARGET_STATE=0`/`APO_STATE=1`/`OFF_TARGET_STATE=2` 常量；`source_constrained_pose_weights`——两级 softmax：先在唯一数据源之间分配质量（每源保底 minimum_source_mass=0.10），再在源内 pose 间分配（注释：加采样复本不会自动压过单个实验 pose），pose 权重上限 maximum_pose_weight=0.60，超限时向均匀分布混合（非硬裁剪，保 sum-to-one 与梯度），并返回对先验的 KL；实现强制 fp32（注释：bfloat16 下源保底与 KL 不稳定）；`grouped_state_contrast_loss`——target 态加权平均分须超 apo（softplus margin 0.10）与 off_target 的 worst-case（logsumexp 温度 0.25，margin 0.20），附 mean_target_gap 指标；`grouped_teacher_consistency_loss`——教师组内排序一致性，|teacher_gap| < tie_epsilon(1e-4) 的平局对忽略。不可就地修改：sha256 记录于 `release/zenodo_v1/upload/source_manifest.json`；平局 epsilon、源保底/上限数值是 v2 评估协议的一部分。

### `disorderflow/statecontrast_v2_contract.py`（71 行）
**角色**：显式 target/apo/off-target 训练 manifest 的校验助手。**关键项**：`VALID_STATES = {"target","apo","off_target"}`；`validate_statecontrast_v2_records(records, minimum_teacher_coverage=0.0)`——逐条校验 group_id 存在、state.type 合法、state.source 非空、prior_weight>0、targets.state_training_weight 存在；组级要求"target 且（apo 或 off_target）"才算完整；统计独立教师覆盖率（`independent_teacher_required` 默认 True 的记录中有 score 的比例），低于阈值记 errors；返回 valid/计数/组摘要/errors 结构。不可就地修改：sha256 记录于 `release/zenodo_v1/upload/source_manifest.json`；fail-closed 校验规则是 v2 manifest 门控的权威定义。

### `disorderflow/tools/__init__.py`（0 行）
**角色**：tools 子包空标记。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/tools/dock/__init__.py`（0 行）
**角色**：dock 子包空标记。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/tools/dock/base.py`（28 行）
**角色**：对接引擎抽象基类 `DockingEngine(abc.ABC)`——上下文管理器协议 + `set_receptor`/`set_ligand`/`dock() -> List[FilePath]` 抽象方法。`HDock`/`HDockAntibody` 的接口契约。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/tools/dock/hdock.py`（164 行）
**角色**：HDOCK 对接封装。**关键项**：`fix_docked_pdb`——把 HDOCK 输出的短 ATOM/HETATM 行（56 字符）补齐 occupancy/B-factor；`HDock`——临时目录内调 `hdock`/`createpl` 二进制，`_dump_complex_pdb` 把受体与对接配体逐模型拼成 complex_{i}.pdb；`HDockAntibody`——抗体化封装：`set_antibody` 先做 Chothia 重编号（renumber），`set_antigen(pdb, epitope_sites)`；lsite 固定写 Chothia CDR-H3 区间 `95-102:{重链}`（表位引导对接的约束侧），rsite 由 `DockSite(chain,resseq)` 列表生成； epitope_sites 为 None 时退化为无约束对接。不可就地修改：sha256 记录于 release 冻结清单；CDR-H3 lsite 语义是对接产物可复现性的依据。

### `disorderflow/tools/eval/__init__.py`（0 行）
**角色**：eval 子包空标记。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/tools/eval/__main__.py`（4 行）
**角色**：`python -m disorderflow.tools.eval` 入口，仅 `from .run import main`。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/tools/eval/base.py`（125 行）
**角色**：评估任务模型与扫描器。**关键项**：`EvalTask` dataclass——in_path/ref_path/info/structure/name/method/cdr/ab_chains/residue_first/residue_last/scores，提供生成/参考结构的 Bio.PDB 模型惰性解析与 `to_report_dict`；`TaskScanner`——按 `^\d+(_postfix)?\.pdb$` 模式与 REF1(_postfix).pdb 参考配对扫描结果目录，metadata.json 提供任务元数据（tag 匹配、跨 items 收集抗体链），shelve 数据库记录已访路径支持断点续评（visited 集合）。不可就地修改：sha256 记录于 release 冻结清单；目录布局约定（metadata.json 位置、REF1 命名）被已发布评估表引用。

### `disorderflow/tools/eval/energy.py`（43 行）
**角色**：PyRosetta 界面能评估。`pyrosetta.init` 固定参数（-mute all、-use_input_sc、-relax:default_repeats 2 等）；`pyrosetta_interface_energy` 用 `InterfaceAnalyzerMover`（pack_separated）返回 `dG_separated`；`eval_interface_energy` 从生成模型推断抗原链（非 ab_chains 的全部链），对生成/参考各算 dG 并写 `dG_gen`/`dG_ref`/`ddG=dG_gen−dG_ref`。不可就地修改：sha256 记录于 release 冻结清单；ddG 数值可比性依赖该固定 init 参数。

### `disorderflow/tools/eval/run.py`（66 行）
**角色**：评估主循环（`python -m disorderflow.tools.eval` 实现）。**关键项**：Ray 并行（每任务 1 CPU）依序跑 `eval_similarity`（+可选 `eval_interface_energy`）；`dump_db` 导出 summary.csv——abopt 模式（路径含 'abopt'）跳过 seqid≥100.0 的与野生型完全相同序列（注释：Antibody Optimization 模式忽略恒同序列）；shelve 轮询循环支持增量评估与断点。不可就地修改：sha256 记录于 release 冻结清单；abopt 过滤规则是优化实验汇总表的口径。

### `disorderflow/tools/eval/similarity.py`（125 行）
**角色**：结构相似度评估。**关键项**：`reslist_rmsd`——允许末端截断的最优对齐 RMSD（SD 动态规划：min(对齐 i-j, 跳过长链残基)，等效序列本地对齐后 CA-RMSD）；`entity_to_seq`/`reslist_seqid`（三字母→单字母，非标残基跳过）；`align_sequences`——Biopython pairwise2 全局 BLOSUM62（gap −10/−0.5，不罚末端 gap）返回百分比序列同一性；`extract_reslist` 按 (chain,resseq,icode) 区间取残基；`eval_similarity` 写 `rmsd`/`seqid` 分数。不可就地修改：sha256 记录于 release 冻结清单；rmsd/seqid 口径被设计基准表引用。

### `disorderflow/tools/relax/__init__.py`（0 行）
**角色**：relax 子包空标记。不可就地修改：sha256 记录于 release 冻结清单。
### `disorderflow/tools/relax/__main__.py`（4 行）
**角色**：`python -m disorderflow.tools.relax` 入口，仅 `from .run import main`。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/tools/relax/base.py`（117 行）
**角色**：弛豫任务模型与扫描器。**关键项**：`RelaxTask` dataclass——in_path/current_path/status（success/failed）与柔性区间（flexible_residue_first/last，(chain,resseq,icode) 元组），提供 tag 后缀路径管理（`_openmm`/`_rosetta`/`_fixbb` 管线标签）、`update_if_finished`（产物已存在且非空即跳过——幂等断点续跑）、空产物文件判失败并删除；`TaskScanner`——扫描 `^\d+\.pdb$|^REF\d\.pdb$`，metadata.json 提供区间，final_postfix 命中即视为完成。不可就地修改：sha256 记录于 release 冻结清单；文件命名/续跑约定被弛豫产物目录引用。

### `disorderflow/tools/relax/openmm_relaxer.py`（144 行）
**角色**：OpenMM 力场最小化弛豫（管线第一阶段）。**关键项**：`ForceFieldMinimizer(stiffness=10.0, tolerance=2.39 kcal/mol, platform CUDA)`——`_fix` 用 PDBFixer 补非标残基/缺失原子/氢（seed=0）；`_minimize` 用 amber99sb + HBonds 约束，CustomExternalForce 谐波势（k=stiffness）把柔性区间（含插入码比较 `_is_in_the_range`）以外的全部重原子钉在初始位置——只放生成区运动；能量单位 kcal/mol、长度 Å，产物 PDB 插入 INITIAL/FINAL ENERGY REMARK；`run_openmm` 幂等执行并写 `_openmm` 后缀文件。不可就地修改：sha256 记录于 release 冻结清单；弛豫约束范围语义决定下游能量评估可比性。

### `disorderflow/tools/relax/pyrosetta_relaxer.py`（189 行）
**角色**：PyRosetta FastRelax 弛豫（管线第二阶段）。**关键项**：`pyrosetta.init` 固定选项（与 eval/energy.py 一致）；`parse_residue_position`（解析 chain+resseq+icode）；`get_scorefxn` 按名称设置 corrections（beta_july15/beta_nov16/genpot/talaris，注释注明来源 gist）；`RelaxRegion(scorefxn='ref2015', subset='nbrs'|'target'|'all', move_bb)`——TaskFactory RestrictToRepacking（只 repack 不 design），柔性区间 ResidueIndexSelector + NeighborhoodResidueSelector，flip_subset 阻止区间外 repack，MoveMapFactory 开放区间骨架/χ 角，返回 pose 与弛豫前后能量；`run_pyrosetta`（允许骨架动，写 `_rosetta`）与 `run_pyrosetta_fixbb`（move_bb=False 固定骨架，写 `_fixbb`）。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/tools/relax/run.py`（85 行）
**角色**：弛豫主循环（Ray 编排）。**关键项**：三个 remote 包装（openmm 占 1/8 GPU、pyrosetta 占 CPU）；`pipeline_openmm_pyrosetta`（默认，OpenMM→PyRosetta 串行）、`pipeline_pyrosetta`、`pipeline_pyrosetta_fixbb`；`pipeline_dict` 名称分派；主循环扫描→提交→等待→sleep(1.0)，final_postfix（fixbb 或 rosetta）判定完成跳过。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/tools/renumber/__init__.py`（1 行）
**角色**：`from .run import renumber`——暴露 Chothia 重编号函数供 dock/runner 导入。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/tools/renumber/__main__.py`（4 行）
**角色**：`python -m disorderflow.tools.renumber in.pdb out.pdb` 命令行入口。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/tools/renumber/run.py`（84 行）
**角色**：抗体 Chothia 结构重编号（基于 abnumber）。**关键项**：`biopython_chain_to_sequence`（三→一字母，未知残基记 X）；`assign_number_to_sequence`——abnumber.Chain(scheme='chothia') 编号，Fv 序列必须是原序列子串（offset 校验），非 Fv 残基记 None；`renumber_biopython_chain`（残基 id 换成 (hetero, Chothia 序号, letter)；None 保留原样但不含入新链）；`renumber(in,out)` 逐链尝试，ChainParseError 链原样拷贝归入 other_chains；返回 (heavy_chains, light_chains)（可选 other）。design_for_pdb 与 HDockAntibody 依赖其判定重/轻链。不可就地修改：sha256 记录于 release 冻结清单；链类型判定与编号 scheme 是全部结构输入口径的一部分。

### `disorderflow/tools/runner/__init__.py`（0 行）
**角色**：runner 子包空标记。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/tools/runner/design_for_pdb.py`（291 行）
**角色**：单 PDB 设计 runner（DiffAb 的 design_for_pdb 移植）。**关键项**：`create_data_variants` 四种模式——`single_cdr`（逐 CDR 掩蔽重建，记录 residue_first/last 供评估）、`multiple_cdrs`、`full`（MaskAntibody 全抗体重设计）、`abopt`（从天然起步按 optimize_steps 做局部优化变体，命名 `-O{step}`）；`design_for_pdb`——可选先 Chothia 重编号并自动识别重/轻链（nanobody 忽略轻链），加载 config+checkpoint，metadata.json 写变体清单（eval/relax 扫描器依赖该布局），采样后 `reconstruct_backbone_partially` 重建生成区骨架，`apply_patch_to_tensor` 回贴到全上下文张量并加回 batch origin 平移，按 `%04d.pdb` 写设计、REF1.pdb 写参考；inference 变换 PatchAroundAnchor+RemoveNative（abopt 模式不移除天然 CDR）；`args_factory` 供程序化调用。不可就地修改：sha256 记录于 release 冻结清单；输出目录布局与 metadata 契约被评估/弛豫管线消费。

### `disorderflow/tools/runner/design_for_testset.py`（243 行）
**角色**：测试集批量设计 runner——与 design_for_pdb 共享同一 `create_data_variants`（逐字复制的实现），差异是用 `get_dataset` 从注册数据集（如 SAbDab 测试集）取结构而非单 PDB，并按 index 参数切片；其余采样/重建/保存逻辑相同（`%04d.pdb` + REF1.pdb + metadata.json）。不可就地修改：sha256 记录于 release 冻结清单；基准测试集评估依赖其输出目录与文件命名。

### `disorderflow/utils/__init__.py`（0 行）
**角色**：utils 子包空标记。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/utils/conformation_calibration.py`（187 行）
**角色**：构象柔性（RMSF）预测的共享校准指标。**关键项**：`kabsch_align`（CA 迹刚体对齐，含反射校正 det<0 翻转）；`per_residue_rmsf`——全部模型对齐到第一个模型后逐残基 RMSF（要求 (models,residues,3) 形状、≥2 模型 ≥3 残基、全有限值，否则 ValueError）；`sequence_index_map`（local 比对逐残基映射，仅同字母入表）；`safe_spearman`（<3 点或零方差返回 None）；`top_fraction_recall`（前 20% 集合重合率）；`rank_auc`（秩实现 ROC-AUC，免 scikit-learn）；`partial_spearman`（线性移除秩化控制变量的偏 Spearman）；`calibration_metrics`（spearman/top20_recall/flexible_auc + 不确定性两指标）；`calibration_verdict`——按阈值分类 `pass_physical_label`（median spearman≥0.30 且 top20≥0.30 且 partial≥0.15）、`rank_only`（≥0.15/≥0.25）、`fail_physical_label`、`insufficient_evidence`（<minimum_proteins=10），bootstrap（seed=2031, 2000 次）给出 95% CI——判定 AF2-seed RMSF 是否够格做物理监督标签。不可就地修改：sha256 记录于 release 冻结清单；verdict 阈值与随机种子是已发布校准结论的口径。

### `disorderflow/utils/homology_split.py`（120 行）
**角色**：同源聚类拆分工具（防泄漏拆分的权威实现）。**关键项**：`mmseqs_cluster`——MMseqs2 easy-cluster（默认 min_identity=0.3、coverage=0.8、cov-mode 0），可执行文件缺失时**拒绝**继续（RuntimeError："refusing an unsafe random split"——fail-closed 不退化为随机拆分）；`annotate_homology_cluster`（原文为 annotate_homology_clusters）就地为条目写 `<field>_cluster_id`；`grouped_train_val_split`——并查集把共享 pdb_id 或任一 cluster 字段的条目连成连通分量（组），固定 val/train ID 冲突即 ValueError，其余组按 `random.Random(seed).shuffle` + 组规模升序贪心填充验证集直至 `max(len(val_roots), round(N·val_ratio))`——任何组都绝不跨 split。不可就地修改：sha256 记录于 release 冻结清单；拆分防泄漏语义被全部数据集拆分审计引用。

### `disorderflow/utils/inference.py`（59 行）
**角色**：推理辅助。**关键项**：`find_cdrs`——按重/轻链 cdr_flag 返回 `H_CDR1…L_CDR3` 名称列表；`get_residue_first_last`——由 generate_flag 取生成区首末 (chain_id,resseq,icode)；`RemoveNative`——推理时把生成区序列替换为 UNK（remove_sequence）或以 N(0,10²) 噪声坐标替换生成区结构（remove_structure），防止答案泄漏进上下文。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/utils/protein/__init__.py`（0 行）
**角色**：protein 子包空标记。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/utils/protein/constants.py`（343 行）
**角色**：蛋白表示常量（DiffAb 血统）。**关键项**：`CDR` IntEnum（H1=1…L3=6）；`ChothiaCDRRange`——Chothia 六 CDR 的残基号区间（H1 26–32、H2 52–56、H3 95–102；L1 24–34、L2 50–56、L3 89–97）与 `to_cdr` 查询；`Fragment` IntEnum（Heavy=1/Light=2/Antigen=3，抗原掩码与模型 fragment_type 共用）；OpenMM 来源的非标残基替换表 `non_standard_residue_substitutions`（MSE→MET 等，含版权声明）；`ressymb_to_resindex`（A–Y→0–19，X=20）；`AA` IntEnum（20 氨基酸 + UNK=20，支持三字母与`_missing_`非标替换）；后续还有 `BBHeavyAtom`（骨架/重原子索引，N/CA/C/O/CB…）、`max_num_heavyatoms`、`restype_to_heavyatom_names`、`chi_angles_atoms`、`make_coordinate_tensors` 等（骨架索引 0–3 与 CB=4 的布局是全仓库张量契约）。不可就地修改：sha256 记录于 release 冻结清单；索引布局/CDR 区间被所有模型权重与数据集隐含依赖。

### `disorderflow/utils/protein/parsers.py`（142 行）
**角色**：Bio.PDB 结构 → BFN 张量字典的解析器。**关键项**：`ParsingException`；`_get_residue_heavyatom_info`（按 restype_to_heavyatom_names 填 max_num_heavyatoms×3 坐标与掩码）；`_get_residue_torsions`（chi_angles_atoms 四原子二面角，原子齐备才算）；`parse_biopython_structure(entity, unknown_threshold=1.0, max_resseq=None)`——链按 id 排序、残基按 (resseq,icode) 排序、要求 N/CA/C 齐备、UNK 计数超阈值抛异常；`res_nb` 连续编号：CA-CA ≤4.0Å 才 +1，链断裂时按 resseq 差或至少 +2 跳号；返回 (EasyDict 张量, seq_map{(chain,resseq,icode): 索引})。不可就地修改：sha256 记录于 release 冻结清单；res_nb 跳号规则决定 chain_nb/res_nb 拓扑语义。

### `disorderflow/utils/protein/writers.py`（78 行）
**角色**：BFN 张量字典 → PDB 写出器。**关键项**：`save_pdb(data, path)`——按 chain_nb 分链，`StructureBuilder` 逐残基逐原子写（restype_to_heavyatom_names 顺序、掩码过滤缺失原子、单字符原子名按 PDB 列宽补齐、可选 b_factor）；未知氨基酸类型告警跳过；屏蔽 BiopythonWarning；返回 structure 对象（path=None 时）。所有设计产物 PDB 由此写出。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/utils/runtime_checks.py`（186 行）
**角色**：BFN 置信度/设计管线的蛋白适配性运行时诊断（OOD 与 IDP 检测）。**关键项**：`Severity`/`Flag`/`DiagnosticReport`；阈值常量 `LENGTH_MIN=50`/`LENGTH_MAX=250`（训练分布）、`PLDDT_IDP_THRESHOLD=0.55`/`PLDDT_LOW_THRESHOLD=0.60`、`IPTM_IDP_THRESHOLD=0.20`/`IPTM_LOW_THRESHOLD=0.35`、`PAE_HIGH_THRESHOLD=12.0`/`PAE_WARN_THRESHOLD=8.0`；`diagnose_protein` 按长度/pLDDT/ipTM/PAE 逐项给 INFO/WARNING flag、维护 is_in_distribution 与 idp_likelihood（low/possible/high）；已知 IDP 注记引用实测"BFN pLDDT 无序区 r=0.21 vs 折叠区 r=0.48"（IDP blindness 披露）；`format_report` 转用户可读文本。不可就地修改：sha256 记录于 release 冻结清单；阈值是已披露适用域边界的量化表述。
### `disorderflow/utils/transforms/__init__.py`（11 行）
**角色**：变换注册表的门面导出——mask（MaskSingleCDR/MaskMultipleCDRs/MaskAntibody）、merge（MergeChains）、patch（PatchAroundAnchor）、ppi（MergePPIChains/MaskInterface/PatchAroundInterface）、mask_region（MaskRegion）、merge_protein（MergeProtein）、patch_protein（PatchProtein）及工厂 `get_transform`/`Compose`。配置文件中 `transform:` 列表经此解析。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/utils/transforms/_base.py`（78 行）
**角色**：变换基础设施。**关键项**：`Compose`（torchvision 替代品，注释说明为免重依赖；元组中间结果展开续传）；`register_transform(name)` + `get_transform(cfg)`（按 `type` 键实例化，其余键作构造参数）；`_index_select_data`/`_mask_select_data`——对首维等于残基数的张量/列表做索引/掩码选择（patch 变换的裁剪基础）。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/utils/transforms/align_utils.py`（78 行）
**角色**：patch 对齐工具。**关键项**：`kabsch_rotation(P,Q)`——SVD 最优旋转（det<0 修正）+ 两质心；`align_to_reference(pos_heavyatom, mask_heavyatom, fragment_type)`——用重链（Fragment.Heavy=1）前 20 个 CA 残基（保守框架区）构造刚体标架（首残基为原点、末-首为 x 轴、中点叉积定 z），返回对齐坐标与 Local→Global 变换（<3 残基时回退单位变换）。PatchAroundAnchor/PatchProtein 依赖它建立与全局一致的 patch 坐标系（patch_global_origin/rotation 字段）。不可就地修改：sha256 记录于 release 冻结清单；坐标框架决定采样结果的回贴正确性。

### `disorderflow/utils/transforms/mask.py`（221 行）
**角色**：CDR 掩蔽训练/推理变换。**关键项**：`random_shrink_extend`（数据增强：掩蔽区随机收缩 ≤1 / 延伸 ≤2 残基，保 min_length=5）；`MaskSingleCDR`（'mask_single_cdr'）——选择单条 CDR（含 'CDR3' 别名：H3 优先、无重链回退 L3），置 generate_flag 并在 CDR 两侧残基设 anchor_flag，随机选链/随机选 CDR 支持训练增强；`MaskMultipleCDRs`——同链多条 CDR 并集掩蔽；`MaskAntibody`（'mask_antibody'）——整条抗体重/轻链 generate_flag 全真，并在抗原上按 ≤6Å 接触抗原残基选一个 anchor（无接触则取最近）；`RemoveAntigen`（'remove_antigen'）——置空抗原。不可就地修改：sha256 记录于 release 冻结清单；掩蔽语义定义了"目标方向"训练任务。

### `disorderflow/utils/transforms/mask_region.py`（59 行）
**角色**：通用蛋白的区域掩蔽（'mask_region'，无 CDR 概念）。`regions: {chain_id: [0 基残基索引]}`——指定链上设 generate_flag，其余链全上下文；`anchor_mode='flanking'`（设计区两侧邻居作锚）或 `'full'`（全部设计残基作锚）；索引越界被过滤，空有效集退化为全零 flag。bfn_loader.parse_region_spec 的 1 基用户输入最终落到此处。不可就地修改：sha256 记录于 release 冻结清单；0/1 基约定是设计接口契约。

### `disorderflow/utils/transforms/merge.py`（104 行）
**角色**：抗体三链合并（'merge_chains'）。**关键项**：给 heavy/light/antigen 分别填 `fragment_type`（Fragment.Heavy/Light/Antigen，抗原 cdr_flag 清零）；`assign_chain_number_` 按 chain_id 全局排序编号；拼接全部张量属性（aa/pos_heavyatom/mask_heavyatom/generate_flag/anchor_flag/fragment_type/torsion 等）；缺失的可选字段给零默认（generate_flag/anchor_flag 零 bool、epitope_disorder_profile/disorder_confidence 零 float、disorder_supervision_mask 零 bool）；透传 cdr_type。不可就地修改：sha256 记录于 release 冻结清单；batch 键集合是模型输入契约。

### `disorderflow/utils/transforms/merge_protein.py`（59 行）
**角色**：通用蛋白链合并（'merge_protein'，无 heavy/light/antigen 语义）——`structure['chains']` 各链 fragment_type/chain_nb 先按链序填充、再按全局 chain_id 排序重映射，拼接标准张量属性后输出扁平 batch。与 ConfidenceRegressionDataset 文档中的 MaskRegion→MergeProtein→PatchProtein 管线配套。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/utils/transforms/patch.py`（91 行）
**角色**：抗体 patch 裁剪（'patch_around_anchor'）。**关键项**：`PatchAroundAnchor(initial_patch_size=128, antigen_size=128)`——以 anchor CA 为中心取最近 initial_patch_size 个残基 + 最近的 antigen_size 个抗原残基（抗体残基距离置 +inf 排除），generate_flag/anchor_flag 恒入 patch；无 anchor（全 Fv 生成、无抗原）时退化为整抗体并 mock 首残基锚；`_align` 经 align_to_reference 重定向坐标并把 Local→Global 变换存入 patch_global_origin/patch_global_rotation（注释含公式推导）；输出 `patch_idx` 供 apply_patch_to_tensor 回贴。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/utils/transforms/patch_protein.py`（116 行）
**角色**：通用蛋白空间 patch（'patch_protein'，可配置参考链，"no heavy-chain hardcoding"）。**关键项**：`reference_fragment_type` 指定设计链、`context_size` 取上下文近邻、`required_fragment_types`+`required_context_cap` 强制保留指定上下文链的最近 k 个残基（缺失即 ValueError——candidate_interface 置信度管线用此保证抗原上下文入 patch）；generate_flag/anchor 必入；kabsch 式 `_align`（前 20 CA 标架）；记录 patch_idx/patch_global_origin/patch_global_rotation。不可就地修改：sha256 记录于 release 冻结清单；必选上下文强制是置信度评估"目标方向"不变量的载体。

### `disorderflow/utils/transforms/ppi.py`（202 行）
**角色**：通用 PPI 界面设计变换（DIPS-Plus 数据格式）。**关键项**：`MergePPIChains(design_chain='ligand'|'receptor'|'both')`——receptor=上下文（fragment_type=2）、ligand=设计目标（fragment_type=0，借用 Heavy/Antigen 枚举语义），generate_flag 取界面掩码，锚取设计链界面中位残基；`MaskInterface`（'mask_interface'）——现只透传 design_chain/mask_ratio（实际掩蔽在 merge 中完成）；`PatchAroundInterface(patch_size=128)`——以生成区中位残基为中心取连续窗口（不足时贴边补齐），逐张量切片。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/utils/transforms/select_atom.py`（20 行）
**角色**：原子分辨率选择（'select_atom'）——`resolution='full'` 取全部重原子或 `'backbone'` 取前 5（N,CA,C,O,CB），写 pos_atoms/mask_atoms。不可就地修改：sha256 记录于 release 冻结清单。

### `disorderflow/utils/source_balanced_sampler.py`（70 行）
**角色**：数据源均衡且绝不拆开对比组的批采样器。**关键项**：`SourceBalancedCompleteGroupBatchSampler(dataset, max_batch_records, shuffle, seed)`——消费 `dataset.group_indices` 与 `dataset.group_source_ids`；构造期校验"组-源一一对应"与"最大组不超过批容量"（违者 ValueError）；`_ordered_groups` 源间轮转（round-robin）、`torch.Generator().manual_seed(seed+epoch)` 确定性洗牌（`set_epoch` 支持跨 epoch 变化）；组装批时不把组拆到两个批。供 StateContrast-v2 训练使用。不可就地修改：sha256 记录于 `release/zenodo_v1/upload/source_manifest.json`；"完整组同批"是组内排序损失成立的前提。

### `disorderflow/utils/train.py`（209 行）
**角色**：训练工具集。**关键项**：`get_optimizer`（adam/adamw，无 trainable 参数即报错）；`get_scheduler`（plateau/multistep/exp/cosine 重启/None→BlackHole）；`get_warmup_sched`（线性 warmup LambdaLR）；`log_losses`/`ValidationLossTape`（按样本数加权平均的验证损失累积）；`recursive_to`（张量/嵌套容器搬设备，失败回退 CPU——XPU/MPS 兼容）；`reweight_loss_by_sequence_length`（sqrt/linear 长度重加权）；`sum_weighted_losses`（仅加权配置中存在的损失键，注释：无权重键跳过——如评估指标）；`EMAModel`（decay=0.999 影子权重，apply/restore/averaged_state_dict）。不可就地修改：sha256 记录于 release 冻结清单。

---

## 二、modules/ 顶层推理模块（4 个文件）

### `modules/af2_jax_runner.py`（466 行）
**角色**：JAX 原生 AlphaFold2-Multimer 运行器，用于抗体-表位验证（教师标签/评分）。**关键项**：导入期以 `_MockModule` 假装 tensorflow（注释：AF 的 TF 代码只用于 MSA 特征生成，而本 runner 用纯 numpy 自建特征；Python 3.14 需要合法 `__spec__`）；`AA_ORDER='ARNDCQEGHILKMFPSTWYV'`（AlphaFold 规范序）、`RESTYPE_NUM=20`、`NUM_ATOM_TYPES=37`；`_load_model`——colabfold JAX 加载 `alphafold2_multimer_v3` model_1，单序列模式（num_msa=1/num_extra_msa=1）、关 distogram/masked_msa/experimentally_resolved 头、eval 无 dropout、关 bfloat16，权重缓存于 `~/.cache/colabfold/params/`，全局单例缓存；`build_multimer_features(ab_seq, epi_seq)` 构造 multimer 特征；`run_multimer_prediction` 返回 pLDDT/ipTM/PAE 等；`validate_antibody_epitope` 高层入口。它是 PAE 证据链教师侧的执行核心。不可就地修改：sha256 记录于 `results/pae_guarded_abba_v16/protocol.json`、`results/pae_queued_v19/protocol.json` 等 PAE 冻结协议（改动即教师标签哈希校验失效）。

### `modules/bfn_loader.py`（449 行）
**角色**：无 UI 依赖的 BFN 模型加载与设计/评分入口（webapp 与评估脚本共用）。**关键项**：`_detect_device`（CUDA>XPU>CPU）；`DEFAULT_MODEL_CONFIG=configs/demo_design.yml`、checkpoint 路径来自 `app_config.yaml` 或环境变量 `DISORDERFLOW_CHECKPOINT`；`_validate_candidate_interface_checkpoint`——candidate_interface_v1 架构要求 checkpoint 含全部 `.candidate_interface_` 权重，缺失即 RuntimeError（fail-closed，防止静默用旧头评分）；`load_bfn`（全局单例，ckpt 的 train.loss_weights 回填模型 loss_weight；head_iptm 形状不匹配时剥离后 strict=False 加载）；`parse_region_spec`——1 基用户区间（`B:26-33,51-58`）转 0 基 MaskRegion 索引（start<1 或 end<start 拒绝）；`build_region_batch`——preprocess_protein_structure + mask_region→merge_protein→patch_protein（可带 required_fragment_types 抗原上下文）→PaddingCollate，context/design 链重叠、缺链、空设计区均拒绝；fragment_type 重标（设计链=Heavy、抗原链=Antigen）；`inject_candidate_sequence`——严格长度与 20 字母表校验后把候选序列精确注入 generate_flag 位置；`_score_exact_candidate_confidence`（candidate_interface_v1 专用，PAE 范围=candidate_to_antigen）；`run_bfn_design`——采样 num_samples 个设计，逐样本报告 sequence/ppl/entropy/max_prob/plddt/plddt_std/iptm/pae，支持表位无序谱条件化（`epitope_disorder_profile` 写入 batch 并派生 disorder_pliability；注释明言模型 disorder 头不被推理支持、不作回退）；`sort_by` 置信度排序（注释记录历史教训：不排序时 OC 实验收的是前 N 个随机样本、ipTM≈0.05 的根因之一）。不可就地修改：sha256 记录于 release 冻结清单（`release/ecls_v1/source_manifest_frozen_v1.json`、`release/zenodo_v1/upload/source_manifest.json`）。

### `modules/peptide_conformer_ensemble.py`（140 行）
**角色**：局部肽段集合的几何与质控原语（H3 T1 集合分析的工具层）。**关键项**：`BACKBONE={N,CA,C,O}`；`kabsch_rmsd`（(N,3) 形状校验、det<0 反射修正）；`antibody_aligned_rmsd`——把抗体叠合到参考后、在同一抗体坐标系里分别返回抗体骨架 RMSD 与肽段 RMSD（刚体运动与肽段自身变化解耦）；`residue_contact_pairs`（4.5Å 重原子接触的残基对身份集合）；`contact_retention`（天然接触保持率；天然接触集为空即 ValueError）；`nonlocal_clash_count`（cKDTree 1.5Å 重原子冲突，排除同残基与同链相邻残基）；`ensemble_dispersion`（肽段拟合后两两 RMSD 的均值/最大值）；`conformer_passes`——八项门：能量有限、抗体保持（≤max_antibody_rmsd）、肽段位移上下限、天然接触保持、零非局部冲突、CA/CN 几何异常数不劣于参考，全部通过才算合格构象。不可就地修改：sha256 记录于 `results/publication/h3_submission_package_v1/evidence_manifest.json` 与 iBEC 提交包 `ecls_evidence_manifest.json`（改动即提交证据哈希失效）。

### `modules/peptide_t2_recovery.py`（30 行）
**角色**：T2 肽段扰动恢复实验的接触拆分与恢复度量。**关键项**：`split_contact_pairs(contact_pairs, record_id, restraint_fraction=0.5)`——接触对按 `sha256(record_id|pair)` 排序后确定性拆为 supplied（约束）与 held-out（保留验证）两集，至少 2 个接触才允许，保留集保证非空（`min(len-1,…)`）；`recovery_metrics`——rmsd_recovery_angstrom（初-末 RMSD 差）、held_out_contact_recovery（末-初保留接触数差）及两个正向布尔（rmsd_improved/held_out_contacts_improved）——目标方向显式编码在度量里。哈希键确保任意进程/平台上拆分完全一致。不可就地修改：sha256 记录于 `results/publication/h3_t2.1_contract.json`、`results/publication/h3_submission_package_v1/evidence_manifest.json` 与 iBEC 提交包（T2/T2.1 结论的度量定义）。
---

## 三、tests/ 冻结测试（25 个文件）

> 这些测试大多在校验发布后仍须成立的全局不变量（如 `test_release_lineage.test_current_release_lineage_is_valid` 直接断言当前仓库证据链哈希有效），并断言冻结产物/协议的具体数值（种子、阈值、门控状态）。修改测试或被测冻结物都会破坏"发布后行为未变"的证明。

### `tests/test_bfn_fixed_scoring.py`（142 行）
**守护的不变量**：BFN 固定序列评分（score_fixed）链路的精确语义。**关键测试**：`test_region_spec_converts_one_based_positions_exactly`（bfn_loader 1 基→0 基转换、'B:0' 拒绝）；`test_candidate_injection_is_exact_and_validates_length`（候选序列注入只改 generate_flag 位、长度不符拒绝）；`test_score_fixed_propagates_outputs_and_masks_padding`（用桩 receiver 验证 plddt/pae/disorder/contact/iptm/state_compatibility 逐位传播、padding 处清零——评分方向不被填充污染）；`test_model_score_uses_unmasked_embeddings_and_propagates_result`（model.score 必须 remove_structure/remove_sequence=False——完整上下文评分）；`test_pair_contact_noisy_or_aggregation_and_empty_antigen`（noisy-OR 接触聚合数值正确、空抗原恒 -20、bfloat16 饱和不产生 NaN）；`test_patch_protein_guarantees_required_context_fragment`（required_fragment_types 强制抗原上下文进 patch）。不可就地修改：sha256 记录于 release 冻结清单；它锁定置信度评分的数值契约。

### `tests/test_contact_v2_confirmatory_contract.py`（42 行）
**守护的不变量**：successor-v3 contact-v2 确认性评估的准入合同。`validate_admission(policy, manifest, …)` 必须要求：≥12 个独立同源组件（11 个即 "at least 12" 拒绝、component_id 重复即 "one representative" 拒绝）；manifest 分类为 `future_confirmatory_untouched`、`checkpoint_accessed_before_manifest_freeze=False`（出现过 checkpoint 访问即拒绝——"checkpoint-access"）、ready 未标记即拒绝。平局/边界（11 vs 12）被显式参数化验证。不可就地修改：sha256 记录于 release 冻结清单。

### `tests/test_h3_candidate_reranking.py`（34 行）
**守护的不变量**：H3 候选重排序基准的三个原语。`normalized_native_rank` 端点方向（天然最优=1.0、最差=0.0、中间=0.5）；`parse_mpnn_fasta` 只按 [2,3,4] 提取 H3 切段并解析 ProteinMPNN 头部 score；`unique_candidates` 去重且天然序列必须来自生成（native_generated 标志）、非法残基计入 failures。不可就地修改：sha256 记录于 release 冻结清单。

### `tests/test_h3_ecls_statistics.py`（21 行）
**守护的不变量**：ECLS（表位-CDR 似然分数）统计推断原语。`sign_flip_p` 对三个方向一致单位给精确双侧符号翻转 p=0.25（枚举所有翻转）；`leave_one_out_range`（留一法区间 [1.5, 2.5]）；`benjamini_hochberg` 保序（输入顺序不被重排）。锁定小样本精确检验的数值——发表统计口径的根基。不可就地修改：sha256 记录于 release 冻结清单。

### `tests/test_h3_epitope_delta.py`（56 行）
**守护的不变量**：H3 表位 delta 基准的评分方向与聚合。`sequence_nll` 只在指定 H3 位置上计 NLL；`percentile_lower_is_better`（NLL 越低百分位越高——目标方向编码）；`select_inference_units` 确定性选择推断单位（按 id 排序）；`bootstrap_mean_ci` 同种子完全可复现；`aggregate_inference_units` 单位内重复记录取平均。不可就地修改：sha256 记录于 release 冻结清单。

### `tests/test_h3_generator_calibration.py`（49 行）
**守护的不变量**：生成器（ProteinMPNN）自似然偏差的校准程序。`candidate_nnr` 证明 apo 权重可反转"生成器偏爱自己序列"的偏差（λ=0 时 NNR=0，λ=2 时 =1.0）；`aggregate_units` 单位内多种子平均；`choose_mpnn_coefficient` 选**最小**通过系数（≥0.95 平均 NNR 且 ≥0.90 top1）；`exact_sign_permutation_p` 精确符号置换检验；`benjamini_hochberg` 单调性。不可就地修改：sha256 记录于 release 冻结清单。

### `tests/test_h3_interface_contacts.py`（103 行）
**守护的不变量**：H3 界面接触提取器（`h3_interface_contacts` 顶层模块）的物理正确性。用带插入码的 toy 复合 PDB 验证：`extract_h3_interface` 按预期序列锚定 H3/肽段、保留插入码（icode='A'）、接触 ≤4.5Å 且链身份正确；`test_score_has_no_native_identity_input`——评分字典**不得含 identity_to_native 字段**（评分不得以天然序列同一性为输入，防循环论证）；链缺失/H3 序列不匹配即 ValueError（fail-closed）；`write_standardized_backbone` 产出连续 1..N 重编号且插入码残基映射可追溯。不可就地修改：sha256 记录于 release 冻结清单。

### `tests/test_h3_publication_package.py`（23 行）
**守护的不变量**：H3 发表包的完整性合同。`test_all_lightweight_package_inputs_exist`（PACKAGE_FILES 全部在库）；`test_temporal_final_hash_and_terminal_contract_are_valid`——temporal final 决断必须 `decision="accepted"` 且 `rerun_permitted=False`（一次性终局、禁止重跑）；`test_main_result_table_preserves_positive_and_negative_conclusions`——主结果表必须同时保留阳性结论（"positive one-time temporal final"）与阴性结论（"development gate rejected"）——负结果不许被丢弃。不可就地修改：sha256 记录于 release 冻结清单。

### `tests/test_h3_seqres_constructs.py`（6 行）
**守护的不变量**：`sequence_sha256` 稳定性（同输入同哈希、异输入异哈希）——5CSZ H3 SEQRES 构建体导出的序列身份凭证。不可就地修改：sha256 记录于 release 冻结清单。

### `tests/test_h3_t1_ensemble.py`（17 行）
**守护的不变量**：T1 集合基准的 `summarize_sequence` 必须用**共享参考**逐构象计 ECLS（对同一 reference 给出 [-1.0, +1.0] 的逐构象优势、均值 0、最差 1.0）——排除参考漂移造成的假优势。不可就地修改：sha256 记录于 release 冻结清单。

### `tests/test_h3_t1_ensemble_analysis.py`（5 行）
**守护的不变量**：`exact_sign_flip_p`（三个一致效应 → p=0.25）——T1 集合分析的小样本精确检验口径。不可就地修改：sha256 记录于 release 冻结清单。

### `tests/test_h3_t2_recovery.py`（17 行）
**守护的不变量**：`state_advantage` 必须**拒绝桩 runner 结果被当作真实优势**——即使 monkeypatch 掉 run_conditional_probs，天然与扰动状态的 ECLS 都必须为 0（无真实 runner 即无优势）。防止 T2 结论被伪造信号污染。不可就地修改：sha256 记录于 release 冻结清单。

### `tests/test_h3_t2_recovery_analysis.py`（5 行）
**守护的不变量**：T2 恢复分析的 `exact_sign_flip_p`（三一致效应 p=0.25）——与 T1/T2.1 分析共用同一精确检验口径。不可就地修改：sha256 记录于 release 冻结清单。

### `tests/test_peptide_conformer_ensemble.py`（75 行）
**守护的不变量**：肽段集合质控原语（modules/peptide_conformer_ensemble）的几何语义。抗体整体平移 +4 时 antibody_rmsd≈0 且肽段自身变化 >0.7Å（刚体运动与肽段变化解耦）；接触对按残基身份（非坐标）记录且保持率 1.0；相邻残基冲突不计入非局部冲突（3 残基样例恰好 1 处）；集合离散度可检测形状变化；**天然参考自带的几何异常不算失败**（ca_geometry_outliers ≤ reference 时 conformer_passes 通过——不因天然结构本身的离群而误杀构象）。不可就地修改：sha256 记录于 `results/publication/h3_submission_package_v1/evidence_manifest.json` 与 iBEC 提交包证据清单。

### `tests/test_peptide_h3_publication_split.py`（53 行）
**守护的不变量**：发表拆分的防泄漏规则。`cluster_tokens` 规范化官方聚类编码（字符串列表/'/'分隔/空）；`exclude_conflicts`——候选与训练侧在**任一轴**（抗体簇/抗原精确序列等）相撞即被剔除；`sequence_homology` 检出被覆盖片段（coverage=1.0）而不误报无关序列；`exact_alignment_map` 在官方 H3 片段相对解析坐标有插入时仍保持连续映射（锁 statecontrast_structural 同名函数的合同）。不可就地修改：sha256 记录于 H3 提交证据清单（results/publication/h3_submission_package_v1 与 iBEC 包）。

### `tests/test_peptide_h3_temporal_split.py`（42 行）
**守护的不变量**：时间拆分的准入规则。`metadata_eligible` 要求肽段沉积日期**严格晚于**训练截止（=截止日拒绝）、agtypes 必须是纯 PEPTIDE（PROTEIN/PEPTIDE 拒绝）且满足长度窗口——"训练截止后才公开"的时间隔离不因复合型标注放水；`metadata_record` 使用官方轴（official_ab_cluster/official_antigen_cluster/cdr_h3_sequence_exact）作为冲突检测键。不可就地修改：sha256 记录于 H3 提交证据清单。

### `tests/test_peptide_t2_recovery.py`（18 行）
**守护的不变量**：T2 度量（modules/peptide_t2_recovery）。`test_contact_split_is_deterministic_and_disjoint`——接触拆分必须确定（同输入两次调用结果逐集合相等）、supplied 与 held-out 不相交且并集等于全集（拆分溯源不丢不重）；`test_recovery_metrics_have_positive_direction`——rmsd_recovery=1.0>0、held_out_contact_recovery=0.4>0、两个方向布尔为 True（目标方向正确定义）。不可就地修改：sha256 记录于 H3 提交证据清单与 `results/publication/h3_t2.1_contract.json`。
### `tests/test_release_lineage.py`（52 行）
**守护的不变量**：发布谱系（lineage）验证器——全仓库证据链哈希的终极守卫。`collect_references` 支持结构化（`$.structured`）与 `source`/`source_sha256` 后缀两种引用形式；`sha256(path, "canonical_lf")`——CRLF 与 LF 检出产生**同一**规范哈希（跨平台结账不误报）；`run_validation` 递归检测陈旧哈希（引用哈希 ≠ 实际 → "hash mismatch"、status=invalid）；`test_current_release_lineage_is_valid`——**当前仓库的全部谱系文档实际校验通过（errors==[]）**；`test_source_only_release_lineage_is_valid`——SOURCE_ONLY_DOCUMENTS 子集亦通过。本注解文档开头"修改冻结文件会使证据链失效"正是由该验证器机械执行。不可就地修改：sha256 记录于 release 冻结清单。

### `tests/test_sabdab_successor_v3_snapshot.py`（159 行）
**守护的不变量**：SAbDab 快照获取与发现的可复现性。`normalize_pdb` 兼容三种 SAbDab 形式（pdb_000012QA / pdb_00007st8_H_L / 含空白）规范为小写 4+ 字符 PDB；`acquire_snapshot`——记录完整溯源（classification=metadata_only、row/instance/pdb 计数、重复实例行数、字节计数、最小/最大更新日期、请求 URL），**拒绝覆盖已有输出**（FileExistsError，write-once）；`discover` 精确差分——对基线实例/PDB、历史归档、肽段 H3 审计、v1 发现、v2 保留集、v3 开发集、固定 PDB、非肽段九类排除逐类核验非零，且保留同 PDB 的多实例（new-A-B 与 new-H-L 双候选），输出候选确定性排序，同样拒绝覆盖；输入各带 sha256。不可就地修改：sha256 记录于 release 冻结清单。

### `tests/test_successor_v3.py`（83 行）
**守护的不变量**：successor-v3 开发侧合同与声明边界。`build_antigen_sequence_mismatch`——交换抗原序列时保留支架段不动、目标几何布局不变（逐位断言）、单样本批返回 (None, None)；dev manifest 必须是 `frozen_development_only; not confirmatory`、代表数==组件数、`prohibited_overlap` 全假（与冻结集零重叠）；**源码级断言**：bfn_model.py 必须含 `batch['antigen_mismatch_pair_feat']`、core.py 必须含 `losses['antigen_mismatch_rank']`、训练配置必须含 `selection: H_CDR3`、`prohibited_initializer: stage_b` 与具体的 `allowed_initializer_sha256`（初始化白名单哈希被逐字锁定）；分析/谱系 JSON 必须维持 "exposed development only; not confirmatory" 分类、52 有效组件、bootstrap CI 上界 <0、contact auroc ≥0.60、声明边界含 "does not reverse the v2 terminal negative"。不可就地修改：sha256 记录于 release 冻结清单。

### `tests/test_successor_v3_confirmatory_contract.py`（94 行）
**守护的不变量**：确认性（confirmatory）评估协议的一次性合同。协议 YAML 冻结校验：status=frozen_before_live_snapshot_acquisition、主终点种子 [4103, 4111, 4127]、五轴隔离最小同一性（vh/vl 0.9、paired_cdr 0.7、h3 0.5、antigen 0.3）、`confirmatory_evaluation_attempts=1`（one-shot）、组件不足时动作以 `freeze_feasibility_failure` 开头（不许放宽阈值）；在冻结暴露结构上验证 contact_metrics ≥1 接触位、`write_canonical_pdb`→`record_batch`→`PaddingCollate` 的规范批路径（H/L/P 链、fragment_type=Antigen=2）；评估器**拒绝覆盖已存在的 attempt 文件**（"one-shot" FileExistsError）；可行性决断 JSON 必须记录 metadata_feasibility_gate_failed（PDB 上界 9 < 12 组件）、"without structure download"、"neither supports nor rejects" 的声明边界——门控失败被如实冻结而非隐藏。不可就地修改：sha256 记录于 release 冻结清单。

### `tests/test_successor_v3_exploratory.py`（239 行）
**守护的不变量**：探索性（exploratory）侧最大的合同测试束。发现确定性且排除参考（write-once）；future-readiness 门恰好 12 个新条目且 write-once（11=not ready / 12=ready 参数化）；`wsl_path` Windows→/mnt/ 映射；封存的探索结果 manifest 逐工件 sha256 复验（39 独立组件、描述性阈值对照 all_thresholds_passed=False——阴性如实保留）；`binary_metrics` 检测分数方向（auroc=1.0、正 logit 均值 > 负）与校准（brier ∈[0,1]）；参考联合扩展只标记已查看 ID、追加 SV3R 序列记录且 write-once；followup/policy 工件哈希复验、`pooling_prior_failed_snapshots: forbidden`（不许合并既往失败快照）；`assign_folds` 折间组件完整且 H3 均衡；`attach_hard_mismatch` 预计算的困难抗原失配优先于动态构造；contact-v2 registry 冻结时**不含湿实验结果**（measurements_present=False、panel=planned_not_executed）。不可就地修改：sha256 记录于 release 冻结清单。

### `tests/test_successor_v3_future_snapshot_runner.py`（38 行）
**守护的不变量**：未来快照计划的节奏门（cadence gate）。`build_plan` 依据冻结策略 YAML（successor_v3_future_snapshot_policy.yml）+ 冻结 SAbDab 快照 + reference_union_manifest_v3：在准许时刻**之前**（09:58:12 < 09:58:13.565916）必须 eligible=False 且 execute 抛 "Cadence gate closed"；恰过时刻（09:58:14）才放行并给出确定性输出目录名（…_2026_08_14）与 "no coordinate" 访问边界——未来发现的时间隔离不被提前触发。不可就地修改：sha256 记录于 `release/zenodo_v1/upload/source_manifest.json`。

### `tests/test_successor_v3_isolation.py`（81 行）
**守护的不变量**：隔离审计（audit_successor_v3_isolation）的确定性。精确 PDB 排除规范化 ID（PDB_00001ABC≡1abc）且排除顺序确定；五轴（vh/vl/paired_cdr/h3/antigen）命中经并查集传递连通成组件（[a,b,c] 与 [d,e]）；代表选取按固定序（接触位 → 接触对 → 分辨率 → id 决胜，输入任意排列结果不变）；`select_donor` 失配供体先取异组件、再长度差、再 id；`write_json_once` 拒绝覆盖已冻结输出；**源码级断言**：审计脚本不得 import torch、不得 import disorderflow.models、不得 import results——审计保持模型无关（model-free）与结果无关。不可就地修改：sha256 记录于 release 冻结清单。

### `tests/test_successor_v3_rcsb_acquisition.py`（69 行）
**守护的不变量**：RCSB 快照获取的查询冻结与精确发现。`QUERY_SPEC` 检索日期（2026-07-14）与排序（rcsb_id asc）被逐字断言——查询即协议；`query_sha256` 写入获取报告；输出 write-once（二次获取触发网络调用即 fail）；`discover` 依据获取清单 + SAbDab summary + reference_union（引用哈希需匹配 expected_reference_sha256）做精确差分，仅留新条目 3CCC，metadata_feasibility_passed=False（1 < 12 组件），输出 write-once。不可就地修改：sha256 记录于 release 冻结清单。
