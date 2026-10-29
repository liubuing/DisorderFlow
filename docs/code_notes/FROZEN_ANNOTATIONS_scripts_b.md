# 冻结脚本配套注解（二）：scripts/ 其余冻结脚本（按字母序）

> **为什么本文件存在**：本文件列出的 `scripts/` 脚本的 sha256 被冻结协议 JSON 记录（位于 `results/` 下各证据链目录的 `protocol.json`/批清单/发布包清单中）。**修改脚本文件本身会使已发布证据链的哈希校验失效**，因此这些脚本不可就地加注释——本文件是其只读配套注解。
>
> 覆盖范围：`frozen_list.txt`（249 项）中 `scripts/` 下、除 `FROZEN_ANNOTATIONS_scripts_a.md` 已覆盖的 v23–v30 类似物门控链与运行时脚本外的全部 114 个文件，按字母序排列。`disorderflow/`、`modules/`、`tests/` 的冻结文件本轮不做（另有安排）。
>
> 条目格式：路径与行数 → 角色与管线位置 → 关键函数/常量 → 冻结哈希所在位置（无法定位到具体目录时记为"results/ 证据链"）。

---

### `scripts/analyze_abeta_4hix_af2_gate.py`（324 行）
**角色**：Aβ/4HIX 线的候选门控器：对 H3 候选构象施加冻结的 AF2、PRODIGY、稳定性与多样性门控，输出排序与去冗余 shortlist。
**关键**：`validate_inputs()`（收据哈希校验）、`run_prodigy()`（外部 PRODIGY 亲和打分）、`fixed_rank_score()`（固定排序分）、`apply_gates()`、`hamming()`/`diverse_shortlist()`（序列汉明距离去冗余）、`export_shortlist()`。
**冻结**：冻结哈希记录于 results/ 证据链（Aβ prospective/publication 产物链）。

### `scripts/analyze_ecls_pae_evidence.py`（207 行）
**角色**：ECLS×PAE 证据的事后开发性诊断（docstring 明示 never opens frozen final/GP2 data）：相关性、成员资格、一致性、关联与成对稳定性分析。
**关键**：`stability()`（candidate×model×seed 网格，所有重采样按候选配对）、`teacher_value()`（读取冻结教师标签）、输出目录常量 `OUT = results/ecls_pae_evidence_dev_v1`。
**冻结**：sha256 记录于 `results/ecls_pae_evidence_dev_v1/` 协议；修改会使该诊断证据链哈希校验失效。

### `scripts/analyze_h3_ecls_statistics.py`（124 行）
**角色**：对已保存的 ECLS 结果做事后簇级 sign-flip 敏感性分析（H3 肽段课题的统计稳健性附注）。
**关键**：`sign_flip_p()`（双侧符号翻转 P 值：≤20 单位精确枚举、之上 Monte Carlo）、`leave_one_out_range()`、`benjamini_hochberg()`（FDR q 值）。
**冻结**：哈希被 `results/publication/` 发布包清单记录；修改会使发布产物溯源失效。

### `scripts/analyze_h3_generator_calibration.py`（320 行）
**角色**：generator-aware likelihood contrast 重排序的校准与报告（开发→主表→图的全流程），属 H3 课题 reranking 分析层。
**关键**：`choose_mpnn_coefficient()`（MPNN 系数选择）、`exact_sign_permutation_p()`（推断单元效应的配对符号翻转精确检验）、`benjamini_hochberg()`、`bootstrap_ci()`、`plot_summary()`。
**冻结**：哈希被 `results/publication/` 发布包清单记录。

### `scripts/analyze_h3_t1_ensemble.py`（166 行）
**角色**：汇总冻结的 T1 肽段集合（peptide-ensemble）开发基准：构象优势分析、CSV 表与汇总图。
**关键**：`exact_sign_flip_p()`（精确符号翻转检验）、`conformer_advantage()`、`analyze()`、`write_csv()`/`plot_summary()`。
**冻结**：哈希被 `results/publication/` 发布包清单记录。

### `scripts/analyze_h3_t2.1_recovery.py`（132 行）
**角色**：生成 T2.1 基准的统计量、表格与决策报告（T2 恢复实验的后续细化轮）。
**关键**：`load_results()`（读取冻结结果）、`cluster_table()`（按簇聚合的决策表）。
**冻结**：哈希被 `results/publication/` 发布包清单记录。

### `scripts/analyze_h3_t2_recovery.py`（146 行）
**角色**：为冻结的 T2 recovery 基准生成表格与图形（生成→恢复闭环的终点分析）。
**关键**：`exact_sign_flip_p()`、`analyze()`、`write_csv()`、`plot()`。
**冻结**：哈希被 `results/publication/` 发布包清单记录。

### `scripts/analyze_ibec_e1_generator_behavior.py`（302 行）
**角色**：分析 iBEC E1 的冻结 BFN 与 ProteinMPNN 生成器行为（多样性/还原率对比），产出解释性报告。
**关键**：`hamming_fraction()`、`position_entropy()`、`mean_pairwise_hamming()`、`native_recovery()`（天然序列还原）、`paired_bootstrap()`、`interpretation()`→`build_report()`。
**冻结**：冻结哈希记录于 results/ 证据链（iBEC 系列产物目录）。

### `scripts/analyze_successor_v3_contact.py`（214 行）
**角色**：审计 successor-v3 探索性接触标签、掩码与打分分离度（接触定义与模型可用性的前置审计）。
**关键**：`binary_metrics()`、`length_bin()`/`resolution_bin()`（分层切片）、`score_model()`、`context_difference()`。
**冻结**：冻结哈希记录于 results/ 证据链（successor_v3 探索阶段协议）。

### `scripts/audit_ibec_e2_idp_evidence.py`（288 行）
**角色**：审计冻结的 1,289 条记录构象数据集（LMDB confidence dataset）中的证据层级，为 iBEC E2 提供 Disorder/IDP 证据分类报告。
**关键**：`load_lmdb()`、`canonical_fingerprint()`/`sequence_sha256()`（指纹与收据）、`disorder_intervals()`/`interval_metrics()`、`classify()`（证据分级）、`validate_inputs()`（fail-closed）。
**冻结**：冻结哈希记录于 results/ 证据链（iBEC 系列产物目录）。

### `scripts/audit_pae_evidence_v20.py`（85 行）
**角色**：抗体公共池工作的只读源审计与可复现论文证据汇总：核对 `results/pae_queued_v19/`（SOURCE）的运行收据并产出 `results/pae_evidence_v20/`（OUT）证据摘要，是 v20 证据章节的机器凭证。
**关键**：常量 `SOURCE = results/pae_queued_v19`、`OUT = results/pae_evidence_v20`；单函数 `main()` 串联全流程。
**冻结**：sha256 记录于 `results/pae_evidence_v20/` 协议。

### `scripts/audit_pae_public_provenance_v9.py`（137 行）
**角色**：v9 溯源审计：把 Ridge 训练数据、生成器数据、短匹配与化学有效性分开（数据用途隔离），输入 `data/pae_public_crosscheck_v8`，输出 `data/pae_public_provenance_v9`。
**关键**：`best_overlap()`（双序列 ≥80% 覆盖的全部无空位偏移——即 v25 重用的 80% 覆盖规则的出处）、`STANDARD`（20 标准残基字母表，化学有效性判定）、`chemistry()`、`load_lmdb()`。
**冻结**：冻结哈希记录于 results/ 证据链（pae_public v8→v9 交叉核对链）。

### `scripts/audit_pae_public_sequences_v8.py`（135 行）
**角色**：v8 公共源元数据的序列分诊（triage），docstring 明示不声称结构或预训练独立性；输出写往 `data/pae_public_crosscheck_v8`。
**关键**：`read()/freeze()/digest()` 三件套（write-once 冻结语义，与 `prepare_pae_dense_v6` 同源）、`ungapped()`、`prepare()`、`run()`。
**冻结**：冻结哈希记录于 results/ 证据链（pae_public v8 交叉核对链）。

### `scripts/audit_pae_public_sources_v8.py`（107 行）
**角色**：版本化、无模型的公共源交叉核对（与严格 v7 分开）：固定 GitHub 数据集仓库与 revision 拉取清单并规范化比对。
**关键**：常量 `REPO = 'Zhaonan99/Antibody-antigen-complex-structure-benchmark-dataset'`、`REVISION = '6891e6e6...'`（固定 commit——可复现性的关键）；`normalize()`、`get_json()`。
**冻结**：冻结哈希记录于 results/ 证据链（pae_public v8 交叉核对链）。

### `scripts/audit_pae_public_structures_v8.py`（63 行）
**角色**：坐标来源 Chothia 注释完成后的序列重复隔离审计：确认结构导入未破坏 v8 序列分诊结论。
**关键**：常量 `PARENT = data/pae_public_crosscheck_v8`、`OUT = PARENT/structure_sequence_audit`；单函数 `main()`。
**冻结**：冻结哈希记录于 results/ 证据链（pae_public v8 交叉核对链）。

### `scripts/audit_successor_v3_exploratory.py`（233 行）
**角色**：对回溯性 successor-v3 探索队列做五轴隔离审计（ MMseqs2 敏感性版）：在 v8 式序列轴之外补充 MMseqs 搜索，防止精确匹配规则漏检同源暴露。
**关键**：常量 `DEFAULT_MMSEQS`/`DEFAULT_LIBRARY`（WSL 内固定路径的 mmseqs-avx2 二进制）、`mmseqs_version()`（记录工具版本）、`run_search()`、`wsl_path()/wsl_command()`（跨系统调用封装）。
**冻结**：冻结哈希记录于 results/ 证据链（successor_v3 探索阶段隔离审计链）。

### `scripts/audit_successor_v3_isolation.py`（353 行）
**角色**：successor-v3 确认性队列的无模型隔离审计：按五序列轴做精确 PDB 排除 + 身份阈值比对 + 连通分量（UnionFind）家族划分，选代表/供体结构并写出 FASTA。
**关键**：常量 `AXES`（身份阈值：VH/VL 0.9、paired CDR 0.7、H3 0.5、抗原 0.3）、`COVERAGE = 0.8`、`MINIMUM_COMPONENTS = 12`（可行性下限，fail-closed）；`write_json_once()`（write-once 语义）、`run_mmseqs_search()`、`set_sha256()`（产物自记录哈希）。
**冻结**：冻结哈希记录于 results/ 证据链（successor_v3 确认性隔离协议）。

### `scripts/benchmark_aayl_learnability.py`（233 行）
**角色**：AAYL 线的冻结、描述性"单突变→多突变可学习性"实验（`--prepare` 先于 `--run`；docstring 明示标签此前已暴露、非盲测）。
**关键**：常量 `SEEDS = [20260916..20260920]`（5 个日期型种子）、`COHORT = data/aayl_original_retrospective_v1/cohort.json`、`SCORES = results/aayl_original_retrospective_v1/scores.json`、`OUT = results/aayl_learnability_v1`；`top_weights()`（并列 top-k 边界的分数成员归属，与行序无关）。
**冻结**：sha256 记录于 `results/aayl_learnability_v1/` 协议。

### `scripts/benchmark_aayl_low_order.py`（164 行）
**角色**：AAYL 低阶突变基准（仅开发用）：单+双突变训练、三突变测试，并附标签审计。
**关键**：`select_train()`（确定性训练集选择）、`label_audit()`（标签质量审计）、`fit_predict()`；`OUT = results/aayl_low_order_v2`。
**冻结**：sha256 记录于 `results/aayl_low_order_v2/` 协议。

### `scripts/benchmark_ecls_pae_joint.py`（163 行）
**角色**：ECLS 与 PAE 特征的联合筛选基准（仅开发用，设计骨架上运行、不执行 AF2），评估两路特征联合后对天然峰的重排序能力。
**关键**：`screening()`（期望 top-20% 召回，并列边界用分数成员归属）、`joint_scores()`；`OUT = results/ecls_pae_joint_dev_v1`；含 `read()/digest()/write()` 冻结三件套。
**冻结**：sha256 记录于 `results/ecls_pae_joint_dev_v1/` 协议。

### `scripts/benchmark_h3_candidate_reranking.py`（462 行）
**角色**：H3 课题的跨生成器 ECLS native-spike 重排序基准（仅开发用）：同一骨架分别用 ProteinMPNN、BFN、ESM-IF 生成候选，统一打分池化后比较天然序列排名。
**关键**：`normalized_native_rank()`（并列感知归一化排名：1 最好、随机期望 0.5——主终点定义）、`generate_mpnn()/generate_bfn()/generate_esmif()` 三生成器、`load_esmif()`、`unique_candidates()`（去重）、`score_pool()/aggregate()`。
**冻结**：哈希被 `results/publication/` 发布包清单记录。

### `scripts/benchmark_h3_epitope_delta.py`（357 行）
**角色**：评估"肽段条件化"对 ProteinMPNN H3 似然的影响（开发复合物上，有/无肽段条件的 NLL 差）。
**关键**：`run_mpnn_state()`（固定骨架状态打分）、`sequence_nll()`（序列负对数似然）、`percentile_lower_is_better()`（低分位更优的百分位转换）、`select_inference_units()`、`bootstrap_mean_ci()`；常量 `MPNN_ALPHABET`（含 X）、`BACKBONE_NAMES`（N/CA/C/O）。
**冻结**：哈希被 `results/publication/` 发布包清单记录。

### `scripts/benchmark_h3_mpnn_sanity.py`（200 行）
**角色**：ProteinMPNN 骨架条件 H3 似然在已复核复合物上的 sanity 基准（实现与方向性的最小验证）。
**关键**：`prepare_reference()`（参考骨架准备）、`run_mpnn()`、`score_sequences()`、`benchmark_reference()`；常量 `MPNN_ALPHABET`。
**冻结**：冻结哈希记录于 results/ 证据链（H3 sanity 基准链）。

### `scripts/benchmark_h3_scorer_sanity.py`（229 行）
**角色**：CPU-only H3 界面打分器实现诊断：验证打分函数本身的行为（含组成保持对照下的表现）。
**关键**：`composition_shuffles()`/`_multiset_permutations()`（组成打乱对照——检验打分不是组成伪影）、`benchmark_reference()`、`write_report()`。
**冻结**：冻结哈希记录于 results/ 证据链（H3 sanity 基准链）。

### `scripts/benchmark_h3_t1_ensemble.py`（306 行）
**角色**：T1 肽段集合 ECLS 基准的执行体（仅开发用）：对每条记录生成/评分集合构象并聚合统计。
**关键**：`select_units()`（推断单元选择）、`run_conditional_probs()`（条件概率）、`summarize_sequence()`、`benchmark_record()`、`aggregate()`。
**冻结**：哈希被 `results/publication/` 发布包清单记录。

### `scripts/benchmark_h3_t2.1_recovery.py`（165 行）
**角色**：T2.1 v2 逐结构基准的编排器：跨 sealed-final（封存终版）记录运行恢复评测。
**关键**：`selected_hash()`（选中记录集合的哈希——可复现选择凭证）、`select_structures()`、`benchmark_record()`。
**冻结**：哈希被 `results/publication/` 发布包清单记录。

### `scripts/benchmark_h3_t2_recovery.py`（213 行）
**角色**：T2 扰动恢复与 ECLS 基准的执行体（仅开发用）：扰动后能否按 ECLS 恢复原状态。
**关键**：`state_advantage()`（状态优势统计）、`selected_hash()`、`benchmark_record()`、`aggregate()`。
**冻结**：哈希被 `results/publication/` 发布包清单记录。

### `scripts/benchmark_pae_revision_v4.py`（87 行）
**角色**：v4 PAE 代理线的耗时基准：度量从设计 PDB 到排序分数的墙钟时间（docstring 明示不含教师标签访问）。
**关键**：单函数 `main()`；耗时口径写死（排除教师标签 I/O），保证与后续 timing 线可比。
**冻结**：sha256 记录于 `results/pae_surrogate_revision_v4/` 协议。

### `scripts/benchmark_successor_v3_contact_baselines.py`（185 行）
**角色**：successor-v3 接触预测的基线对比：几何、序列特征与冻结编码器特征按组件折（component fold）跑交叉验证逻辑回归。
**关键**：`sequence_features()`、`cross_validated_logistic()`（按折 CV）、`metrics()`；常量 `AA`。
**冻结**：冻结哈希记录于 results/ 证据链（successor_v3 接触基线链）。

### `scripts/binder_confidence_v1.py`（432 行）
**角色**：binder 置信度特征的冻结器：把候选结合器的序列/结构特征固化为特征工件，并在**无实验标签时 fail-closed**（不产生任何训练声明）。
**关键**：`validate_inputs()`、`build_feature_rows()`/`write_feature_artifacts()`、`expression_labels()`/`bli_aggregates()`/`assay_and_candidate_labels()`（表达量与 BLI 结合实验标签）、`training_readiness()`（训练就绪门）、`analyze_measurements()`/`write_labeled_table()`。
**冻结**：冻结哈希记录于 results/ 证据链（binder 验证链）。

### `scripts/build/acquire_rcsb_successor_v3_snapshot.py`（171 行）
**角色**：successor-v3 数据获取第 1 步（RCSB 侧）：用固定查询从 RCSB search API 拉取抗体条目 ID 快照，确定性、write-once。
**关键**：常量 `ENDPOINT`、`PAGE_ROWS = 10000`、`QUERY_SPEC`（full_text "antibody" 组合查询）；`query_sha256()`（查询本身哈希入档）、`write_json_once()`、`fetch_page()/result_ids()`。
**冻结**：冻结哈希记录于 results/ 证据链（successor_v3 RCSB 获取协议）。

### `scripts/build/acquire_sabdab_successor_v3_snapshot.py`（127 行）
**角色**：successor-v3 数据获取第 1 步（SAbDab 侧）：下载 all-summary 快照并落盘采集清单；`CLAIM_BOUNDARY` 常量明示仅原始元数据采集、不做任何候选选择或模型推断。
**关键**：常量 `SOURCE_URL`（SAbDab all-summary API）、`MANIFEST_NAME = 'acquisition.json'`、`CLAIM_BOUNDARY`；`parse_summary()`、`acquire_snapshot()`。
**冻结**：冻结哈希记录于 results/ 证据链（successor_v3 SAbDab 获取协议）。

### `scripts/build/build_peptide_h3_publication_split.py`（473 行）
**角色**：柔性肽 H3 基准的"仅元数据"严格拆分审计（发表用）：八轴（PDB/官方簇/序列精确匹配等）查重 + 序列同源排除，产出带哈希的拆分清单。
**关键**：常量 `AXES`（八元组拆分轴）；`map_official_h3()`（官方 H3 簇对齐映射）、`sequence_homology()`（对称局部重叠同源性统计）、`exclude_homology_conflicts()`（后记录与任一先记录同源即排除）、`digest_ids()`/`write_manifest()`。
**冻结**：哈希被 `results/publication/` 发布包清单记录。

### `scripts/build/build_peptide_h3_temporal_split.py`（286 行）
**角色**：构建 ProteinMPNN 训练截止之后的时间切分柔性肽 H3 基准（防泄漏的时间维隔离）。
**关键**：`parse_date()`（时间切分依据）、`metadata_eligible()`、`scored_ids()`（已打分集合交集）、`extract_cifs()`、`preprocess_records()`。
**冻结**：哈希被 `results/publication/` 发布包清单记录。

### `scripts/build/build_release_artifact_bundle.py`（87 行）
**角色**：发布工程：把发布物打成确定性 ZIP（固定时间戳）并生成带校验和的外部清单。
**关键**：常量 `ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)`（消除 ZIP 内时间戳不确定性——字节级可复现的关键）；`digest()`、`expand_paths()`。
**冻结**：冻结哈希记录于 results/ 证据链（发布包构建链）。

### `scripts/build/build_release_source_manifest.py`（77 行）
**角色**：发布工程：生成带校验和的源码与轻量证据发布清单（哪些文件进 manifest 由后缀集合决定）。
**关键**：常量 `TEXT_SUFFIXES`（.py/.json/.md/.yml 等文本后缀）；`digest()`、`matches_any()`。
**冻结**：冻结哈希记录于 results/ 证据链（发布包构建链）。

### `scripts/build/build_successor_v3_confirmatory_reference.py`（267 行）
**角色**：构建 successor-v3 确认性隔离所用的冻结参考联合清单（v25 重叠审计的比对对象）：合并 CDR 注释与 phase3 验证集等来源，按规范化 ID 去重。
**关键**：常量 `DEFAULT_INPUTS`（multiscaffold_confirmatory_v2 的 CDR 注释、phase3_v5_1 val.lmdb 等）、`CDR_ORDER`；`deduplicate_records()`（确定性来源优先级）、`audit_exposed_pdbs()`、`write_json_once()`、`set_sha256()`。
**冻结**：冻结哈希记录于 results/ 证据链（successor_v3 确认性参考协议）。

### `scripts/build/build_successor_v3_contact_dev_dataset.py`（208 行）
**角色**：为已暴露（exposed）的 successor-v3 接触开发集构建按组件折划分的 LMDB（训练/开发数据落盘）。
**关键**：`assign_folds()`（组件折分配）、`select_donors()`、`attach_hard_mismatch()`（硬失配标记）、`write_lmdb()`。
**冻结**：冻结哈希记录于 results/ 证据链（successor_v3 接触开发数据协议）。

### `scripts/build/build_successor_v3_contact_v2_registry.py`（105 行）
**角色**：封存 successor-v3 contact-v2 工件注册表：把开发协议与未来确认政策等 config YAML 的哈希固化。
**关键**：常量 `ARTIFACTS`（`configs/benchmarks/successor_v3_contact_v2_development.yml`、`..._future_confirmation_policy.yml` 等到工件的映射）；`digest()`、单函数 `main()`。
**冻结**：冻结哈希记录于 results/ 证据链（successor_v3_contact_v2 注册表协议）。

### `scripts/build/build_successor_v3_development_manifest.py`（155 行）
**角色**：把已暴露的肽段/H3 记录冻结为 successor-v3 仅开发用折（dev-only folds），家族划分防泄漏。
**关键**：`connected_component_labels()`（连通分量家族标签）、`component_fold()`、`read_ids()`。
**冻结**：冻结哈希记录于 results/ 证据链（successor_v3 开发清单协议）。

### `scripts/build/discover_rcsb_successor_v3_snapshot.py`（124 行）
**角色**：successor-v3 候选发现（RCSB 侧）：从 RCSB 快照中排除当前 SAbDab 与全部冻结参考 PDB，得到新条目候选。
**关键**：常量 `REFERENCE_UNION_V3_SHA256`（参考联合清单的期望哈希——输入完整性断言）；`sabdab_pdbs()`、`discover()`、`write_json_once()`。
**冻结**：冻结哈希记录于 results/ 证据链（successor_v3 RCSB 发现协议）。

### `scripts/build/discover_sabdab_successor_v3_snapshot.py`（329 行）
**角色**：successor-v3 候选发现（SAbDab 侧）：对新采集快照与既有档案做精确差集（exact-diff），产出仅元数据的候选行；`CLAIM_BOUNDARY` 明示不做结构访问与任何确认性声明。
**关键**：常量 `FIXED_PDBS = {'3stb', '4hix'}`（固定纳入的两个锚点结构）、`CLAIM_BOUNDARY`；`instance_ids()`、`merge_candidate_rows()`、`discover()`。
**冻结**：冻结哈希记录于 results/ 证据链（successor_v3 SAbDab 发现协议）。

### `scripts/build/discover_successor_v3_exploratory.py`（155 行）
**角色**：构建确定性的回溯性探索 SAbDab 肽段队列（探索阶段的候选清单）。
**关键**：常量 `MAX_PDBS = 240`（探索队列规模上限）、`FIXED_PDBS = {'3stb', '4hix'}`；`ranking()`（确定性排序）、`discover()`。
**冻结**：冻结哈希记录于 results/ 证据链（successor_v3 探索队列协议）。

### `scripts/build/extend_successor_v3_reference_union.py`（146 行）
**角色**：把探索阶段新暴露的全部 ID 并入 successor-v3 参考联合清单（参考集单调增长、不删除），供后续隔离审计比对。
**关键**：`extend()`（只增不改的扩展逻辑）、`set_sha256()`、`normalize_pdb()`。
**冻结**：冻结哈希记录于 results/ 证据链（successor_v3 参考联合扩展协议）。

### `scripts/build/materialize_rcsb_candidate_interface_extension.py`（553 行）
**角色**：物化并接触过滤 RCSB 校准扩展候选：下载 mmCIF、按角色解析链、ANARCII 编号、按长度/接触阈值筛选组合并写出规范 PDB——v26 物化器在 RCSB 扩展线上的前身/同类。
**关键**：常量长度门（肽 5–50、重链 ≥90、轻链 ≥80、H3 4–30）、`HEAVY_LIGHT_PAIRING_CUTOFF = 8.0` Å、`MAX_COMBINATIONS = 4096`；`write_canonical_pdb()`（重命名链、不 deepcopy 父结构——与 v26 同一 OOM 规避手法）、`number_sequences()`、`combo_rank()`。
**冻结**：冻结哈希记录于 results/ 证据链（candidate_interface 扩展物化协议）。

### `scripts/build/materialize_successor_v3_confirmatory.py`（238 行）
**角色**：物化并接触过滤 successor-v3 确认性候选（确认性坐标门控）：解析链角色、计算接触度量、写出规范 PDB 与 write-once 元数据。
**关键**：常量 `CONTACT_CUTOFF = 4.5`（Å，接触截断——v26 物化器直接导入复用该常量以保持两线口径一致）；`contact_metrics()`、`write_canonical_pdb()`、`write_json_once()`。
**冻结**：冻结哈希记录于 results/ 证据链（successor_v3 确认性物化协议）。

### `scripts/build/resume_successor_v3_exploratory_materialization.py`（42 行）
**角色**：探索阶段物化的断点续跑器：从中断时已下载好的本地 CIF 目录恢复，避免重复网络请求。
**关键**：薄封装（42 行，单函数 `main()`），复用探索物化主脚本逻辑，仅重定向输入源。
**冻结**：冻结哈希记录于 results/ 证据链（successor_v3 探索物化协议）。

### `scripts/build_abeta_4hix_expression_package.py`（442 行）
**角色**：构建仅供评审的 3D6 scFv 表达与盲法 BLI 交接包：密码子优化级 DNA 设计、限制酶位点审计、H3 打乱对照构建与盲法孔板布局。
**关键**：常量 `CODONS`（密码子表）、`FORBIDDEN_DNA`（BsaI/BsmBI/SapI 酶切位点与多聚腺苷化信号——DNA 审计禁用序列）；`shuffled_h3()`（H3 组成打乱对照）、`replace_h3()`、`dna_audit()`、`assign_blind_ids()`/`plate_wells()`/`build_bli_layout()`（盲法交接）。
**冻结**：冻结哈希记录于 results/ 证据链（Aβ 表达包与 prospective 产物链）。

### `scripts/build_h3_publication_package.py`（296 行）
**角色**：ECLS/GCLC 研究的轻量、带校验和评审包构建器：汇集协议、手稿草稿、config、结果与复现说明进 `results/publication/`，并校验时间切分终版。
**关键**：常量 `PACKAGE_FILES`（纳入包的固定文件清单）；`package_manifest()`（逐文件 sha256 清单——本文件大批 H3 脚本哈希即由其记录）、`environment_report()`、`verify_temporal_final()`、`main_results()`。
**冻结**：sha256 记录于 `results/publication/` 发布包清单。

### `scripts/check_successor_v3_future_readiness.py`（63 行）
**角色**：把冻结的元数据门施加于未来的 exact-new 快照发现（前瞻可行性门）：判断新快照是否足以开展下一轮确认。
**关键**：常量 `MINIMUM_ENTRIES = 12`（最低条目数门槛，不满足即记录可行性失败而非放宽）；`check()`。
**冻结**：冻结哈希记录于 results/ 证据链（successor_v3 未来快照协议）。

### `scripts/crosscheck_rcsb_families_v21.py`（54 行）
**角色**：v21 家族刷新的补充核对：用 RCSB 抗体全文检索直接发现，补充 SAbDab 刷新可能遗漏的条目。
**关键**：常量 `OUT = data/pae_family_refresh_v21/rcsb_direct`；54 行薄脚本，单函数 `main()`。
**冻结**：冻结哈希记录于 results/ 证据链（pae_family_refresh_v21 数据链）。

### `scripts/curate_successor_v3_components.py`（151 行）
**角色**：从元数据与坐标接触指纹中遴选 exact-new 组件（双特异性 T 细胞接合器等固定精选集），做接触指纹 Jaccard 相似度核对。
**关键**：常量 `SELECTION`（逐 PDB 固定选择：12ER teclistamab、12ES linvoseltamab、22HG S2H97、26PF S54…含链角色与抗原标注）；`contact_fingerprint()`、`jaccard()`、`curate()`。
**冻结**：冻结哈希记录于 results/ 证据链（successor_v3 组件遴选协议）。

### `scripts/diagnose_aayl_failure_v3.py`（218 行）
**角色**：AAYL 线的冻结开发诊断与受控模型选择修复（docstring 明示：无新生物学声明、不获独立测试地位、旧结果保留）。
**关键**：常量 `SEED = 20260921`、`POLICIES = ['legacy_mse','wide_mse','wide_rank','shift_rank']`（四种损失/排序策略）、`ALPHAS`（Ridge 正则 9 点网格 1e-2…1e6）；`raw_audit()`/`sentinel_audit()`（哨点审计）、`coverage_and_distribution()`；`OUT = results/aayl_failure_v3`。
**冻结**：sha256 记录于 `results/aayl_failure_v3/` 协议。

### `scripts/diagnose_ecls_numeric_invariance.py`（73 行）
**角色**：ECLS 数值不变性的事后双精度诊断（docstring 明示不修改原始 pass/fail 结论）：检验打分对浮点精度的敏感性。
**关键**：73 行薄脚本，单函数 `main()`；仅读已保存结果、不重跑模型。
**冻结**：sha256 记录于 `results/ecls_hard_controls_scored_v1/` 协议。

### `scripts/discover_pae_families_v21.py`（85 行）
**角色**：v21 公共家族刷新：刷新公共元数据、扩大注释覆盖面（docstring 明示不扩大肽段长度范围）——v22 收割与 v23 门控链的元数据上游。
**关键**：常量 `OUT = data/pae_family_refresh_v21`；85 行薄脚本，单函数 `main()`。
**冻结**：冻结哈希记录于 results/ 证据链（pae_family_refresh_v21 数据链）。

### `scripts/enrich_rcsb_successor_v3_metadata.py`（95 行）
**角色**：为 exact-new ID 采集 write-once 的 RCSB 条目与聚合实体元数据（发现之后的富集步骤）。
**关键**：常量 `ENDPOINT = 'https://data.rcsb.org/rest/v1/core'`；`fetch_json()`（带收据的抓取）、`entity_record()`、`enrich()`。
**冻结**：冻结哈希记录于 results/ 证据链（successor_v3 元数据富集协议）。

### `scripts/eval/ablation_eval.py`（0 行，空文件）
**角色**：0 字节占位文件（2026-06-14 冻结时的原始状态）；同目录的实质实现位于未冻结的 `ablation_study.py`。保留占位以维持历史目录结构。
**冻结**：空文件哈希（e3b0c442…）同样被冻结清单记录；不可就地填充内容，否则哈希失效。

### `scripts/eval/aqp4_3gd8_af2_design.py`（0 行，空文件）
**角色**：0 字节占位文件；AQP4/3GD8 AF2 设计的实际工作在同目录未冻结脚本中进行。占位文件本身被冻结清单收录。
**冻结**：空文件哈希被冻结清单记录；修改（包括填充实现）会使哈希校验失效。

### `scripts/eval/aqp4_water_channel_design.py`（0 行，空文件）
**角色**：0 字节占位文件；实质实现在同目录未冻结的 `aqp4_water_channel_design_v2.py`。占位文件本身被冻结清单收录。
**冻结**：空文件哈希被冻结清单记录；修改会使哈希校验失效。

### `scripts/evaluate_mpnn_likelihood_baseline.py`（256 行）
**角色**：ProteinMPNN 固定骨架似然基线 vs AF2 界面 PAE：对 multiscaffold_confirmatory_v2 生成骨架（与候选生成、AF2 标注同一批骨架）上的每个实体（设计+天然+组成打乱）打条件对数似然。docstring 明确估计量：脚手架内 Spearman（MPNN H3 NLL vs AF2 实体均值界面 PAE，2 模型×3 种子）；并警示历史 v1 用的是 runner 全局界面字段、不可作同目标比较。
**关键**：`locate_h3()`（H3 区行索引）、`run_mpnn()/score_sequence()`、`spearman()`；常量 `AF2_FILES`（calibration_ext 的 af2/af2_model2 results.json）、`HOLDOUT_COMPONENTS = {'V2C001','V2C002','V2C003'}`、`WORK = results/candidate_interface_mpnn_baseline_v1`。
**冻结**：sha256 记录于 `results/candidate_interface_mpnn_baseline_v1/` 与 `results/ecls_hard_controls_scored_v1/` 证据链。

### `scripts/evaluate_pae_dense_v6.py`（59 行）
**角色**：v6 密集教师线的终点评估器：仅在完整 dense teacher 运行结束后评估冻结的 PAE 基线（先完成、后评估的时序门）。
**关键**：59 行薄脚本，单函数 `main()`；消费 `results/pae_dense_v6/` 内已验证的运行产物。
**冻结**：sha256 记录于 `results/pae_dense_v6/` 协议。

### `scripts/evaluate_successor_v3_confirmatory.py`（274 行）
**角色**：successor-v3 确认性评估的一次性（one-shot）执行器：加载冻结模型、在确认性折上评估并汇总，write-once 落盘。
**关键**：`load_frozen_model()`（冻结权重）、`unbatch()/record_batch()`（批记录核对）、`bootstrap_ci()`、`write_json_once()`（结果不可重写）。
**冻结**：冻结哈希记录于 results/ 证据链（successor_v3 确认性评估协议）。

### `scripts/evaluate_successor_v3_contact_v2.py`（192 行）
**角色**：在留出的 exposed 组件折上评估 contact-v2 检查点（开发侧模型选择依据）。
**关键**：`metrics()`、`component_metrics()`（按组件聚合，避免记录级伪重复）。
**冻结**：冻结哈希记录于 results/ 证据链（successor_v3_contact_v2 评估协议）。

### `scripts/evaluate_successor_v3_contact_v2_confirmatory.py`（199 行）
**角色**：contact-v2 的确认性评估：在未触碰的独立未来组件上做一次性评估，先验准入校验 fail-closed。
**关键**：`validate_admission()`（确认准入条件——数据必须来自未来快照）、`load_model()`、`paired_component_difference()`（配对组件差统计——主检验量）。
**冻结**：冻结哈希记录于 results/ 证据链（successor_v3_contact_v2 确认性协议）。

### `scripts/evaluate_successor_v3_development.py`（230 行）
**角色**：在冻结的 exposed 开发面板上评估 successor-v3 特异性模型（开发侧终点分析）。
**关键**：常量 `SEEDS = (3203, 3217, 3229)`（三个质数种子——推理集成种子）；`contact_labels()`、`component_bootstrap()`（组件级 bootstrap）、`summarize()`。
**冻结**：冻结哈希记录于 results/ 证据链（successor_v3 开发评估协议）。

### `scripts/evaluate_successor_v3_exploratory.py`（165 行）
**角色**：在回溯性探索面板上评估冻结模型（探索性、非确认性地位，与确认性评估严格分开）。
**关键**：`evaluate_exploratory()`；165 行薄脚本。
**冻结**：冻结哈希记录于 results/ 证据链（successor_v3 探索评估协议）。

### `scripts/fetch_release_artifacts.py`（69 行）
**角色**：发布工程的获取端：取回（或恢复）不可变的发布工件包并逐文件校验校验和——与 build_release_artifact_bundle 配对。
**关键**：`digest()`（校验和核对）、69 行薄脚本，单函数 `main()`。
**冻结**：冻结哈希记录于 results/ 证据链（发布包构建链）。

### `scripts/generate_h3_peptide_ensemble.py`（289 行）
**角色**：T1 基准的集合生成器：在已沉积复合物姿势周围做受约束的肽段集合采样（T1 生成侧）。
**关键**：`prepare_system()`（体系构建）、`sample_replica()`（副本采样）、`peptide_geometry()`（几何统计）、`generate_ensemble()`。
**冻结**：哈希被 `results/publication/` 发布包清单记录。

### `scripts/generate_h3_peptide_t2.py`（309 行）
**角色**：T2 基准的生成侧：产生中等扰动程度的肽段恢复集合（扰动→恢复闭环的扰动端）。
**关键**：`build_system()`、`run_stage()`（分阶段运行）、`frame_metrics()`（逐帧度量）、`generate_t2()`。
**冻结**：哈希被 `results/publication/` 发布包清单记录。

### `scripts/generate_h3_peptide_t21.py`（396 行）
**角色**：T2.1 生成侧：确定性扭转扰动恢复实验，含 4 臂对照（supplied/random/null 等体系），用 OpenMM 位置/接触约束控制扰动与恢复。
**关键**：常量 `CUTOFF = 1.0`（nm，约束/接触截断）；`_add_antibody_restraint()/_add_contact_restraint()/_add_peptide_position_restraint()`（三类约束）、`check_accepted()`（接受判据）、`run_recovery_arm()`、`_make_supplied_system()/_make_random_system()/_make_null_system()`（对照臂）。
**冻结**：哈希被 `results/publication/` 发布包清单记录。

### `scripts/generate_multiscaffold_confirmatory_v2.py`（397 行）
**角色**：生成全部冻结的 v2 多脚手架确认性候选尝试——docstring 明示**不做打分与选择**（生成与评估彻底分离）；含 BFN 重置与紊乱谱投影到抗原 patch 的条件化。
**关键**：`atomic_json()`（原子写 JSON）、`reset_bfn()`（每尝试重置生成器状态）、`disorder_profile()`/`project_profile_to_antigen_patch()`（紊乱条件化）、`attempt_rows()`、`canonicalize_structure()`。
**冻结**：冻结哈希记录于 results/ 证据链（multiscaffold_confirmatory_v2 生成协议）。

### `scripts/generate_pae_dense_v6.py`（30 行）
**角色**：v6 密集教师线的生成辅助：复用已验证的规范设计 PDB（30 行薄工具，单函数 `verified_canonical()`），docstring 明示避免整棵复制 Bio.PDB 层级（OOM 规避手法，与 v26 同源）。
**关键**：`verified_canonical()`——从验证过的运行目录取规范设计 PDB 而非重新解析。
**冻结**：sha256 记录于 `results/pae_dense_v6/` 协议。

### `scripts/guard_pae_batch_v16.py`（85 行）
**角色**：有界 WSL 冒烟守护工作器 v16（docstring：guard 失败时只杀自己拥有的工作器进程组）——v19 守护器的中间演化版，被 v16 ABBA 运行与 v17 短批运行调用。
**关键**：`prepare()`（切批并冻结协议）/`run()`（守护执行）两阶段，与后续版本同构；`OUT = results/pae_guard_smoke_v15`（冒烟输出）。
**冻结**：sha256 记录于 `results/pae_guarded_abba_v16/` 与 `results/pae_short_batches_v17/` 批协议。

### `scripts/guard_pae_batch_v18.py`（86 行）
**角色**：v18 屏幕批守护工作器（与 v15/v16 同一守护契约：有界运行、只杀自有进程组），被 `run_pae_screen_batches_v18` 调用。
**关键**：`write()/prepare()/run()` 三函数结构与 v16 几乎一致（86 vs 85 行，差异为批协议细节）；批协议内断言自身 sha256。
**冻结**：sha256 记录于 `results/pae_screen_batches_v18/` 批协议。

### `scripts/guard_pae_smoke_v15.py`（84 行）
**角色**：v15 冒烟守护工作器：守护机制的最小验证版（小批、有界、可快速验证 kill 语义），是 v16→v19 守护器谱系的起点。
**关键**：`write()/prepare()/run()`；`OUT = results/pae_guard_smoke_v15`。
**冻结**：sha256 记录于 `results/pae_guard_smoke_v15/` 冒烟协议。

### `scripts/harden_pae_surrogate_v4.py`（393 行）
**角色**：v4 PAE 代理强化实验（AF2 前开发实验；docstring 明示从不修改 v1–v3 冻结证据、AF2 矩阵只读作标签、GP2 测试不评估）：构建特征/标签、拟合标量头、做筛查式评估。**所有模型选择先在 protocol.json 冻结再建特征/标签**。
**关键**：常量 `SEEDS = [2041, 2053, 2069]`（质数种子）；`freeze_protocol()`（先冻结后执行）、`directional_target()`（仅教师标签路径，绝不进特征）、`screening()`（均匀随机平局下的期望召回）、`ScalarHead`（v2 的代数标量 PAE 分支、全新参数）、`paired_bootstrap()`。
**冻结**：sha256 记录于 `results/pae_surrogate_revision_v4/` 协议。

### `scripts/materialize_pae_public_v8.py`（55 行）
**角色**：v8 公共交叉核对链的坐标门：用 SAbDab 链角色对通过元数据分诊的存活者做坐标级核对。
**关键**：55 行薄脚本，单函数 `main()`；`OUT = data/pae_public_crosscheck_v8`。
**冻结**：冻结哈希记录于 results/ 证据链（pae_public v8 交叉核对链）。

### `scripts/materialize_successor_v3_coordinates.py`（84 行）
**角色**：在元数据就绪检查通过后物化 exact-new 的 RCSB PDB 坐标（successor-v3 坐标获取门）。
**关键**：`materialize()`、84 行薄脚本；`sha256()`（收据哈希）。
**冻结**：冻结哈希记录于 results/ 证据链（successor_v3 坐标物化协议）。

### `scripts/number_pae_direct_v21.py`（55 行）
**角色**：对 RCSB 关键词直接检索候选做基于序列的抗体角色复核（编号步，direct 分支）。
**关键**：55 行薄脚本；`OUT = data/pae_family_refresh_v21/rcsb_direct/numbering`。
**冻结**：冻结哈希记录于 results/ 证据链（pae_family_refresh_v21 数据链）。

### `scripts/number_pae_unlabelled_v21.py`（56 行）
**角色**：同 v21 direct 编号复核，处理无标签（unlabelled）分支的候选。
**关键**：56 行薄脚本；`OUT = data/pae_family_refresh_v21/rcsb_direct/numbering_unlabelled`。
**冻结**：冻结哈希记录于 results/ 证据链（pae_family_refresh_v21 数据链）。

### `scripts/pae_screening_next_v7.py`（215 行）
**角色**：v7 下一轮筛选：冻结既有 Ridge 筛选器、度量设计输入延迟、刷新新 ID（docstring 明示：不拟合 v6 标签、不跑新教师、成本重放为回溯性质）。
**关键**：常量 `OLD = results/pae_surrogate_revision_v4`、`DENSE = results/pae_dense_v6`、`OUT = results/pae_screening_v7`；`rank_ids()`（组件内仅候选的稳定选择）、`benchmark()`（延迟基准）、`discover()`（新 ID 刷新）。
**冻结**：sha256 记录于 `results/pae_screening_v7/` 协议。

### `scripts/pae_sequence_order_v5.py`（210 行）
**角色**：v5 暴露开发审计：检查生成器混淆与序列顺序效应（AF2 前；只读 v4 保留行、不加载 GP2）。
**关键**：常量 `ALPHAS = [0.1 … 1e4]`（Ridge 网格）；`order_features()`/`bigrams()`（序列顺序特征）、`by_scaffold()`/`within_arm()`（分层对照）、`choose()`（模型选择）、`permutation_diagnostic()`（置换诊断）；`OLD = results/pae_surrogate_revision_v4`、`OUT = results/pae_sequence_order_v5`。
**冻结**：sha256 记录于 `results/pae_sequence_order_v5/` 协议（并被 `results/pae_dense_v6/` 引用）。

### `scripts/parse_h3_structure_panel.py`（90 行）
**角色**：解析 H3 结构预测面板：用**绝对门控**（不按 ipTM 排名）筛选通过的结构。
**关键**：`parse_scores()`、`mean()`；90 行独立薄脚本（无仓库根常量，可独立运行）。
**冻结**：冻结哈希记录于 results/ 证据链（H3 结构面板链）。

### `scripts/pipeline/analyze_shortlist_variable_regions.py`（252 行）
**角色**：对 shortlist 序列做启发式 VH/VL 边界定位与继承 developability 标记的区域归属核查——docstring 明示仅为分诊用的 ANARCI 替代（识别常见 J 区基序、分离可变区与尾部恒定区/构建序列）。
**关键**：常量 `HEAVY_J_MOTIFS`（WGQGTLVTVSS 等）、`LIGHT_J_MOTIFS`（FGQGTKVEIK 等）；`find_variable_region()`、`motif_sites()`、`cys_sites()`、`flag_region()`。
**冻结**：冻结哈希记录于 results/ 证据链（pipeline 短名单分析链）。

### `scripts/pipelines/run_successor_v3_contact_v2_pipeline.py`（100 行）
**角色**：successor-v3 contact-v2 开发生命周期的单一入口：串联 build→train→evaluate 等子命令，并校验未来确认政策。
**关键**：常量 `CHECKPOINT`（固定检查点路径 `logs/bfn_successor_v3_contact_v2_exposed_2026_08_09__…/checkpoints/best.pt`）、`COMMANDS`（子命令→脚本与参数映射）；`verify_future_policy()`（确认政策校验）。
**冻结**：冻结哈希记录于 results/ 证据链（successor_v3_contact_v2 管线协议）。

### `scripts/plot_ecls_narrative.py`（199 行）
**角色**：从已保存分数与单一固定输入结构绘制证据导向的 ECLS 叙事图（天然状态图与排名图）。
**关键**：固定配色常量（`INK`/`TEAL`/`RUST`/`GOLD`/`GRAY`——发表图风格统一）；`draw_native()`、`draw_ranking()`、`build()`。
**冻结**：冻结哈希记录于 results/ 证据链（ECLS 图表链）。

### `scripts/prepare_aayl_remeasurement.py`（147 行）
**角色**：AAYL 重测量准备：恢复精确序列的源记录并准备临时的因子化试点（docstring 明示：不含任何检测提交、订购、测量、插补或跨批标签合并）。
**关键**：常量 `PANEL = results/aayl_failure_v3/measurement_repair_panel.json`（修复面板输入）、`SOURCE = data/abbibench_sequence_mapping_v1/original_alpha_seq`、`OUT = results/aayl_remeasurement_v1`；`source_recovery()`、`factorial()`、`pilot()`。
**冻结**：sha256 记录于 `results/aayl_remeasurement_v1/` 协议。

### `scripts/prepare_ecls_hard_controls.py`（92 行）
**角色**：ECLS 硬对照准备：构造"分数盲"的近天然对照并审计结构合格性（对照不接触打分器，防选择泄漏）。
**关键**：`near_native_controls()`（近天然对照构造）、`ca_coordinates()`（CA 坐标提取）；`OUT = results/ecls_pae_evidence_dev_v1`。
**冻结**：sha256 记录于 `results/ecls_pae_evidence_dev_v1/` 协议。

### `scripts/prepare_successor_v3_contact_v2_experiments.py`（177 行）
**角色**：为前瞻性 SPR/BLI 验证挑选多样化突变面板（湿实验交接前的实验设计）。
**关键**：`paired_t_power()`（配对 t 检验功效分析——面板规模的量化依据）；`sha256()`（输入收据）。
**冻结**：冻结哈希记录于 results/ 证据链（successor_v3_contact_v2 实验设计链）。

### `scripts/probe_pae_workers_v14.py`（95 行）
**角色**：v14 工作器探针：固定 ABBA 执行探针（docstring 明示不改 AF2 参数与候选），诊断 worker 在真实批次下的行为。
**关键**：常量 `SOURCE = results/pae_timing_v13/screened`（探针作业来源）、`OUT = results/pae_worker_probe_v14`；`status()/prepare()/run()` 三段结构。
**冻结**：sha256 记录于 `results/pae_worker_probe_v14/` 协议（timing 诊断链）。

### `scripts/qualify_pae_public_v9.py`（71 行）
**角色**：v9 资格门：对公共交叉核对链的全部成员施加化学合格性判定，同时**保留生物学分层**（家族标签不因筛除而丢失）。
**关键**：常量 `FAMILIES`（PUB001 HIV1 p24 表位模拟肽、PUB002 gp120 V3、PUB003/005 疟原虫 CSP NANP、PUB004 磷酸化 tau…——生物 stratification 的权威映射）；`SOURCE = data/pae_public_crosscheck_v8`、`OUT = data/pae_public_provenance_v9`。
**冻结**：冻结哈希记录于 results/ 证据链（pae_public v8→v9 溯源链）。

### `scripts/remeasure_pae_screen_v13.py`（101 行）
**角色**：v13 计时诊断：对 screened 臂做全新测量；参照系是重启后已完成的 full 臂（docstring 明示参考基准的选取）。
**关键**：`prepare()/monitor()/records()/run()` 四段（准备、监控、收据、执行）；`OUT = results/pae_timing_v13`、`OLD = results/pae_timing_v12`。
**冻结**：sha256 记录于 `results/pae_timing_v13/` 协议。

### `scripts/report_aayl_low_order.py`（97 行）
**角色**：AAYL 低阶基准 v2 的报告器：先核验已保存的候选预测，再写出完整的 v2 开发报告（验证先于陈述）。
**关键**：97 行薄脚本，单函数 `main()`；消费 `results/aayl_low_order_v2/` 产物。
**冻结**：冻结哈希记录于 results/ 证据链（aayl_low_order_v2 报告链）。

### `scripts/report_pae_revision_v4.py`（180 行）
**角色**：v4 PAE 代理修订文档生成器：**直接从实验工件构建修订文本而非字面量**（杜绝手抄数字与产物漂移）。
**关键**：180 行薄脚本，单函数 `main()`；消费 `results/pae_surrogate_revision_v4/` 全部实验产物。
**冻结**：sha256 记录于 `results/pae_surrogate_revision_v4/` 协议。

### `scripts/resume_pae_dense_batched.py`（94 行）
**角色**：v12/v13 计时线与 v10/v11 试点共用的"仅传输层"修正运行器：给 AF2 worker 加批次上界与 WSL 本地 stdin 重定向以恢复长跑；docstring 明示模型、种子、recycle、选择与评估代码全部不变。
**关键**：常量 `BATCH_SIZE = 96`（每批作业数）、`REAL_POPEN = subprocess.Popen`（保留原 Popen 引用再包一层）；`BatchedWorker` 类（有界批处理 worker）、`popen()`（拦截包装）。
**冻结**：sha256 记录于 `results/pae_timing_v12/`、`results/pae_timing_v13/` 与试点目录协议。

### `scripts/run_aayl_learnability.py`（16 行）
**角色**：兼容性入口：让冻结的 ESM 可学习性实验（`benchmark_aayl_learnability`）能在 PyTorch ≥2.6 环境运行（仅环境适配，不改实验语义）。
**关键**：16 行薄入口。
**冻结**：冻结哈希记录于 results/ 证据链（aayl_learnability_v1 链）。

### `scripts/run_abeta_4hix_af2.py`（294 行）
**角色**：Aβ/3D6 线的冻结多种子 AF2-Multimer 分诊运行器：对 H3 shortlist 逐候选组装实体并跑 AF2（含 H3 打乱对照）。
**关键**：`build_entities()`（三链实体组装，被 v28 复用的同型实现）、`windows_to_wsl()`（跨系统路径转换）、`shuffled_h3()/replace_h3()`（组成打乱对照）、`validate_contract()`（输入契约校验）、`atomic_json()`（原子写）、`run_chunk()`（分块执行）。
**冻结**：冻结哈希记录于 results/ 证据链（Aβ prospective 产物链）。

### `scripts/run_abeta_4hix_prospective.py`（500 行）
**角色**：Aβ/4HIX-3D6 线的冻结前瞻性保守 H3 候选协议（500 行全流程）：保守替换文库生成 → BFN 行 → 打分 → 可开发性标注 → 门控 → 多样性 shortlist 导出。
**关键**：常量 `CONSERVATIVE_GROUPS`（六组保守替换：AVLIM/FYW/STNQ/DE/KRH/GAS）、`HYDROPHOBIC`/`POSITIVE`/`NEGATIVE`；`nglyco_count()`（N-连接糖基化计数——可开发性门）、`allowed_substitutions()`、`conservative_library()`、`apply_gates()`、`export_shortlist()`（docstring：导出可审计分诊文件，非合成规范）。
**冻结**：冻结哈希记录于 results/ 证据链（Aβ prospective 产物链）。

### `scripts/run_guarded_abba_v16.py`（75 行）
**角色**：v16 ABBA 守护复刻：对被打断的 v14 工程探针在有守护的条件下复跑（工程对照实验）。
**关键**：常量 `OUT = results/pae_guarded_abba_v16`；`prepare()/status()/run()` 三段结构；`sha()/read()/save()` 基础工具。
**冻结**：sha256 记录于 `results/pae_guarded_abba_v16/` 协议。

### `scripts/run_ibec_tau_candidates.py`（602 行）
**角色**：iBEC Tau 线的冻结候选协议（correct-mask、仅 H3 可变，602 行全流程）：保守文库 + ProteinMPNN 生成 + BFN 对照 + 姿势接触打分 + 可开发性标注。
**关键**：常量 `CONSERVATIVE_GROUPS`（六组保守替换，与 4HIX 线同口径）；`tau_pose_rows()`、`build_fixed_positions()`（仅 H3 可变的固定位）、`run_proteinmpnn()/run_bfn_control()`（双生成器）、`build_pose_contact_maps()/contact_score_for_h3()`（接触打分）、`developability_row()`。
**冻结**：冻结哈希记录于 results/ 证据链（iBEC 候选协议）。

### `scripts/run_ibec_tau_sanity.py`（224 行）
**角色**：Tau/5MP3 目标的冻结结构与 BFN 路径 sanity 检查（候选协议前的最小正确性验证）。
**关键**：`selected_tau_poses()`（选定姿势）、`structural_checks()`（结构检查）、`score_poses()`（打分核验）。
**冻结**：冻结哈希记录于 results/ 证据链（iBEC sanity 链）。

### `scripts/run_multiscaffold_v2_af2.py`（359 行）
**角色**：v2 多脚手架线的冻结多种子 AF2-Multimer 评估运行器：对 v2 选择与对照（含 H3 打乱）统一跑 AF2。
**关键**：`build_entities()`/`windows_to_wsl()`/`shuffled_h3()`/`atomic_json()`（与 4HIX AF2 运行器同型）；`run_chunk()` docstring：单个持久 WSL worker 摊销 JAX/模型编译——后续 wsl_batch 架构的前身。
**冻结**：sha256 记录于 `results/pae_public_pilot_v10/`、`results/pae_public_pilot_v11/` 证据链。

### `scripts/run_pae_dense_v6.py`（136 行）
**角色**：v6 密集教师线的 AF2 teacher worker：可断点续跑、为长时本地 GPU 运行写进度。
**关键**：`key()`/`target()`（作业键与目标路径）；136 行薄 worker；进度状态直写、已验证产出跳过。
**冻结**：sha256 记录于 `results/pae_dense_v6/` 协议（v10/v11 试点亦引用）。

### `scripts/run_pae_public_pilot_v10.py`（323 行）
**角色**：v10 公共池回溯筛选试点（docstring 明示：有界回溯试点，**绝非独立验证**）：从 v9 合格候选选槽 → 生成 → 组装实体 → AF2 → 方向性教师标签 → 冻结 Ridge 评估；v11 的前身。
**关键**：常量 `SOURCE = data/pae_public_provenance_v9/qualified_candidates.json`、`WEIGHTS = results/pae_screening_v7/ridge_fixed10.npz`（冻结 Ridge 权重 alpha=10）、`SEEDS = [20260918, 20261918, 20262918, 20263918]`（4 个日期型变体种子）；`teacher()`（方向性 PAE 教师标签）、`screening_metrics()`、`verify()`。
**冻结**：sha256 记录于 `results/pae_public_pilot_v10/` 协议。

### `scripts/run_pae_screen_batches_v18.py`（72 行）
**角色**：v18 全冻结 screened 队列的批运行器：每守护 worker 六个作业（批量口径的定型版）。
**关键**：常量 `SOURCE = results/pae_timing_v13/screened`（作业来源）、`OUT = results/pae_screen_batches_v18`；`prepare()/status()/run()` 三段（切批并冻结协议→查态→守护执行）。
**冻结**：sha256 记录于 `results/pae_screen_batches_v18/` 批协议。

### `scripts/run_pae_short_batches_v17.py`（73 行）
**角色**：v17 短批可行性运行：单次守护的 24 作业运行、四个全新 worker（docstring 明示**不作加速声明**——这是纯工程可行性试验）。
**关键**：常量 `OLD = results/pae_guarded_abba_v16`（对照基线）、`OUT = results/pae_short_batches_v17`；`prepare()/status()/run()` 三段结构与 v18 同型。
**冻结**：sha256 记录于 `results/pae_short_batches_v17/` 批协议。

### `scripts/run_successor_v3_future_snapshot.py`（158 行）
**角色**：以守护方式运行仅元数据的 successor-v3 未来快照工作流（前瞻队列的周期性采集，供未来确认性评估）。
**关键**：常量 `DEFAULT_POLICY = configs/benchmarks/successor_v3_future_snapshot_policy.yml`（政策文件——采集节奏与门槛的权威来源）；`cadence_days()`（采集间隔）、`build_plan()/execute_plan()`（计划→执行两段，全程元数据、无结构下载）。
**冻结**：冻结哈希记录于 results/ 证据链（successor_v3 未来快照协议）。

### `scripts/score_aayl_original_retrospective.py`（100 行）
**角色**：AAYL 线起点：对经源核验的 AlphaSeq H3 变体做描述性 ECLS 评估（回溯打分，产出后续 learnability/low_order 实验共用的 scores.json）。
**关键**：常量 `SOURCE = data/aayl_original_retrospective_v1`、`OUT = results/aayl_original_retrospective_v1`；`digest()`（收据）、`correlation()`。
**冻结**：sha256 记录于 `results/aayl_original_retrospective_v1/` 协议。

### `scripts/score_ecls_hard_controls.py`（180 行）
**角色**：ECLS 硬对照打分：显式种子下的对照 ECLS 分数与开发数据上的配对不变性检查（数值/刚性变换不变性的正向证据）。
**关键**：`rigid_transform()`（刚性变换施加）、`ecls_scores()`（ECLS 打分本体）、`make_batch()`；`OUT = results/ecls_hard_controls_scored_v1`。
**冻结**：sha256 记录于 `results/ecls_hard_controls_scored_v1/` 协议。

### `scripts/smoke_h3_t2.1.py`（82 行）
**角色**：T2.1 管线的工程冒烟测试：在单一案例上验证端到端可跑通（非科学结论，仅管线健康检查）。
**关键**：`smoke_test()`；82 行薄脚本。
**冻结**：哈希被 `results/publication/` 发布包清单记录。

### `scripts/strengthen_pae_v12.py`（156 行）
**角色**：v12 事后固定基线比较（docstring 明示：**不在 v11 标签上做任何选择或拟合**）：对既有筛选特征做固定口径的召回比较，探讨教师标签利用的强化空间。
**关键**：`recall()`（固定 top-20% 目标、可变筛选预算，截断并列整合）、`feature_matrix()`；`SOURCE = results/pae_public_pilot_v11`、`OUT = results/pae_strengthening_v12`。
**冻结**：sha256 记录于 `results/pae_timing_v12/`、`results/pae_timing_v13/` 协议（输出目录为 `results/pae_strengthening_v12/`）。

### `scripts/time_pae_screening_v12.py`（128 行）
**角色**：v12 计时比较起点：同设备全新墙钟测量——从候选池到 AF2 输出的端到端耗时（为 v13 重测与 v14 探针提供基准）。
**关键**：常量 `SOURCE = results/pae_public_pilot_v11`、`WEIGHTS = results/pae_screening_v7/ridge_fixed10.npz`、`OUT = results/pae_timing_v12`；`prepare()/run()`（计时作业的准备与守护执行）、`status()`。
**冻结**：sha256 记录于 `results/pae_timing_v12/` 协议。

### `scripts/validate_release_lineage.py`（225 行）
**角色**：发布谱系校验器：递归校验冻结的 DisorderFlow 发布谱系文档与本地工件——逐一核对谱系文档引用的代码/数据/检查点哈希与交叉契约，是"哈希冻结"机制的**主动验证端**。
**关键**：常量 `DEFAULT_DOCUMENTS`（`publication/ECLS_SCOPE_FREEZE.yml`、`publication/successor_v3_contact_v2_registry.json`、`configs/benchmarks/successor_v3_contact_v2_frozen_code_rescan.json` 等）、`SOURCE_ONLY_DOCUMENTS`；`validate_document()/validate_checkpoint()/validate_cross_contracts()`、`collect_rescan_maps()`。配套测试 `tests/test_release_lineage.py`。
**冻结**：冻结哈希记录于 results/ 证据链（发布谱系与 frozen_code_rescan 链）。

### `scripts/verify_ecls_saved_results.py`（225 行）
**角色**：ECLS 已保存结果核验器（docstring 明示：只从已保存分数核验、**从不调用模型**）：重算排名与图中三个合并区间，验证图表数据与统计陈述一致——发表图的可复现凭证。
**关键**：`verify_ranking()`（重算排名与三个作图用 pooled 区间）、`verify_figure_data()`、`verify_manifest()`、`bootstrap()`、`native_rank()`。
**冻结**：冻结哈希记录于 results/ 证据链（ECLS 发表核验链）。

---

*（本文件至此覆盖 `frozen_list.txt` 中 `scripts/` 下除 `FROZEN_ANNOTATIONS_scripts_a.md` 已注解的 12 个 v23–v30 门控链/运行时脚本外的全部 114 个冻结脚本；`disorderflow/`、`modules/`、`tests/` 冻结文件另行安排。）*

