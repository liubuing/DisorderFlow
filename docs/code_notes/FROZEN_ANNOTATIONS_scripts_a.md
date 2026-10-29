# 冻结脚本配套注解（一）：PAE 类似物迁移证据链 v23–v30 及运行时基础设施

> **为什么本文件存在**：下列脚本的 sha256 被冻结协议 JSON 记录（`results/pae_analog_*/protocol.json`、`results/pae_analog_preregistration_v27/preregistration.json` 等），修改文件本身会使已发布证据链的哈希校验失效。因此这些脚本**不可就地加注释**，本文件是只读配套注解。清单来源：`C:\Users\34342\AppData\Local\Temp\comment_batches\frozen_list.txt`（自动哈希比对生成）。
>
> 本文件覆盖注解作者第一手参与的脚本；其余冻结脚本由后续自动化批次续写。

---

## 一、v23–v30 类似物迁移门控链（按执行顺序）

### `scripts/audit_pae_analog_inputs_v23.py`（201 行）
**角色**：门控链第 2 环。对 v22 收割元数据（`data/pae_analog_scope_v22/`，TCR 272 条 + VHH 52 条）做离线完整性/化学审计：收据哈希校验、5–50 aa 规范序列长度、长度一致性、标准字母表、核酸/受体描述词排除、实体回溯匹配。
**关键约定**：`freeze()` 拒绝重写已存在文件（write-once 语义）；输出 `eligible_for_model_run` 恒为 0——在链级身份、重叠、溯源、拆分门控全部通过前禁止跑模型（fail-closed）。结果：TCR 266 / VHH 38 条通过（`results/pae_analog_audit_v23/`）。

### `scripts/number_pae_analog_chains_v24.py`（217 行）
**角色**：门控链第 3 环，链级身份核验。用 ANARCII 2.0.8（CPU、accuracy 模式）对通过 v23 的条目做受体链编号：TCR 池用 TCR 模型（IMGT，CDR3-β = 105–117 位置），VHH 池用抗体模型（Chothia），抗体模型编号失败的链再用 shark 模型复验（区分 VNAR 与普通错误）。
**关键约定**：编号批文件 write-once、可断点续跑；标签字典以**序列原文**为键（历史上曾因哈希键/序列键混用导致全部 miss，修复后经 8DNT 冒烟验证）。结果：TCR 265 / VHH 24 条通过（VHH 中 13 条因存在轻链被排除——那是常规 Fab 而非单域抗体）。

### `scripts/audit_pae_analog_overlap_v25.py`（226 行）
**角色**：门控链第 4 环，回顾性暴露审查。对 v3 参考联合清单（`data/successor_v3_confirmatory/reference_union_manifest_v3.json`，4,749 条记录 + 576 个精确暴露 PDB）做三轴比对：精确 PDB ID、肽段精确/包含匹配、4-mer 种子引导的 80% 双向覆盖 best_overlap（抗原轴 identity ≥0.3、受体轴 ≥0.9，v9 规则）。
**关键约定**：`SeedIndex` 以共享 4-mer 为合格比对的必要条件（不会漏掉真命中），按共享种子数排序 + 扫描上限（受体 500 / 肽段 5000）控制耗时；灵敏度边界（种子+封顶+无 MMseqs）如实写入协议。结果：TCR 265 条全部暴露干净；VHH 仅 3/24 干净——VHH 被项目既往工作穷尽，这本身成为论文的可行性结论。

### `scripts/materialize_pae_analog_structures_v26.py`（468 行）
**角色**：门控链第 5 环，坐标门控与组件物化。下载全部通过 v24 条目的 mmCIF（带 User-Agent 收据），按序列比对解析受体/伙伴/肽段链（identity ≥0.9、受体解析覆盖 ≥0.7），坐标链重新 ANARCII 编号，提取候选区（TCR：CDR3-β IMGT 105–117；VHH：Chothia 93–102，镜像项目 H3 定义），计算候选区-肽段重原子接触（截断 4.5 Å，`materialize_successor_v3_confirmatory.CONTACT_CUTOFF`），写出规范组件 PDB。
**关键约定**：化学约定比抗体管线更严——受体 MSE→M 记录、肽段仅严格标准残基；组件 PDB 由 `write_component_pdb()` 手写 ATOM 行生成（连续重编号 1..N），避免 Bio.PDB deepcopy 沿父图递归复制（曾致 OOM）与多字符链 ID 超 PDB 格式限制两个病理。结果：856/4,567 组合通过（TCR 849 + VHH 7）；主淘汰为接触门（3,710——pMHC 中不少 CDR3 主要接触 MHC 而非肽段，符合预期）。

### `scripts/split_pae_analog_families_v27.py`（199 行）
**角色**：门控链第 6 环，家族不相交拆分。对 v26 通过组件去重（(pdb, 受体实体, 肽段序列) 三元组）、剔除 v25 暴露命中条目、仅保留 TCR（VHH 唯一干净组件降为描述性试点），按 v25 同款轴对肽段聚类（连通分量 = 家族），确定性规则（家族按 (规模升序, 键) 选取）分出评估侧与训练侧。
**关键约定**：家族零重叠是确认性的核心——评估家族不得出现在训练侧；最小规模（评估 ≥6 组件 ≥3 家族）不满足时记录可行性失败而不放宽阈值。结果：确认性 TCR 170 组件；评估 6 组件/6 家族，训练 12 组件/10 家族。

### `scripts/freeze_pae_analog_preregistration_v27.py`（68 行）
**角色**：把预注册文档（`docs/PAE_ANALOG_PREREGISTRATION_V27.md`）及其引用的全部上游门控产物哈希固化为机器可读 `results/pae_analog_preregistration_v27/preregistration.json`。必须在任何候选生成/AF2 标签之前运行；文档 sha256（65831f97…）是"先冻结后执行"承诺的机器凭证。

### `scripts/generate_pae_analog_candidates_v28.py`（249 行）
**角色**：候选生成与教师作业准备。对 v27 每个组件：把 v26 冻结 PDB 连续重编号为设计 PDB（`renumber_pdb`），ProteinMPNN（v_48_020 可溶权重、温度 0.3、种子 20260919）在受体链上生成候选（评估组件 14 个/训练组件 7 个，仅候选区可变），构建 native 与组成打乱对照（打乱种子 7307，要求非天然），并产出六重复 AF2 作业清单（`:` 分隔链序遵循抗体约定：候选链在前）。
**关键约定**：`inject()` 按位置逐个替换（候选区索引可能因插入码不连续）；本脚本不跑任何 AF2 预测——作业只写出、不执行。

### `scripts/run_pae_analog_teacher_v29.py`（218 行）
**角色**：守护式教师标签运行器（Windows `prepare` + WSL `run` 两阶段）。prepare 把 v28 作业切成 12 槽/批并冻结批协议；run 对每批先做 GPU 准入（连续 3 个 10 秒采样：无计算进程、显存 <3000 MiB，DEV-001 后利用率阈值 20%→60%），再调用未修改的 `guard_pae_batch_v19.run()`，验证通过才计入完成。
**关键约定**：断点续跑跳过已验证批次；任何非 verified 终态的 attempt 目录视为已消耗（守护器禁止静默续跑）；DEV-002 后显存守护线 18.5→19.5 GB、耗尽批次记录于 `exhausted.json` 并跳过而非中止整个运行；status/attempts/result 均为活遥测、直写覆写。结果：1,080/1,080 槽验证通过、零失败批次。调度差异 DEV-001/002/003 记录于 `results/pae_analog_deviation_log/`，须在论文方法学调度附注中披露。

### `scripts/evaluate_pae_analog_v30.py`（262 行）
**角色**：终点评估。从**磁盘上已验证**的 attempt 目录（非 result.json 的 attempts 列表——续跑会跳过早期批次不写该表）索引全部 PAE；教师标签按 v4 冻结公式：候选区起始索引取自 v26 冻结 `indices_zero_based`（不做序列搜索，避免歧义），`pae[区行, 肽列].mean()/31`，六重复平均；特征 = 候选区组成 20 + 肽段组成 20 + 长度 + 6 个骨架几何标量（CB 距离，GLY 回退 CA）；StandardScaler+Ridge(alpha=10) 仅在训练组件拟合；主终点 = 20% 预算（3/14）下教师最优 20% 召回（均匀随机平局期望），组件等权；配对组件 bootstrap 10,000（种子 20260918）；MPNN NLL 符号只在训练侧确定。
**结果**：Ridge 50.0% vs 随机 21.4%，差 +28.6 pp，95% CI [17.5, 39.7]，6/6 家族为正；仅组成消融与全特征一致；native 与组成打乱对照在教师层不可区分（详见 `docs/PAE_ANALOG_TRANSFER_V30.md`）。

## 二、运行时基础设施（从抗体管线原样复用）

### `scripts/guard_pae_batch_v19.py`（88 行）
**角色**：单批守护工作器（须在 WSL 内运行）。启动 `scripts/utils/af2_wsl_batch.py` 子进程（start_new_session 独立进程组），周期采样 GPU 总显存/可用主机内存/计算进程属主；总显存连续 2 次达守护线、主机内存不足 2 GiB 或超时则 SIGTERM→SIGKILL 杀掉**自己拥有的**进程组（绝不碰他人进程）。
**关键约定**：启动时 `assert` 自身/工作器/作业清单三个 sha256 与协议一致——这就是冻结脚本不可就地修改的直接原因；验证 = 无守护原因 + 退出码 0 + 作业 ID 全集合匹配 + 每个 PAE 文件哈希与 worker 回执一致；`status.json` 存在即拒绝重跑（"No silent resume"）。

### `scripts/run_pae_queued_v19.py`（86 行）
**角色**：v19 排队运行器（抗体公共池 144 槽），GPU 准入逻辑的原始出处；v29 的 `admission()` 与重试语义（每批最多 2 次新鲜尝试、仅显存/超时可重试）从它镜像。

### `scripts/utils/af2_wsl_batch.py`（148 行）
**角色**：WSL 侧 AF2 批处理器。stdin 收 JSON 作业流（`seq` 支持 `:` 分隔多链、`epi_seq`、`seed`、`output_pae/pdb`），模型加载一次、JAX 编译一次后逐作业推理；输出 JSONL 含 `success/iptm/ptm/max_pae` 与 `pae_path/pae_sha256`（守护器校验所需）。含 JAX 0.10+ 的 `jnp.clip` 兼容 shim。

## 三、前置管线（教师标签路线上游）

### `scripts/run_pae_public_pilot_v11.py`（327 行）
**角色**：抗体公共池回溯筛选试点（v20 证据的执行体）：选槽 → ProteinMPNN 生成 → 组装三链实体（重+轻+抗原）→ 双模型 × 三种子 WSL 守护运行 → 方向性 PAE 教师标签 → 冻结 Ridge 评估。其 `build_entities`/`windows_to_wsl` 被 v28/v29 复用。

### `scripts/prepare_pae_dense_v6.py`（103 行）
**角色**：v6 密集筛选的数据准备与通用工具出口；本链全部脚本复用其 `read()/freeze()/digest()` 三件套——`freeze` 的 write-once 语义（拒绝重写已存在文件）是整个证据链不可变性的机制基础。
