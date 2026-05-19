# BFN 置信度头微调 — 软件与算法改良进程报告

**生成时间:** 2026-05-16
**项目:** AntibodyDesignBFN — Confidence Head Fine-Tuning
**目标:** 使 BFN 对通用蛋白质输出有意义的 pLDDT/ipTM/PAE 置信度预测

---

## 1. 问题背景

BFN (Bayesian Flow Network) 原始模型仅在抗体 CDR 数据上以极低权重 (loss_weight=0.01) 训练了置信度头 (pLDDT/ipTM/PAE)。对通用蛋白质，三个置信度预测均接近 0，导致下游级联过滤器和置信度回收模块完全失效。

**核心目标:** 用 AlphaFold2 作为教师模型，微调 BFN 置信度头，使其对通用蛋白质也能输出有意义的置信度预测。

---

## 2. 架构缺陷修复 (Bug Fixes)

### 2.1 PAE 输出尺度不匹配 — 90,000 倍初始损失改善

**发现:** receiver.py 中 PAE 输出使用 `F.softplus(head_pae) * 10.0`，范围为 [0, ~25]，但训练目标 `af2_pae / 31.0` 范围为 [0, 1]。存在约 10× 尺度差距。

**修复:** `receiver.py:148,165` — 将 `softplus * 10` 改为 `sigmoid`，输出约束到 [0, 1]

**效果:**
| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 初始 PAE loss | 74.30 | 0.118 |
| 初始总 loss | 22.59 | 0.491 |

### 2.2 `seq_only` 模式阻止置信度计算

**发现:** `core.py:30` 判断 `seq_only` 时只检查结构相关 loss (dist/fape/pos/rot)，不检查置信度 loss。当仅训练置信度时 → `seq_only=True` → `receiver.py:139-141` 将 pLDDT/ipTM/PAE 硬编码为零。

**修复:** `core.py:33` — 扩展 `seq_only` 判断逻辑，加入置信度 loss key 检查。`receiver.py` seq_only 分支保留置信度预测。

### 2.3 `evaluate_confidence.py` — 零相关被误判为 N/A

**发现:** 当 Pearson r 恰好为 0.0 时，`bool(0.0) = False`，显示 "PAE r=N/A" 而非正确值。

**修复:** 将 `if value` 改为 `if value is not None`

### 2.4 `train.py` — 常规 checkpoint 保存过期的 `min_val_loss`

**发现:** 验证后先构造 ckpt_state (含旧 min_val_loss) 并保存 `it.pt`，再更新 min_val_loss 保存 `best.pt`。若从 `it.pt` 恢复训练，min_val_loss 可能是 `inf`。

**修复:** 先更新 min_val_loss，再统一构造 ckpt_state 并保存

### 2.5 `confidence_dataset.py` — 未定义变量 `db_path`

**发现:** pickle 目录回退分支中引用了未定义的 `db_path`（应为 `self.db_path`）。

**修复:** 两处 `db_path` → `self.db_path`

### 2.6 `prepare_confidence_dataset.py` — 过早应用 PaddingCollate

**发现:** 数据集构建阶段调用了 `PaddingCollate`，给张量添加了 batch 维度。DataLoader 加载时会再次应用，导致形状错误 `(N, 1, L)` 而非 `(N, L)`。

**修复:** 移除构建阶段的 PaddingCollate，由 DataLoader 统一处理

---

## 3. 架构升级

### 3.1 pLDDT 头加深 (v1 → v2)

**问题:** pLDDT 头原为单一的 `Linear(256, 1)` — 257 个参数，无非线性激活。而 ipTM 头 (`Linear→ReLU→Linear`) 有 66,049 参数，PAE 头 (`Linear→ReLU→Linear`) 有 33,025 参数。架构不对称导致 pLDDT 性能瓶颈。

**升级:**

| 组件 | 旧架构 | 新架构 | 参数量变化 |
|------|--------|--------|-----------|
| head_plddt | `Linear(256,1)` + sigmoid | `Linear(256→128)→ReLU→Linear(128→32)→ReLU→Linear(32→1)` + sigmoid | 257 → ~33K |

**理由:** ipTM 使用相同的冻结特征（全局池化）能达到 r=0.78，说明特征质量足够。pLDDT 需要更强的逐残基解释能力，单层线性显然不足。

### 3.2 冻结骨干微调机制

**实现:** `train.py:131-151` — `freeze_backbone` 配置项，通过参数名后缀匹配 (`head_plddt`, `head_iptm`, `head_pae`, `conf_embed`, `iptm_embed`, `pae_embed`, `pair_proj`) 选择性解冻。

- 可训参数: 133,635 / 10,200,662 (1.3%) → 升级后 ~166,000 (1.6%)
- 训练速度: ~30ms/iter (batch_size=1, CUDA)
- 显存占用: ~2GB (远低于 RTX 5060 8GB 限制)

---

## 4. 数据管线建设

### 4.1 数据来源

| 来源 | 训练 | 验证 | 说明 |
|------|------|------|------|
| EBI AFDB v6 | 160 | 40 | UniProt Swiss-Prot 蛋白，长度 50-250 |
| ColabFold 本地 | 16 | 4 | 本地 AF2 预测 |
| PDB 实验结构 | 16 | 4 | PDB 结构 + 本地 AF2 |
| **v1 合计** | **187** | **47** | **234 蛋白质** |

### 4.2 数据集 v2 — 扩充至 353 蛋白质

**新增:** 从 EBI AFDB v6 下载 350 个新的 UniProt 蛋白 (长度 30-250)，349 个成功 → 223 train / 56 val。

**合并:** 与原有 ColabFold + PDB 条目去重合并。
- **v2 合计: 281 train / 72 val (353 蛋白质, +51%)**

**长度覆盖改善:**
| 类别 | 旧 | 新 |
|------|-----|-----|
| 短蛋白 (<80aa) | ~6 | 35 |
| 中蛋白 (80-200aa) | ~200 | 222 |
| 长蛋白 (>200aa) | ~28 | 96 |

### 4.3 疾病蛋白质数据集 (构建中)

覆盖 10 个器官/疾病类别:
- 肝病 (肝炎、肝硬化、肝癌)
- 肾病 (肾炎、多囊肾、肾衰)
- 心脏病 (心肌病、心律失常、长QT)
- 肺病 (囊性纤维化、肺纤维化、COPD)
- 神经退行性疾病 (阿尔茨海默、帕金森、ALS、亨廷顿)
- 癌症 (癌基因、肿瘤抑制因子)
- 代谢疾病 (糖尿病、肥胖、痛风)
- 血液/免疫疾病 (血友病、地中海贫血、SCID)
- 肌肉骨骼疾病 (肌营养不良、成骨不全)
- 眼科疾病 (视网膜色素变性、黄斑变性)

预计新增 ~200 个人类疾病相关蛋白质。

### 4.4 关键基础设施

- **LMDB 存储:** pickle 序列化，自动 map_size 估算
- **增量保存:** 每 30 个蛋白自动保存，防止中途失败丢失数据
- **HTTP 重试:** `resilient_get()` 带指数退避重试 (最多 3 次)
- **断点续传:** `--resume` 标志，自动跳过已处理蛋白
- **多源去重:** 基于 UniProt accession 去重合并

---

## 5. 训练系统

### 5.1 训练配置

| 参数 | 值 |
|------|-----|
| 基础检查点 | YueHuLab/AntibodyDesignBFN best.pt (113MB) |
| 批次大小 | 1 |
| 学习率 | 1e-4 |
| 调度器 | Plateau (factor=0.5, patience=30) |
| 预热步数 | 200 |
| 最大梯度范数 | 1.0 |
| 损失权重 | pLDDT=1.0, ipTM=1.0, PAE=0.3 |
| AMP | 禁用 (float16 不兼容) |

### 5.2 训练日志自动生成

`generate_training_log.py` — 解析训练输出，自动生成 8 节 Markdown 报告:
1. 训练配置
2. 关键修复
3. 训练进度 (里程碑表)
4. 验证损失曲线
5. 最终指标
6. 评估结果
7. 与上次训练对比
8. 已知问题

日志自动保存到 `C:\biological\AntibodyDesignBFN-main\日志\`

---

## 6. 评估框架

`evaluate_confidence.py` — 完整评估管线:
- **pLDDT:** 逐残基 Pearson r, Spearman ρ, MAE
- **ipTM:** 逐蛋白质 Pearson r, Spearman ρ, MAE
- **PAE:** 逐残基对 Pearson r, Spearman ρ, MAE
- 输出: 全局汇总 + 逐蛋白质详细 JSON

---

## 7. 训练结果演进

### 7.1 第一轮训练 (v1 架构 + 234 蛋白质)

| 指标 | 初始 (iter 100) | 最终 (iter 20000) |
|------|-----------------|-------------------|
| Val loss | 0.3759 | 0.0529 |
| pLDDT val | 0.2848 | 0.0219 |
| ipTM val | 0.0555 | 0.0146 |
| PAE val | 0.1186 | 0.0546 |

**评估:**
| 指标 | pLDDT | ipTM |
|------|-------|------|
| Pearson r | 0.483 | 0.777 |
| Spearman ρ | 0.351 | 0.780 |
| MAE | 0.121 | 0.114 |

### 7.2 第二轮训练 (v2 架构 + 353 蛋白质) — 进行中

当前进度: iter ~11000/30000, best val loss 0.0606

---

## 8. 已知局限

1. **IDP 盲区:** 训练数据全部为折叠良好的蛋白质，对 Tau 等固有无序蛋白 (IDP) 预测偏高 (pLDDT r≈0.21)
2. **长度泛化:** 训练数据 L=21-250，超长蛋白 (>250aa) 未经测试
3. **编码器冻结:** 抗体训练的编码器特征可能不适配通用蛋白质
4. **反馈嵌入未训练:** `conf_embed`/`iptm_embed`/`pae_embed` 仅在推理时使用 (recycle>0)，训练期间无梯度信号

---

## 9. 审计发现的次要问题 (已全部修复 ✅ — 2026-05-16 第二轮)

| 问题 | 位置 | 修复 | 状态 |
|------|------|------|------|
| MPS + float16 不兼容 | train.py:182-185 | autocast 在 MPS 上禁用，仅 CUDA 使用 float16 | ✅ |
| Warmup 覆盖 scheduler LR | train.py:234-241 | 预热期间暂停 scheduler.step()，避免内部 LR 衰减 | ✅ |
| receiver 膨胀内存 | receiver.py:151-152 | 添加 O(L²·D) 内存警示注释，batch=1 时安全 | ✅ |
| 侧链角 log_weights 不更新 | core.py:313-316 | theta_ang[0] 每步累加 delta_alpha，与其他 theta 保持一致 | ✅ |
| Pearson r NaN 未防护 | evaluate_confidence.py:88 | 添加 std > 1e-12 防护，避免常量输入产生 NaN | ✅ |
| 固定种子分割 | build_af2_dataset.py | 添加 --seed CLI 参数 (默认 2026)，允许不同分割 | ✅ |

---

## 10. 文件变更汇总

### 新建文件
| 文件 | 用途 |
|------|------|
| `evaluate_confidence.py` | 置信度评估管线 |
| `generate_training_log.py` | 训练日志自动生成 |
| `generate_worklog_docx.py` | Word 工作日志生成 |
| `build_af2_dataset.py` | EBI AFDB 数据集构建 |
| `build_disease_dataset.py` | 疾病蛋白质数据集构建 |
| `prepare_confidence_dataset.py` | PDB 结构数据集构建 |
| `test_tau_protein.py` | Tau IDP 端到端测试 |
| `antibodydesignbfn/datasets/confidence_dataset.py` | 置信度回归数据集类 |
| `configs/train/bfn_confidence_combined.yml` | 训练配置 |

### 修改文件
| 文件 | 变更 |
|------|------|
| `receiver.py` | PAE sigmoid 修复; pLDDT 头加深; seq_only 置信度保留 |
| `core.py` | seq_only 判断扩展; 置信度 loss 计算 |
| `train.py` | freeze_backbone 支持; checkpoint 修复; pLDDT 迁移处理; MPS float16 修复; warmup scheduler 修复 |
| `evaluate_confidence.py` | PAE 注释更新; 零值检查修复; Pearson r NaN 防护 |
| `confidence_dataset.py` | db_path 变量修复 |
| `prepare_confidence_dataset.py` | PaddingCollate 修复; --seed 参数 |
| `core.py` | seq_only 判断扩展; 置信度 loss 计算; 侧链角 log_weights 修复 |
| `receiver.py` | PAE sigmoid 修复; pLDDT 头加深; seq_only 置信度保留; O(L²·D) 内存注释 |
| `build_af2_dataset.py` | --seed 参数 |
| `app_config.yaml` | BFN checkpoint 更新为 v3 best.pt |

---

## 11. 下一步

1. 完成疾病蛋白质数据集构建并合并
2. 完成 v2 训练 (30000 iter)
3. 评估 pLDDT 是否达到 r>0.6 目标
4. 若未达标: 考虑解冻部分编码器层、继续扩充数据、或引入序列对比特征

---
*此报告覆盖 2026-05-15 至 2026-05-16 的所有关键改良*
