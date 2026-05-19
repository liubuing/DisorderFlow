# ProteinStudio 完整项目指南

> **面向读者**：大一本科生（具备基础生物学和 Python 知识即可）
> **最后更新**：2026-05-19
> **项目版本**：V3.0（BFN V6 Phase 2）

---

## 目录

1. [项目是什么](#1-项目是什么)
2. [学前知识速成](#2-学前知识速成)
3. [项目整体架构](#3-项目整体架构)
4. [程序清单](#4-程序清单)
5. [核心算法讲解](#5-核心算法讲解)
6. [Web 平台功能](#6-web-平台功能)
7. [训练改进历程](#7-训练改进历程)
8. [使用指南](#8-使用指南)
9. [常见问题](#9-常见问题)
10. [术语表](#10-术语表)

---

## 1. 项目是什么

**ProteinStudio** 是一个蛋白质序列设计平台。简单说：给你一个蛋白质的三维形状（骨架），它能设计出最适合这个形状的氨基酸序列。

### 1.1 它能做什么

| 功能 | 说明 | 类比 |
|------|------|------|
| 序列设计 | 给蛋白质骨架，生成氨基酸序列 | 给你一个人体模型，设计合身的衣服 |
| 置信度评估 | 预测序列/结构的质量好坏 | 给衣服打分（剪裁/面料/版型） |
| 靶点设计 | 针对特定目标蛋白设计抗体 | 给锁配钥匙 |
| 迭代精修 | 反复设计→验证→改进，直到最优 | 试穿→修改→再试穿 |

### 1.2 核心创新

传统方法（如 ProteinMPNN）只能设计序列，但无法告诉你这个序列**有多好**。

BFN 模型**自带评分系统**：
- **pLDDT**：每个氨基酸位置的置信度（0-1，越高越好）
- **ipTM**：整体结构的折叠可信度（0-1，越高越好）
- **PAE**：预测误差估计（越低越好）
- **PPL + Entropy**：序列本身的质量指标

这就像不仅帮你设计衣服，还当场告诉你"这里的缝线可能不牢固"。

---

## 2. 学前知识速成

### 2.1 蛋白质是什么

蛋白质是生命的"分子机器"，由 20 种氨基酸按特定顺序连接而成。氨基酸序列决定了蛋白质如何折叠成三维形状。

```
氨基酸 → 序列 → 折叠 → 三维结构 → 功能
(字母)   (单词)  (折纸)   (形状)    (作用)
```

### 2.2 关键概念

| 概念 | 解释 |
|------|------|
| **氨基酸** | 蛋白质的基本单元，共 20 种，用字母 ACDEFGHIKLMNPQRSTVWY 表示 |
| **PDB 文件** | 记录蛋白质三维坐标的文件格式，包含每个原子的 x, y, z 位置 |
| **抗体** | 免疫系统的"精确制导武器"，能识别并结合特定目标（抗原） |
| **CDR** | 抗体的"弹头"部分，决定抗体能识别什么目标（共 6 个 CDR 环） |
| **骨架 (Backbone)** | 蛋白质的主链原子 N-CA-C-O，形成蛋白质的基本形状 |

### 2.3 什么是神经网络（机器学习速成）

神经网络就像一个"超级函数拟合器"：
- **输入**：蛋白质的骨架坐标
- **输出**：每个位置该放什么氨基酸（以及这个预测的可信度）
- **训练**：给模型看成千上万个已知蛋白质，让它自己学会规律
- **推理**：用训练好的模型对新蛋白质做出预测

---

## 3. 项目整体架构

```
ProteinStudio/
├── 🖥️  Web 界面层
│   ├── app.py                    # Gradio Web 界面（主入口，~2400行）
│   ├── manage.py                 # 服务管理 CLI
│   └── app_config.yaml           # 平台配置
│
├── 🧠 算法核心层
│   └── antibodydesignbfn/        # BFN 核心包
│       ├── models/               # 模型架构定义
│       ├── modules/              # 神经网络组件
│       │   ├── bfn/              # 贝叶斯流网络核心
│       │   ├── encoders/         # 编码器（几何注意力、对相互作用）
│       │   ├── diffusion/        # 扩散模型
│       │   └── common/           # 公共工具（几何、拓扑、结构）
│       ├── datasets/             # 数据加载（抗体、蛋白质、LMDB）
│       ├── tools/                # 工具（对接、评估、弛豫）
│       └── utils/                # 工具函数（训练、推理、变换）
│
├── 🔧 辅助工具层
│   ├── cascade_filter.py         # 级联过滤器（3 级筛选）
│   ├── target_design_helpers.py  # 靶点设计辅助（表位分析、接触分析）
│   ├── af2_validator.py          # AlphaFold2 验证集成
│   ├── iterative_refiner.py      # 迭代精修引擎
│   ├── batch_evaluate_checkpoints.py  # 批量评估检查点
│   └── design_seq.py / design_protein.py  # 命令行设计工具
│
├── 📦 第三方集成
│   ├── ProteinMPNN/              # ProteinMPNN 参考实现
│   └── diffab/                   # DiffAb 参考实现
│
├── 🏋️ 训练系统
│   ├── train.py                  # 训练脚本
│   ├── configs/train/            # 训练配置（V4→V6 各阶段）
│   ├── build_*_dataset.py        # 数据集构建脚本
│   └── merge_*_dataset.py        # 数据集合并脚本
│
└── 📊 评估系统
    ├── evaluate_confidence.py    # 置信度评估
    ├── evaluate_testset.py       # 测试集评估
    └── validate_8proteins*.py    # 8 蛋白验证管线
```

---

## 4. 程序清单

### 4.1 核心程序（按使用频率排序）

#### ⭐ app.py — Web 平台主程序

**作用**：整个平台的 Web 界面，集成了所有功能到一个浏览器窗口中。

**怎么运行**：
```bash
python app.py
# 浏览器打开 http://127.0.0.1:7860
```

**包含的功能模块**：
- 8 个功能标签页（单步设计、批量处理、结构预测、设置、靶点设计、迭代精修、置信度评估、统一工作流）
- 模型加载与管理
- AlphaFold2 集成
- FASTA ↔ PDB 转换

**关键函数**：
| 函数名 | 功能 |
|--------|------|
| `run_bfn_antibody()` | BFN 抗体 CDR 设计 |
| `run_bfn_protein()` | BFN 通用蛋白设计 |
| `run_bfn_confidence_evaluation()` | BFN 置信度独立评估 |
| `run_unified_pipeline()` | 一键式设计→过滤→验证工作流 |
| `run_target_design()` | 靶点导向约束设计 |
| `create_ui()` | 构建整个 Gradio 界面 |

#### ⭐ manage.py — 服务管理工具

**作用**：管理 Web 服务的启动、停止、重启，以及批量处理任务。

```bash
python manage.py start       # 启动
python manage.py stop        # 停止
python manage.py restart     # 重启
python manage.py status      # 状态
python manage.py batch <dir> # 批量处理
python manage.py config      # 查看配置
python manage.py test        # 自检
```

---

### 4.2 算法程序

#### train.py — 模型训练

**作用**：从头训练或微调 BFN 模型。

**用法**：
```bash
# 从头训练
python train.py configs/train/bfn_confidence_combined_v6_phase2.yml

# 从检查点恢复
python train.py config.yml --resume checkpoint.pt

# 微调
python train.py config.yml --finetune checkpoint.pt
```

**训练过程**：
1. 加载训练和验证数据集
2. 每 N 步评估验证集损失
3. 自动保存最佳模型检查点
4. 生成 `training_log.md` 记录详细训练历程

#### design_seq.py — 抗体序列设计（命令行版）

**作用**：对给定抗体 PDB 文件进行 CDR 序列设计。

```bash
python design_seq.py input.pdb --heavy H --light L --config configs/demo_design.yml
```

#### design_protein.py — 通用蛋白设计（命令行版）

**作用**：对给定蛋白质 PDB 文件的指定区域进行序列设计。

```bash
python design_protein.py input.pdb --region A:10-50 --config configs/demo_design.yml
```

#### batch_evaluate_checkpoints.py — 批量检查点评估

**作用**：对训练过程中保存的所有检查点进行批量测试，找出最佳模型。

```bash
python batch_evaluate_checkpoints.py \
  --config configs/test/bfn_testset.yml \
  --test_set data/test.csv \
  --ckpt_dir ./logs/.../checkpoints \
  --start_ckpt 100 --end_ckpt 5000 --step 100
```

---

### 4.3 数据和评估程序

| 程序 | 作用 |
|------|------|
| `build_af2_dataset.py` | 从 AlphaFold2 预测结果构建训练数据集 |
| `build_disease_dataset.py` | 从疾病相关蛋白构建数据集 |
| `build_brain_disease_dataset.py` | 从脑部疾病蛋白构建数据集（V6 新增） |
| `build_idp_dataset.py` | 从 DisProt 构建无序蛋白数据集 |
| `build_lmdb_from_af2.py` | 将 AF2 结果转换为 LMDB 格式 |
| `merge_confidence_datasets.py` | 合并多个置信度数据集 |
| `evaluate_confidence.py` | 独立评估模型置信度预测准确性 |
| `evaluate_testset.py` | 在测试集上评估模型性能 |
| `validate_8proteins_pipeline.py` | 8 蛋白完整管线验证 |

---

### 4.4 辅助模块

#### cascade_filter.py — 级联过滤器

**作用**：对设计结果进行三级筛选和排名。

**三级过滤流程**：
```
输入 N 条序列
    ↓
第 1 级：硬阈值过滤
  - pLDDT < 0.6 → 拒绝
  - ipTM < 0.4 → 拒绝
  - PPL > 100 → 拒绝
  - entropy > 2.5 → 拒绝
    ↓
第 2 级：去重
  - 相同序列只保留 PPL 最好的
    ↓
第 3 级：综合评分排名
  分数 = 0.35×ipTM + 0.25×pLDDT + 0.15×(1/PPL) + 0.10×(1-entropy) + 0.15×recovery
    ↓
输出排名结果
```

**核心函数**：
| 函数 | 说明 |
|------|------|
| `apply_cascade()` | BFN 置信度级联过滤 |
| `apply_cascade_af2()` | AF2 验证后的级联过滤 |
| `format_filtered_fasta()` | 格式化输出 FASTA |

#### target_design_helpers.py — 靶点设计辅助

**作用**：提供表位预测、接触分析、界面评分等功能。

**核心功能**：
| 函数 | 说明 |
|------|------|
| `score_epitope_residues()` | 多维度评分表位残基（SASA + 亲水性 + 凸出指数） |
| `find_residues_facing_region()` | 找到抗体中面向表位的残基 |
| `analyze_contacts()` | 分析抗体-抗原接触面 |
| `score_interface_properties()` | 评估界面特性（氢键、盐桥、疏水性） |

#### iterative_refiner.py — 迭代精修引擎

**作用**：通过"设计→验证→筛选→再设计"的循环不断优化序列。

**工作流程**：
```
第 1 轮：设计 N 条候选序列
           ↓ AF2 预测结构
           ↓ 级联筛选 Top-K
第 2 轮：用 Top-K 的 AF2 结构作为新模板再设计
           ↓ ...
           ↓ 直到收敛或达到最大轮次
输出：最优序列 + 完整精修报告
```

---

## 5. 核心算法讲解

### 5.1 贝叶斯流网络 (Bayesian Flow Networks, BFN)

#### 什么是 BFN

BFN 是一种新型生成模型。和传统的扩散模型不同，BFN 直接在**参数空间**中操作，而不是在数据空间。

**类比理解**：
- 扩散模型（如 DALL-E）像"绘画"：从噪声逐步生成图像
- BFN 像"投票"：每个位置对 20 种氨基酸有不同"偏好"，随着时间推移，偏好越来越明确

#### BFN 的工作原理

```
阶段 1：发送者 (Sender Process)
  已知的氨基酸序列 → 逐步加入噪声 → 完全随机

阶段 2：接收者 (Receiver Process)  
  神经网络观察当前状态，预测"去噪"方向
  → 输出每个位置 20 种氨基酸的概率分布
  → 同时输出 pLDDT/ipTM/PAE 置信度

阶段 3：采样
  从完全随机开始，逐步"去噪"，最终得到氨基酸序列
```

#### 为什么 BFN 适合蛋白质设计

1. **离散数据天然适配**：氨基酸是离散的（20 选 1），BFN 在概率空间中操作，天然处理离散分布
2. **自带置信度**：接收者网络输出分布的不确定性直接对应 pLDDT
3. **连续时间**：可以灵活控制采样步数，权衡速度和质量

---

### 5.2 几何 Transformer (Geometric Transformer)

#### 什么是 Transformer

Transformer 是一种基于"注意力机制"的神经网络架构。Attention 的核心思想是：让序列中的每个位置都能"看到"其他所有位置，按相关性加权聚合信息。

**类比**：班级里每个同学（氨基酸）要了解全班（整条蛋白链），但更关注和自己关系近的同学（空间近邻）。

#### IPA (Invariant Point Attention) — 等变注意力

普通 Transformer 不关心三维空间，但蛋白质设计必须考虑空间位置。IPA 解决了这个问题：

1. 每个残基有一个局部坐标系（基于 N-CA-C 骨架原子）
2. 注意力权重同时考虑序列特征和空间距离
3. 输出**等变**于旋转和平移（不管蛋白怎么转，预测结果一样）

**通俗理解**：IPA 像是给每个氨基酸装上了"GPS + 指南针"，它知道自己在空间中的位置和朝向，并能据此和其他氨基酸"对话"。

---

### 5.3 置信度头 (Confidence Heads) — 模型的"自评系统"

这是 V4-V6 训练的核心创新。模型中增加了三个专门的输出头：

#### pLDDT 头 (per-residue Local Distance Difference Test)

**预测什么**：每个氨基酸位置的结构置信度（0-1）

**怎么训练**：用 AlphaFold2 的 pLDDT 分数作为"标准答案"，让 BFN 学会预测 AF2 会给每个位置打多少分。

**结构**：V6 使用 3 层 MLP（V5 只用 2 层），输入编码器特征，输出 0-1 之间的分数。

#### ipTM 头 (interface predicted TM-score)

**预测什么**：全局结构折叠质量（0-1）

**怎么训练**：用 AF2 的 ipTM 作为标准答案。

**V6 架构（context-only pooling）**：
- 对所有残基对做平均池化
- 因为只看"上下文"（蛋白骨架），不看具体序列
- 一个蛋白一个 ipTM 值（不随序列变化）

#### PAE 头 (Predicted Aligned Error)

**预测什么**：每对残基之间的预测误差（Å）

**怎么训练**：用 AF2 的 PAE 矩阵作为标准答案。

---

### 5.4 为什么 ipTM 不随序列变化

这是 V5→V6 的关键设计决定。ipTM 用 **context-only pooling**：

```
传统方法：pair特征 → 考虑序列 → ipTM（序列改变则 ipTM 改变）
V6 方法：pair特征 → 全局平均 → ipTM（只依赖骨架，不依赖序列）
```

**好处**：ipTM 更稳定，不会被序列微小变化误导
**代价**：同一蛋白的不同设计，ipTM 完全相同

**解决方案**：用 **PPL + Entropy** 作为序列质量代理指标

---

### 5.5 PPL 和 Entropy — 序列质量"双子星"

由于 ipTM 不随序列变化，我们引入了两个序列相关指标：

#### PPL (Perplexity，困惑度)

**定义**：模型对"正确答案"的确定程度
- PPL 越低 → 模型越确信预测 → 序列越好
- PPL = exp(负对数似然)

**计算**：
```python
# 模型输出 logits（20 种氨基酸的原始分数）
logits = model_output  # shape: [n_residues, 20]
# 取 softmax 对数
log_prob = log_softmax(logits)
# 选取实际预测氨基酸的概率
nll = -log_prob[预测位置的氨基酸索引]  # 负对数似然
ppl = exp(nll.mean())  # 困惑度
```

#### Entropy (熵)

**定义**：预测分布的"尖锐程度"
- Entropy 越低 → 预测越集中（模型越确定）→ 质量越高
- Entropy 范围：[0, log(20)] ≈ [0, 2.996]

**计算**：
```python
# p = softmax(logits)
p = exp(log_prob)
# H = -Σ p·log(p)
entropy = -(p * log_prob).sum(dim=-1).mean()
```

**两者结合**：级联过滤器将 PPL 和 Entropy 组合为"质量分数"：
```
quality_score = 0.15 × PPL⁻¹（归一化）+ 0.10 × entropy⁻¹（归一化）
```

---

## 6. Web 平台功能

### 6.1 八个标签页详解

#### Tab 1：🎯 单步设计

**什么时候用**：对单个 PDB 文件进行序列设计。

**操作步骤**：
1. 上传 PDB 文件（或输入 FASTA 序列自动生成模板 PDB）
2. 选择设计工具：BFN 抗体 / BFN 通用 / ProteinMPNN / ESM-IF
3. 设置参数（设计区域、样本数等）
4. 点击"开始设计"
5. 查看结果排名、下载 FASTA

**输出**：
- 每条序列的 PPL、pLDDT、ipTM、PAE、Entropy
- 级联过滤后排名
- FASTA 文件下载
- 一键发送到 AlphaFold2 验证

#### Tab 2：📦 批量处理

**什么时候用**：同时对多个 PDB 文件进行设计。

**操作步骤**：
1. 上传多个 PDB 文件
2. 选择工具和参数
3. 点击"开始批量处理"
4. 结果汇总为 JSON + CSV

#### Tab 3：🔮 结构预测

**什么时候用**：只有氨基酸序列，需要预测三维结构。

**操作步骤**：
1. 输入 FASTA 序列
2. 设置 AF2 参数（模型数、回收步数）
3. 点击"开始预测"
4. 预测完成后可一键发送到设计

#### Tab 4：⚙️ 设置

**什么时候用**：修改配置、查看系统状态、预加载模型。

**功能**：
- 编辑 `app_config.yaml`
- 查看各组件状态（模型、工具链）
- 预加载 ESM-IF 或重载 BFN 模型

#### Tab 5：🎯 靶点设计

**什么时候用**：针对特定靶点蛋白设计结合抗体。

**5 步工作流**：
1. 输入靶点结构（上传或 AF2 预测）
2. 表位分析（SASA + 亲水性 + 凸出指数评分）
3. 提供抗体结构（上传或生成模板）
4. 约束设计（只在面向表位的区域设计）
5. 结果验证（接触分析、界面特性评分）

#### Tab 6：🔄 迭代精修

**什么时候用**：需要最优序列，愿意花更多计算时间。

**原理**：
```
模板 PDB → 设计 N 条 → AF2 验证 → 级联筛选 Top-K
                                            ↓
                       用 AF2 结构更新模板 ←┘
                       （重复直到收敛）
```

#### Tab 7：📊 置信度评估

**什么时候用**：不做设计，只想知道蛋白质结构的质量。

**功能**：
- 上传 PDB，指定评估区域
- 输出 pLDDT（每残基）、ipTM（全局）、PAE（残基对误差）
- 逐残基质量表格

#### Tab 8：🚀 统一工作流

**什么时候用**：一键完成完整流程。

**流程**：
1. Stage 0：预设计置信度评估
2. Stage 1：BFN 序列设计（8 条样本）
3. Stage 2：级联过滤（PPL + entropy + pLDDT + ipTM）
4. Stage 3：（可选）AF2 验证 + 重新排名

---

## 7. 训练改进历程

### 7.1 整体路线

```
V4（初始版本）→ V5（架构探索）→ V6（大规模数据 + 最优架构）
```

### 7.2 各版本详情

#### V4 Phase 1（基线版本）

| 项目 | 设置 |
|------|------|
| 数据集 | 629 个 Swiss-Prot 蛋白 (L=50-250) |
| ipTM 架构 | 全局均值池化 |
| 设计 ipTM 范围 | 0.000（恒定—根本不能区分蛋白） |
| ipTM Pearson r | 0.893 |

**问题**：ipTM 头完全无效，对所有蛋白输出相同的值。

#### V5 Phase 1-5（架构探索阶段）

| Phase | ipTM 架构 | 设计 ipTM 范围 | Pearson r | 发现 |
|-------|----------|---------------|-----------|------|
| Phase 1 | 全局 + mask aug | 0.695 恒定 | 0.938 | 有进步但跨蛋白无区分 |
| Phase 2 | 全局 + mask aug | 全蛋白评分 | 0.938 | 增加数据到 1,149 蛋白 |
| Phase 3 | pair-aware | 0.504 恒定 | 0.931 | 退步 |
| Phase 4 | contrastive | 0.521 恒定 | 0.934 | 略好但仍不理想 |
| **Phase 5** | **context-only** | **0.289-0.492** | **0.957** | ✅ **突破** |

**Phase 5 的关键发现**：context-only pooling（全局平均池化）效果最好。ipTM 跨蛋白可变，蛋白内部恒定——这正是我们要的行为。

#### V6 Phase 1-2（数据和模型扩展）

| Phase | 数据集 | ipTM r | pLDDT r | 关键改进 |
|-------|--------|--------|---------|---------|
| Phase 1 | 2,032 蛋白 | 0.950 | 0.817 | 加入脑部疾病蛋白，数据集扩大 77% |
| Phase 2 | 2,032 蛋白 | 0.950 | 0.825 | 3 层 pLDDT 头 + 解冻 encoder |

**V6 Phase 2 最终指标**：
- 最佳 val loss：**0.0494**（Phase 1：0.0510，提升 3.1%）
- pLDDT 头：2 层 → **3 层 MLP**
- ipTM 头：全新 context-only pooling
- 可训练参数：**55.6%**（解冻 encoder 最后 4 层）
- 训练时长：约 11 小时 40 分钟

### 7.3 数据集增长

```
V4: 503 train / 126 val
V5: 919 train / 230 val（+83%）
V6: 1,625 train / 407 val（+77% vs V5，+223% vs V4）
```

### 7.4 架构演变总结

| 阶段 | ipTM 架构 | 为什么 |
|------|----------|--------|
| 1. 全局均值 | 简单粗暴 | 效果差，0.893 r |
| 2. 全局 + mask | 增加训练多样性 | 改善，但仍不能跨蛋白区分 |
| 3. pair-aware | 允许残基对交互 | 反而退步 |
| 4. contrastive | 对比学习 | 略好转 |
| 5. **context-only** | 只看骨架上下文 | ✅ 最优方案 |

---

## 8. 使用指南

### 8.1 环境安装

```bash
# 1. 安装 Python 依赖
cd ProteinStudio
pip install -e .
pip install -r requirements.txt

# 2. 下载模型权重
# 从 HuggingFace 下载：https://huggingface.co/YueHuLab/AntibodyDesignBFN
# 放到 checkpoints/ 目录

# 3. 安装 AlphaFold2 (ColabFold) — 可选
pip install colabfold

# 4. 安装 ProteinMPNN 权重 — 可选
# 从 https://github.com/dauparas/ProteinMPNN 下载权重文件
```

### 8.2 快速上手

**场景 1：我想给一个蛋白质设计序列**

1. 准备 PDB 文件
2. `python app.py` 启动 Web 平台
3. Tab 1 "单步设计" → 上传 PDB → 选择 "BFN (通用蛋白设计)"
4. 设置设计区域（如 "A:10-25"）
5. 点击 "开始设计"

**场景 2：我只有序列，没有结构**

1. Tab 3 "结构预测" → 输入 FASTA 序列
2. 等待 AF2 预测完成（约 5-15 分钟）
3. 点击 "发送到设计"
4. 回到 Tab 1 完成设计

**场景 3：我要给特定靶点设计抗体**

1. Tab 5 "靶点设计"
2. 步骤 1：上传靶点 PDB 或 AF2 预测
3. 步骤 2：分析表位 → 确认表位区域
4. 步骤 3：上传抗体 PDB 或生成模板
5. 步骤 4：开启"距离约束"，设置阈值
6. 步骤 5：开始设计，查看验证报告

**场景 4：我想要最优序列（不赶时间）**

1. Tab 6 "迭代精修"
2. 上传 PDB → 设置区域 → 启动
3. 等待多轮迭代完成
4. 查看精修报告

### 8.3 命令行使用

```bash
# 抗体 CDR 设计
python design_seq.py antibody.pdb \
  --heavy H --light L \
  --config configs/demo_design.yml \
  --num_samples 5

# 通用蛋白设计
python design_protein.py protein.pdb \
  --region A:10-50 \
  --config configs/demo_design.yml

# 批量评估检查点
python batch_evaluate_checkpoints.py \
  --config configs/test/bfn_testset.yml \
  --test_set data/test.csv \
  --ckpt_dir ./logs/.../checkpoints \
  --start_ckpt 100 --end_ckpt 5000
```

### 8.4 配置文件说明

`app_config.yaml` 各字段含义：

```yaml
models:
  bfn:
    checkpoint: path/to/best.pt        # BFN 模型路径（最重要！）

confidence:                            # 置信度阈值
  plddt_high: 0.80                     # pLDDT ≥ 0.8 = HIGH
  plddt_medium: 0.60                   # pLDDT ≥ 0.6 = MEDIUM
  iptm_high: 0.70
  iptm_medium: 0.40
  pae_low: 4.0

workflow:                              # 工作流配置
  cascade:
    thresholds:                        # 级联过滤阈值
      plddt_min: 0.6
      iptm_min: 0.4
      entropy_max: 2.5
    weights:                           # 综合评分权重
      iptm: 0.35
      plddt: 0.25
      ppl_inv: 0.15
      entropy_inv: 0.10
      recovery: 0.15
```

### 8.5 结果解读

设计结果排名表中各列的含义：

| 列名 | 含义 | 好坏判断 |
|------|------|---------|
| **PPL** | 困惑度 | 越低越好（< 30 算好） |
| **Entropy** | 预测熵 | 越低越好（< 2.0 算好） |
| **pLDDT** | 残基置信度 | 越高越好（> 0.7 算好） |
| **ipTM** | 折叠置信度 | 越高越好（> 0.4 算中等） |
| **PAE** | 预测误差 | 越低越好（< 8 算好） |
| **Composite** | 综合评分 | 越高越好 |
| **Quality** | 质量标签 | HIGH > MEDIUM > LOW |

---

## 9. 常见问题

### Q1：为什么我的蛋白 ipTM 很低？

**可能原因**：
- 蛋白长度超过 250（模型训练上限）
- 蛋白有大量无序区域（IDP，模型没见过）
- PDB 结构质量不好

**解决方案**：
- 截取 50-250 残基的核心折叠域
- 使用 Tab 7 "置信度评估"先检查结构质量

### Q2：PPL 很低但 ipTM 也很低，哪个可信？

ipTM 更可信。PPL 只衡量序列质量，ipTM 衡量结构质量。如果 ipTM 很低，说明这个骨架本身可能不适合设计（例如结构不合理）。

### Q3：为什么同一蛋白不同设计的 ipTM 完全一样？

这是 V6 的 **context-only pooling** 设计决策。ipTM 只看蛋白骨架，不看具体序列。用 PPL 和 Entropy 来区分同蛋白不同序列的好坏。

### Q4：如何选择设计区域？

- **抗体 CDR**：自动选择 6 个 CDR 环
- **通用蛋白**：选择表面暴露的 loop 区域（避开疏水核心）
- 用 Tab 7 先做置信度评估，pLDDT 高的区域更适合设计

### Q5：训练需要什么硬件？

- **V6 训练**：NVIDIA GPU，建议 8GB+ 显存
- **推理（设计）**：CPU 即可（使用 `map_location='cpu'`）
- **AlphaFold2**：CPU 可用但慢，GPU 显著加速

### Q6：模型输出 0 是什么情况？

pLDDT 或 ipTM 输出接近 0 可能是：
- 输入结构不符合训练分布（太大/太小/无序）
- 检查点加载不完整（head 权重缺失 → `strict=False` 跳过）
- 建议用 Tab 7 单独评估诊断

---

## 10. 术语表

| 术语 | 英文 | 解释 |
|------|------|------|
| 氨基酸 | Amino Acid | 蛋白质基本单元，共 20 种 |
| 骨架 | Backbone | N-CA-C-O 主链原子 |
| 贝叶斯流网络 | Bayesian Flow Network | 本项目核心算法 |
| CDR | Complementarity Determining Region | 抗体识别区，共 6 个环 |
| 级联过滤 | Cascade Filter | 3 级筛选排名系统 |
| 熵 | Entropy | 预测分布的"尖锐度"，越低越好 |
| FASTA | FASTA | 序列文件格式 |
| 几何注意力 | Geometric Attention | 空间感知的注意力机制 |
| IDP | Intrinsically Disordered Protein | 内禀无序蛋白 |
| IPA | Invariant Point Attention | 等变点注意力 |
| ipTM | interface pTM | 界面折叠置信度 |
| LMDB | Lightning Memory-Mapped Database | 高效数据存储格式 |
| MLP | Multi-Layer Perceptron | 多层感知器（全连接网络） |
| PAE | Predicted Aligned Error | 预测对齐误差 |
| PDB | Protein Data Bank | 蛋白质结构文件格式 |
| pLDDT | predicted LDDT | 预测局部距离差异测试 |
| PPL | Perplexity | 困惑度，越低越好 |
| pTM | predicted TM-score | 预测折叠质量 |
| SASA | Solvent Accessible Surface Area | 溶剂可及表面积 |

---

> **文档版本**：v1.0
> **作者**：liubuing
> **项目地址**：[github.com/liubuing/ProteinStudio](https://github.com/liubuing/ProteinStudio)
> **模型下载**：[huggingface.co/YueHuLab/AntibodyDesignBFN](https://huggingface.co/YueHuLab/AntibodyDesignBFN/)
