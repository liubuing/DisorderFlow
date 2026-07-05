# DisOrderFlow 完整说明书

## 从零理解抗体设计 AI 的底层原理与使用方法

---

> **适用读者**: 计算机背景，无生物学或蛋白质结构经验。
> **读完你将能**: 理解 BFN 如何生成抗体、为什么 disorder conditioning 失败、以及如何自己运行整个管线。
> **预计阅读时间**: 45-60 分钟。

---

## 目录

1. [你需要知道的生物学基础（最简版）](#1-生物学基础最简版)
2. [什么是 Bayesian Flow Network (BFN)](#2-bayesian-flow-network)
3. [DisOrderFlow 架构全景](#3-disorderflow-架构全景)
4. [训练：模型如何学会设计抗体](#4-训练流程)
5. [方式4 的 thesis 与三次失败](#5-方式4-的叙事弧线)
6. [S0 观测性预检（训前闸门）](#6-s0-观测性预检)
7. [§5 降级管线（没有 AI 也能产出候选）](#7-5-降级管线)
8. [完整运行指南](#8-完整运行指南)
9. [结果解读与论文方向](#9-结果解读)
10. [常见问题](#10-常见问题)

---

## 1. 生物学基础（最简版）

### 1.1 抗体是什么

把抗体想象成**一把只有你能用的钥匙**。你的免疫系统会为每一种入侵的病原体（病毒、细菌、毒素）定制一把专属钥匙。病原体表面的特定区域叫**抗原 (antigen)**，抗体上与抗原结合的区域叫**CDR (互补决定区)**。

```
        抗体 (钥匙)
       ┌──────────┐
       │  Framework│  ← 骨架区域，所有抗体几乎一样
       │  ┌──────┐│
       │  │ CDRs ││  ← 6个环 (H1/H2/H3 + L1/L2/L3)
       │  │ 🔑🔑🔑││     这里是抗体唯一不同的地方
       │  └──────┘│
       │  Framework│
       └────┬─────┘
            │ 结合
       ┌────┴─────┐
       │  Epitope │  ← 抗原表面抗体接触的区域
       │  抗原     │
       └──────────┘
```

关键数字:
- 抗体由 **重链 (H)** 和 **轻链 (L)** 两条蛋白链组成
- CDR 共有 6 个环: H1, H2, H3 (重链) + L1, L2, L3 (轻链)
- **CDR-H3** 是最多变、最重要的环，长度 2-30 个氨基酸
- 框架区域 (framework) 在不同抗体间高度保守

### 1.2 氨基酸：蛋白质的字母表

蛋白质是氨基酸串成的链。自然界有 **20 种标准氨基酸**，每种有不同的物理化学性质：

```
疏水的 (怕水，喜欢躲在蛋白内部):
  A(Ala) V(Val) L(Leu) I(Ile) M(Met) F(Phe) W(Trp)

亲水的 (喜欢水，倾向在蛋白表面):
  D(Asp) E(Glu) K(Lys) R(Arg) S(Ser) T(Thr) N(Asn) Q(Gln)

特殊的:
  G(Gly) — 最小，最灵活，像铰链
  P(Pro) — 刚性，打断螺旋
  C(Cys) — 能形成二硫键 (分子内的"钉子")
  Y(Tyr) H(His) — 芳香族，参与π-π堆积
```

**关键直觉**: 一条蛋白序列就是这 20 个字母组成的字符串。不同的字母排列 = 不同的三维折叠 = 不同的功能。

### 1.3 蛋白质的三维结构

氨基酸链会自动折叠成特定的三维形状。这个折叠过程是物理驱动的（疏水残基向内包埋，亲水残基面向水环境）。

```
序列:        D A E F R H D S G Y E V ...
             ↓ 折叠
结构:        三维坐标 (每个原子的 x, y, z)
             ↓
功能:        结合抗原 / 催化反应 / 传递信号
```

**为什么这很重要**: AI 设计抗体 = 设计一串氨基酸序列，使得它折叠后能和特定抗原结合。序列 → 结构 → 功能。

### 1.4 无序蛋白 (IDP) 和 Aβ42

大多数蛋白质有固定的三维结构（像一把钥匙有固定形状）。但有些蛋白**天然就是无序的**——它们在溶液中不断变换形状，像一根煮熟的意大利面条。

**Aβ42 (Amyloid-beta 42)** 就是一种部分无序的蛋白。它是阿尔茨海默病的关键分子：
- 在健康大脑中，Aβ42 被正常清除
- 在 AD 患者中，Aβ42 聚集形成有毒的寡聚体和纤维
- 针对 Aβ42 的抗体是重要的治疗策略

**为什么针对 IDP 设计抗体特别难**: 
- 正常抗原: 固定形状 → 抗体可以精确匹配 (钥匙和锁)
- IDP 抗原: 不断变形 → 抗体的 CDR 需要一定的**柔性**来适应目标

这就是 DisOrderFlow 的出发点。但正如我们后面会看到的，这个直觉**在数据上不成立**。

---

## 2. Bayesian Flow Network (BFN)

### 2.1 直觉：从噪声中逐步"雕刻"出序列

BFN 是一种**生成模型**。它的核心思想很简单:

```
步骤 0:    ?????  (完全随机)
步骤 5:    A?CD?  (开始出现模式)
步骤 10:   AACDE  (轮廓清晰)
步骤 15:   AACDE  (确定)
步骤 20:   AACDE  (最终序列)
```

这就像雕塑家从一块石头中逐步雕刻出雕像。每一步去掉一些不确定性，最终得到一个确定的序列。

### 2.2 和 Diffusion Model 的区别

你可能听说过 Stable Diffusion (图像生成)。BFN 是 Diffusion Model 的表亲:

| | Diffusion Model | BFN (Bayesian Flow Network) |
|---|---|---|
| 处理什么 | 连续值 (像素 RGB) | 离散值 (氨基酸类型) + 连续值 (3D坐标) |
| 噪声类型 | 高斯噪声 | 分类分布 → 均匀分布 (序列) |
| 步数 | 通常 1000 | 通常 20-100 |
| 适合 | 图像、音频 | **蛋白质序列+结构联合生成** |

### 2.3 BFN 的数学直觉 (最小公式版)

对于序列生成 (20种氨基酸 + 未知):

1. **定义参数的先验分布**: 最初我们对每个位置的氨基酸没有偏好 → 均匀分布
2. **Sender 加噪声**: 给定真实序列 x，在时间 t 采样一个带噪声的版本 y
3. **Receiver 去噪**: 神经网络学习从 y 预测原始 x
4. **推理时**: 从均匀分布开始，逐步用 Receiver 去噪，得到最终序列

用代码的思维理解:
```python
# 训练时
x_true = "ACDEF"  # 真实抗体序列
t = 0.5           # 当前时间步 (0=起始, 1=终止)
y_noisy = sender(x_true, t)   # 加噪声
x_pred = receiver(y_noisy, t)  # 神经网络预测
loss = cross_entropy(x_pred, x_true)  # 计算损失

# 推理时 (生成新抗体)
y = uniform_distribution()     # 起始: 完全随机
for t in [0.05, 0.10, ..., 1.0]:
    x_pred = receiver(y, t)    # 预测去噪结果
    y = sender(x_pred, t+Δt)  # 在下一个时间步重新加噪声
# 最终 x_pred 就是生成的抗体序列
```

### 2.4 BFN 的三条"流"

DisOrderFlow 的 BFN 同时处理三种信息:

| 流 | 类型 | 含义 | 维度 |
|---|---|---|---|
| **sequence** | 离散 (21类) | 20种AA + 未知 | (N, L, 21) |
| **position** | 连续 (3D) | CA原子的 x,y,z 坐标 | (N, L, 3) |
| **orientation** | 连续 (SO3) | 每个残基的旋转方向 | (N, L, 3, 3) |
| **sidechain** | 连续 (4角度) | 侧链的χ1-χ4二面角 | (N, L, 4) |

其中 N = batch size, L = 序列长度。

---

## 3. DisOrderFlow 架构全景

### 3.1 整体结构

```
输入
  ├── 抗体框架 (framework) 的结构坐标 + 序列
  ├── 抗原 (antigen) 的结构坐标 + 序列  
  └── [可选] epitope disorder profile (每残基的无序程度)

        ↓ 嵌入 (Embedding)
        
  seq_embed + pos_embed + angle_embed + backbone_embed
        ↓ 混合 (ResidualMLP)
  res_feat (N, L, 256)    ← 每个残基的特征向量
  pair_feat (N, L, L, 128) ← 每对残基之间的关系特征
  
        ↓ 编码器 (GAEncoder: 6层 Geometry-Aware Transformer)
        
  每层 GA Block:
    1. 几何注意力: 用3D坐标计算 attention bias
    2. 节点更新: 残基特征 ← 邻居残基的信息
    3. 边更新: 对特征 ← 节点特征的外积
  
        ↓ 输出头 (Prediction Heads)
        
  ┌──────────────┬──────────────┬──────────────┬──────────────┐
  │ head_seq     │ head_pos     │ head_ori     │ head_ang     │
  │ (21类分类)   │ (3D坐标回归) │ (旋转回归)    │ (角度回归)    │
  │ → 氨基酸序列 │ → CA坐标     │ → 方向        │ → 侧链角度    │
  ├──────────────┼──────────────┼──────────────┼──────────────┤
  │ head_plddt   │ head_iptm    │ head_pae     │ head_disorder│
  │ (逐残基置信度)│ (界面置信度) │ (距离误差)    │ (无序预测)    │
  └──────────────┴──────────────┴──────────────┴──────────────┘
```

### 3.2 核心组件详解

#### GAEncoder (Geometry-Aware Transformer)

这是最核心的神经网络。它像 Transformer 但**知道三维结构**:

```
标准 Transformer Attention:
  Attention(Q, K, V) = softmax(QK^T / √d) · V
  
GAEncoder Attention (几何感知):
  Attention(Q, K, V) = softmax(QK^T / √d + B_geo) · V
                                              ↑
                                    几何偏置项:
                                    - 距离编码 (两个残基有多远)
                                    - 方向编码 (它们在空间的相对朝向)
```

这个几何偏置让模型"知道"哪些残基在空间中靠近，即使它们的序列位置很远。

#### 抗原→CDR 交叉注意力 (Direction D, V17+)

最关键的创新之一:

```
传统方法: CDR 残基只看自己的特征 → 不知道抗原长什么样
交叉注意力: CDR 残基 QUERY 抗原残基的特征 → 模型学会"看"抗原
```

实现上:
```python
# CDR 残基的特征作为 Query
# 抗原残基的特征作为 Key 和 Value
cdr_aware = cross_attention(
    query=cdr_features,      # "我是什么样？"
    key=antigen_features,    # "抗原长什么样？"
    value=antigen_features   # "抗原的哪些信息对我有用？"
)
```

#### Contrastive Head (Direction H, V17h)

强制模型学习"这个 CDR 是否匹配这个抗原":

```
正样本: 真实 (抗体, 抗原) 对 → label=1
负样本: 随机打乱的 CDR + 抗原 → label=0

模型必须学会区分:
  ✓ CDR-H3 "ARDGTTVYYYYGMDV" + Aβ N-term  → 匹配！
  ✗ CDR-H3 "GGGGSSSSYYYY"     + Aβ N-term  → 不匹配！

这个任务比序列恢复 (seq recovery) 更能驱动抗原特异性学习
```

### 3.3 代码中的关键类

| 类名 | 文件 | 职责 |
|---|---|---|
| `AntibodyBFN_Core` | `disorderflow/modules/bfn/core.py` | BFN 训练/推理的主入口 |
| `AntibodyBFN_Receiver` | `disorderflow/modules/bfn/receiver.py` | 神经网络：编码→所有预测头 |
| `GAEncoder` | `disorderflow/modules/encoders/ga.py` | 几何感知 Transformer 编码器 |
| `CategoricalBFN` | `disorderflow/modules/bfn/sequence.py` | 序列的 Bayesian 扩散过程 |
| `PositionBFN` | `disorderflow/modules/bfn/position.py` | 坐标的 Bayesian 扩散过程 |
| `AntibodyBFN` | `disorderflow/models/bfn_model.py` | 模型封装 (sample/forward) |
| `SAbDabDataset` | `disorderflow/datasets/sabdab.py` | SAbDab 抗体数据库加载器 |

---

## 4. 训练流程

### 4.1 训练数据

DisOrderFlow 使用 **SAbDab** (Structural Antibody Database):
- ~5000 个抗体-抗原复合物的三维结构
- 每个样本包含:
  - 抗体框架 (framework) 的序列 + 3D坐标
  - CDR 的真实序列 (训练目标)
  - 抗原 (antigen) 的序列 + 3D坐标

数据预处理:
```
PDB 文件 (蛋白质数据库格式)
  ↓ Bio.PDB 解析
Chothia 抗体编号 (标准化的残基编号方案)
  ↓ CDR 标记
LMDB 数据库 (高效的键值存储)
  ↓ DataLoader
训练 batch
```

### 4.2 损失函数 (Loss Functions)

模型同时优化多个目标:

```python
total_loss = (
    1.00 * seq_loss      +  # 序列重建: cross-entropy (最重要)
    0.50 * pos_loss      +  # 坐标预测: MSE
    0.50 * rot_loss      +  # 方向预测: SO3损失
    0.05 * disorder_loss +  # 无序预测: Huber (仅当有 disorder label)
    0.10 * contact_loss  +  # 接触预测: BCE (Direction F)
    0.03 * contrastive_loss + # 对比学习: BCE (Direction H)
    0.03 * diversity_loss +  # 多样性: 惩罚低熵 (V20+)
    0.02 * anti_degen_loss   # 反退化: 惩罚6-mer重复 (V20+)
)
```

### 4.3 训练配置

```bash
# V20 典型训练命令
python train.py \
  --config configs/train/bfn_v20_amplify_xpu.yml \
  --device cuda \
  --batch_size 8 \
  --gradient_accumulation 4 \
  --lr 1e-4 \
  --warmup 500 \
  --max_iter 4000
```

训练约需 4-8 小时 (RTX 5060 8GB)。

### 4.4 为什么 V18→V20 是突破

| 版本 | head_seq | encoder | 结果 |
|---|---|---|---|
| V18 | ❄️ 冻结 | ❄️ 冻结 | unique_aa=5.3, 6-mer repeats=15/45 |
| V19 | ❄️ 冻结(bug) | ❄️ 冻结 | 与V18相同 (bug确认) |
| **V20** | 🔥 解冻(0.1×lr) | 🔥 部分解冻(0.01×lr) | **unique_aa=15.9, 6-mer repeats=0** |

**关键教训**: 冻结序列预测头 = 模型退化到只输出少数几种氨基酸。必须让 head_seq 参与训练。

---

## 5. 方式4 的叙事弧线

### 5.1 核心假设

> "抗 IDP 抗体的 CDR-H3 应该比抗折叠抗原的抗体更柔，且柔性应与表位无序程度正相关。"

类比: 抓一个不断变形的泥鳅 (IDP)，你的手指需要灵活；抓一个固定的苹果 (折叠蛋白)，手指可以僵硬。

### 5.2 三次失败

```
V8-V12: entropy 当 plasticity 代理
  → 序列熵 ≠ 结构柔性 (消融证伪)
  
V13-V17: BFN disorder head
  → AUC=0.47 (标准基准无效)
  → 1.0 红斑是训练集泄露
  
V18-V20: 训练信号
  → r=+0.309 (V19) = frozen head_seq artifact
  → r=-0.449 (V20) = 信号方向反于预测
  → 消融证明 = diversity loss 副产物
```

### 5.3 根本原因

1. **数据墙**: SAbDab 中 99.93% 是折叠抗原，只有 0.07% IDP
2. **代理错误**: entropy ≠ RMSF (真实的物理柔性)
3. **评估墙**: CDR 侧三路计算验证全死
4. **Loss artifact**: 任何信号消融后消失

---

## 6. S0 观测性预检

### 6.1 设计理念

**在花任何训练算力之前，先用真实数据验证你的核心假设。** 如果假设在观测数据上不成立，就不要训练。

### 6.2 方法

```
收集已知抗 IDP 抗体 (8 个 anti-Aβ + anti-tau + anti-αSyn)
  ↓ 下载 PDB 结构
  ↓ 提取 CDR-H3 (Chothia 编号 95-102)
  ↓ 提取 CDR-H3 的 B-factor (晶体学温度因子 = 柔性代理)
  
收集对照抗体 (抗折叠抗原，如溶菌酶/VEGF/HER2)
  ↓ 同样提取 CDR-H3 B-factor
  
统计检验:
  1. Mann-Whitney U: IDP 组的 CDR-H3 B-factor > 折叠组？
  2. Spearman 相关: epitope disorder ↔ CDR-H3 B-factor？
```

### 6.3 CDR-H3 的自动识别

由于不是所有 PDB 都有 Chothia 编号，代码实现了三层回退:

```
1. Chothia 自动检测:
   - 在 VH 域 (前 ~125 残基) 检查:
     - Cys 在 resseq 20-26? (保守的 FR1 Cys)
     - Cys 在 resseq 88-99? (保守的 FR3 Cys)
   - 两者都有 → Chothia 编号 → CDR-H3 = resseq 95-102

2. 模式匹配 (回退):
   - 找到 WGxG 模体 (FR4 起点)
   - 向前找到最近的 Cys (FR3 Cys)
   - CDR-H3 = Cys+3 到 WGxG-1

3. ANARCI (如果已安装):
   - 标准抗体编号工具
```

### 6.4 运行 S0

```bash
python way4_s0/collect_anti_idp.py --control-n 12
```

输出: `way4_s0_observational.json`

### 6.5 S0 结果 (2026-07-05)

```
Anti-IDP CDR-H3 B-factor:  24.69 ± 8.50  (n=5)
Folded  CDR-H3 B-factor:  35.87 ± 21.69 (n=11)
Mann-Whitney U: p = 0.827  (不显著)

Spearman r(epitope_diso, CDR-H3_Bfactor) = -0.327
p = 0.253  (不显著，且方向为负)
```

**判决: [FALSIFIED]** — thesis 在晶体学 B-factor 数据上不成立。

---

## 7. §5 降级管线

### 7.1 策略

既然 AI 生成路径不成立，改用:

1. **模板种子重设计**: 从已知抗 Aβ 抗体出发，只重新设计非热点残基
2. **已知文库**: 收集文献中所有抗 IDP 抗体的 CDR 序列
3. **柔度匹配筛选**: 用 disorder 作为筛选条件，不是生成驱动

### 7.2 Pillar A: 模板种子重设计

```
输入:
  4HIX (抗 Aβ N-term)  或  5CSZ (抗 Aβ mid-domain)
  ↓
识别热点 (hotspot = CDR 残基与抗原距离 < 6Å)
  ↓
冻结热点 (保持原氨基酸不变)
  ↓
ProteinMPNN 重设计非热点 CDR 位置
  ↓
评分: complexity + entropy + anti-degen
  ↓
输出: 排名变体列表
```

运行:
```bash
python run_v4_pillar_a.py --samples 100 --top 20
```

### 7.3 完整管线

```bash
python way4_s5_fallback_pipeline.py --top 30
```

流程:
```
Source 1: 已知抗 IDP 抗体 CDR (8 literature + 7 PDB)
Source 2: Pillar A 模板种子变体 (69 variants)
Source 3: 晶体模板 CDR
  ↓
去重 (移除相同序列)
  ↓
质量过滤:
  - anti-degen check (无 >4 连续相同氨基酸)
  - CDR 长度 ≥ 4
  - 溶解度 ≥ 0.4
  - 免疫原性风险 ≤ 0.7
  ↓
柔度匹配评分 (Way4 selection filter):
  - CDR disorder ↔ epitope disorder 匹配度
  - Gly/Ser 含量 (柔性代理)
  - 疏水性 (适度的疏水相互作用)
  - 芳香族含量 (π-π 堆积)
  ↓
复合排名 (weighted sum):
  30% pliability + 20% entropy + 20% complexity
  + 15% solubility + 10% immunogenicity + 5% anti-degen
  ↓
输出: Top-30 候选 (JSON), 准备送 SPR/BLI 湿实验
```

---

## 8. 完整运行指南

### 8.1 环境安装

```bash
# 1. 克隆仓库
git clone <repo-url>
cd DisorderFlow

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install torch numpy scipy biopython
pip install -e .

# 4. 下载预训练模型权重 (需要配置)
#    app_config.yaml 中指定 checkpoint 路径
```

### 8.2 硬件要求

| 任务 | 最低 | 推荐 |
|---|---|---|
| S0 数据采集 | CPU, 1GB RAM | 任何机器 |
| BFN 训练 | GPU 8GB VRAM | RTX 5060+ |
| AF2 折叠 | GPU 12GB+ VRAM 或 CPU | 云 GPU |
| MD 模拟 | CPU 4核+ | 云 CPU 集群 |

### 8.3 运行 S0 (必做第一步)

```bash
# 1. 采集抗 IDP 抗体数据 + 统计检验
python way4_s0/collect_anti_idp.py --control-n 12

# 2. 查看结果
python way4_s0/final_report.py
```

输出文件: `way4_s0_observational.json`

### 8.4 运行 Pillar A (模板种子设计)

```bash
# 确保数据文件存在:
ls data/anti_abeta_refs/4HIX.pdb   # ✓
ls data/anti_abeta_refs/5CSZ.pdb   # ✓
ls data/misfolding_targets/3STB.pdb # ✓

# 运行
python run_v4_pillar_a.py --samples 100 --top 20
```

输出: `idp_design_results/pillar_a_variants_<timestamp>.json`

### 8.5 运行完整 §5 管线

```bash
python way4_s5_fallback_pipeline.py --top 30
```

输出: `idp_design_results/s5_wetlab_candidates.json`

### 8.6 候选解读

```python
import json

with open('idp_design_results/s5_wetlab_candidates.json') as f:
    data = json.load(f)

for i, c in enumerate(data['top_candidates'][:10]):
    print(f"#{i+1}: {c['cdr_sequence'][:40]}...")
    print(f"     Score={c['composite_score']:.3f} "
          f"Pliability={c['pliability_score']:.3f}")
    print(f"     Source: {c['source_antibody']}")
```

---

## 9. 结果解读

### 9.1 评分维度

| 分数 | 范围 | 含义 |
|---|---|---|
| composite_score | [0, 1] | 综合排名，越高越好 |
| pliability_score | [0, 1] | CDR 柔度与表位无序的匹配程度 |
| entropy | [0, ~3] | CDR 序列多样性 (log(20)≈3.0为最大) |
| complexity | [0, 1] | CDR 复杂度 (惩罚单氨基酸主导和同聚体) |
| solubility | [0, 1] | 预测可溶性 (越高越容易表达) |
| immunogenicity_risk | [0, 1] | 免疫原性风险 (越低越安全) |

### 9.2 什么样的候选是好的

```
好候选:
  composite_score > 0.55
  pliability_score > 0.70
  entropy > 1.5 (足够多样)
  solubility > 0.55
  immunogenicity_risk < 0.6
  anti_degen = True

可疑候选:
  entropy < 0.8 (可能退化)
  连续相同氨基酸 > 4 (可能形成聚集体)
  solubility < 0.4 (可能不表达)
  immunogenicity_risk > 0.7 (可能有免疫原性)
```

### 9.3 当前可发表的论文

见 [DISORDERFLOW_COMPLETE_GUIDE.md 第9节](#)（略，在用户之前的对话中已讨论）：

| 能写 | 不能写 |
|---|---|
| S0 观测框架 (方法论贡献) | "我们设计了结合Aβ42的抗体" (无湿实验) |
| Diversity/contrastive 突破 | "disorder conditioning 有效" (S0证伪) |
| 完整方法学负结果 | 任何 binding/affinity 主张 |

---

## 10. 常见问题

### Q1: 为什么 B-factor 而不是直接测柔性？

B-factor (温度因子) 是晶体学中的"原子的热振动程度"。B-factor 高 = 原子抖得厉害 = 那个区域比较柔。但 B-factor 也受晶体堆积影响，不是完美的柔性指标。真正测柔性需要做分子动力学 (MD) 模拟，但那很慢。

### Q2: 为什么不用 AlphaFold3？

AF3 是最新的蛋白质结构预测工具，但它对短肽（如 Aβ42 片段）的预测仍不可靠，且需要商业许可。本方案使用 AF2 (ColabFold) 作为替代。

### Q3: 为什么 CDR-H3 这么重要？

CDR-H3 是抗体 CDR 中最长、最多变的环。它通常贡献最多结合能量，也是最难设计的。其他 CDR (H1/H2/L1/L2/L3) 相对保守。

### Q4: 为什么不做端到端的深度学习 (AlphaFold→binding prediction)？

因为 **评估墙**: 
- Rigid-body docking (HDOCK) 不考虑 CDR 构象变化 → 不准确
- AF2 multimer 在 RTX 5060 (8GB) 上 OOM
- 短肽 refold 在 AF2 中失效 (需要 >50 残基才有可靠折叠)

### Q5: 那到底怎么验证抗体能不能结合？

**湿实验是唯一真尺。** SPR (Surface Plasmon Resonance) 或 BLI (Bio-Layer Interferometry) 直接测量抗体与抗原的结合强度 (KD 值)。计算只能做筛选，不能做验证。

### Q6: 如果我想继续训练 BFN，有什么建议？

按 V14 方案铁律：
1. **先过 S0**: 用你想训练的任务的观测数据验证 thesis
2. **不用 entropy 当代理**: 用真实的物理量 (RMSF/B-factor/MD)
3. **消融是必须的**: 任何信号必须在去 conditioning 后消失，才是真实信号
4. **PENDING 不当成立**: 消融没过就只是 pending，不能做主张

---

## 附录 A: 关键文件索引

```
DisOrderFlow/
├── train.py                              # 训练主循环
├── disorderflow/
│   ├── modules/bfn/
│   │   ├── core.py                       # BFN 核心 (训练/推理)
│   │   ├── receiver.py                   # 神经网络 + 所有预测头
│   │   ├── receiver_heads.py             # 置信度头 (pLDDT/ipTM/PAE)
│   │   ├── sequence.py                   # CategoricalBFN (序列流)
│   │   ├── position.py                   # PositionBFN (坐标流)
│   │   └── core_losses.py               # 分组损失 (V14+)
│   ├── modules/encoders/
│   │   └── ga.py                         # Geometry-Aware Transformer
│   ├── datasets/
│   │   └── sabdab.py                     # SAbDab 数据加载
│   └── utils/protein/
│       ├── constants.py                  # AA/Chothia/原子常量
│       └── parsers.py                    # PDB 解析
├── way4_s0/
│   ├── collect_anti_idp.py               # S0 数据采集 + CDR-H3 提取
│   └── final_report.py                   # S0 结果汇总
├── run_v4_pillar_a.py                    # Pillar A 模板种子设计
├── way4_s5_fallback_pipeline.py          # §5 完整降级管线
└── app_config.yaml                       # 运行时配置
```

## 附录 B: 术语表

| 术语 | 英文 | 解释 |
|---|---|---|
| 抗体 | Antibody | 免疫系统产生的结合蛋白 |
| 抗原 | Antigen | 抗体结合的靶标 |
| CDR | Complementarity-Determining Region | 抗体上决定结合特异性的环 |
| CDR-H3 | Heavy chain CDR 3 | 最重要、最多变的 CDR |
| 表位 | Epitope | 抗原上抗体实际接触的区域 |
| B因子 | B-factor | 晶体学中表示原子热运动的参数 |
| RMSF | Root Mean Square Fluctuation | 分子动力学中残基的均方根波动 |
| IDP | Intrinsically Disordered Protein | 天然无序蛋白 |
| Aβ42 | Amyloid-beta 42 | 阿尔茨海默病相关的无序肽 |
| BFN | Bayesian Flow Network | 本方案使用的生成模型 |
| GA | Geometry-Aware | 几何感知 (知道3D结构的神经网络) |
| MPNN | Message Passing Neural Network | ProteinMPNN, 蛋白质序列设计工具 |
| AF2 | AlphaFold2 | 蛋白质结构预测工具 |
| SPR | Surface Plasmon Resonance | 测量结合强度的湿实验方法 |
| BLI | Bio-Layer Interferometry | 另一种结合强度测量方法 |
| KD | Dissociation Constant | 解离常数，结合强度的量化指标 |
| PDB | Protein Data Bank | 蛋白质结构数据库 |
| SAbDab | Structural Antibody Database | 抗体结构数据库 |
| LMDB | Lightning Memory-Mapped Database | 高效键值存储格式 |
| Chothia | Chothia numbering | 一种标准化的抗体残基编号方案 |

---

> **最后**: 这份说明书覆盖了 DisOrderFlow 从生物学基础到代码运行的全部内容。如果某个部分仍然不清楚，最有效的学习路径是:
> 1. 先跑一遍 S0 (`python way4_s0/collect_anti_idp.py`)，看输出 JSON
> 2. 再看 Pillar A 的代码 (`run_v4_pillar_a.py`)，理解热点识别和 MPNN
> 3. 最后看 BFN 核心 (`disorderflow/modules/bfn/core.py`)，从 forward() 函数入口追踪数据流
>
> 代码比文档更诚实。
