# AntibodyDesignBFN 蛋白质序列设计平台 — 完整使用说明书

## 平台概述

本平台集成了 **四种** 核心功能，基于给定的蛋白质骨架结构（PDB 格式）进行序列设计与结构预测：

| 功能 | 方法 | 类型 | 适用场景 |
|------|------|------|----------|
| **BFN** | 贝叶斯流网络 + 几何Transformer | 序列设计 | 抗体CDR设计、通用蛋白设计 |
| **ProteinMPNN** | 消息传递神经网络 (Baker Lab) | 序列设计 | 通用反向折叠，快速高恢复率 |
| **ESM-IF** | GVP-GNN + Transformer (Meta FAIR) | 序列设计 | 高精度反向折叠 |
| **AlphaFold2** | ColabFold (ColabDesign) | 结构预测 | 从序列预测3D结构 |

提供 **Web 图形界面**（Gradio）和 **命令行** 两种使用方式。

---

## 一、环境配置

### 1.1 系统要求

- **操作系统**: Windows / Linux / macOS
- **Python**: 3.10+
- **GPU**: 可选（CPU 也可运行，GPU 加速推荐 CUDA）
- **磁盘空间**: 约 15GB（含模型权重 ~5GB）

### 1.2 安装

```bash
cd AntibodyDesignBFN-main

# Windows（Git Bash / MSYS2）
source ../venv/Scripts/activate

# Linux / macOS
source ../venv/bin/activate

# 验证环境
python -c "
import torch; import numpy; import biotite; import esm;
from antibodydesignbfn.models import get_model;
print(f'torch={torch.__version__}, numpy={numpy.__version__}, cuda={torch.cuda.is_available()}');
print('All dependencies OK')
"
```

### 1.3 模型权重

| 模型 | 来源 | 路径/说明 |
|------|------|-----------|
| BFN | HuggingFace | `~/.cache/huggingface/hub/models--YueHuLab--AntibodyDesignBFN/` |
| ProteinMPNN | 内置 | `ProteinMPNN/vanilla_model_weights/v_48_*.pt` |
| ESM-IF | 自动下载 | 首次运行时从 HuggingFace 下载 ~1.6GB |
| AlphaFold2 | ColabFold | venv 中的 `colabfold_batch`，启动时自动检测 |

---

## 二、Web 图形界面（推荐）

### 2.1 启动与停止

```bash
# 前台启动（可直接看到日志）
python manage.py start

# 后台启动（关闭终端后继续运行）
python manage.py start --bg

# 查看运行状态
python manage.py status

# 重启服务
python manage.py restart

# 停止服务
python manage.py stop
```

启动后访问: **http://127.0.0.1:7860**（仅本机，外部无法访问）

### 2.2 界面概览

Web 界面包含四个标签页：

#### Tab 1 — 单步设计

核心工作区，上传 PDB → 选择工具 → 设参数 → 运行设计。

**操作流程：**

1. **上传 PDB 文件** — 拖拽或点击上传 `.pdb` 文件，自动显示链信息和残基序列
2. **选择设计工具** — 四选一：
   - **BFN (抗体CDR设计)** — 针对抗体结构，配置重链/轻链ID
   - **BFN (通用蛋白设计)** — 任意蛋白，指定区域如 `A:10-25`
   - **ProteinMPNN** — 快速反向折叠，配置温度和目标链
   - **ESM-IF** — 高精度反向折叠，配置温度和链
3. **设置参数** — 各工具参数见下方说明
4. **点击"开始设计"** — 运行并在下方显示结果

**各工具参数说明：**

| 工具 | 参数 | 说明 | 默认值 |
|------|------|------|--------|
| BFN 抗体 | 重链ID | 抗体重链标识 | H |
| BFN 抗体 | 轻链ID | 抗体轻链标识 | L |
| BFN 抗体 | 序列数 | 生成候选序列数量 | 3 |
| BFN 抗体 | 随机采样 | 开启=多样性高，关闭=确定性 | 关闭 |
| BFN 抗体 | 评估模式 | 与原始序列对比恢复率 | 开启 |
| BFN 通用 | 设计区域 | `链:起始-结束` 格式 | A:10-25 |
| BFN 通用 | 序列数 | 生成候选序列数量 | 3 |
| ProteinMPNN | 设计链 | 需要设计的链ID | A |
| ProteinMPNN | 温度 | 低=保守，高=多样 | 0.1 |
| ProteinMPNN | 序列数 | 生成候选序列数量 | 3 |
| ESM-IF | 目标链 | 需要设计的链ID | A |
| ESM-IF | 温度 | 0.05-1.0 | 0.1 |
| ESM-IF | 序列数 | 生成候选序列数量 | 3 |

#### Tab 2 — 批量处理

一次上传多个 PDB 文件，自动逐个处理并汇总结果。

**操作流程：**

1. 上传多个 PDB 文件（可多选）
2. 选择工具和参数
3. 点击"开始批量处理"
4. 结果保存到 `batch_results/` 目录

#### Tab 3 — 结构预测 (AlphaFold2)

从氨基酸序列预测蛋白质 3D 结构，预测结果可一键发送到设计流程。

**操作流程：**

1. **输入序列** — 粘贴 FASTA 格式序列，或上传 `.fasta`/`.fa`/`.txt` 文件
   ```
   >protein_name
   MSDRPTARRWGKCGPLCTRENIMV...
   ```
2. **设置参数：**
   | 参数 | 说明 | 默认值 |
   |------|------|--------|
   | 模型数 | ColabFold 使用的模型数量（1-5） | 3 |
   | 回收次数 | 结构精修迭代次数（0-12） | 3 |
   | 随机采样 | 开启以增加结构多样性 | 关闭 |

3. **点击"开始预测"** — 后台调用 ColabFold，需等待数分钟
4. **点击"发送到设计"** — 将预测的 PDB 结构自动加载到 Tab 1 的设计流程

**注意事项：**
- AlphaFold2 使用 ColabFold 本地版本，不需要网络连接
- 预测时间取决于序列长度和参数设置（通常 3-30 分钟）
- 多序列预测会消耗较多内存
- 输出 PDB 保存在 `alphafold_results/` 目录

#### Tab 4 — 设置

- **系统状态** — 查看各模型加载状态、端口状态、CUDA 可用性
- **模型预加载** — 手动加载 ESM-IF 或重载 BFN 模型
- **配置文件编辑** — 在线编辑 `app_config.yaml`，保存后即时生效

---

## 三、命令行工具

### 3.1 服务管理 (manage.py)

```bash
python manage.py start              # 启动 Web 服务
python manage.py start --bg         # 后台启动
python manage.py stop               # 停止服务
python manage.py restart            # 重启服务
python manage.py status             # 查看服务状态
python manage.py config             # 显示当前配置
python manage.py config --edit      # 在编辑器中打开配置
python manage.py test               # 运行自检（验证所有工具可用）
```

### 3.2 BFN: 抗体 CDR 设计 (design_seq.py)

针对抗体结构设计互补决定区（CDR）序列。

```bash
python design_seq.py <PDB文件> --heavy <重链> --light <轻链> --config <配置> [选项]
```

| 参数 | 说明 | 示例 |
|------|------|------|
| `pdb_path` | PDB 文件路径 | `7DK2_AB_C.pdb` |
| `--heavy` | 重链 ID | `--heavy A` |
| `--light` | 轻链 ID | `--light B` |
| `--config` | 配置文件 | `--config configs/demo_design.yml` |
| `--device` | 运行设备 | `--device cpu` 或 `--device cuda` |
| `--num_samples` | 生成序列数 | `--num_samples 10` |
| `--stochastic` | 随机采样 | `--stochastic` |
| `--eval` | 与原始序列对比 | `--eval` |
| `--output` | 输出 JSON 路径 | `--output result.json` |
| `--skip_renumber` | 跳过 Chothia 重编号 | `--skip_renumber` |

**示例：**

```bash
# 基本用法
python design_seq.py 7DK2_AB_C.pdb --heavy A --light B --config configs/demo_design.yml --device cpu

# 随机采样 5 条，评估模式
python design_seq.py 7DK2_AB_C.pdb --heavy A --light B --config configs/demo_design.yml --num_samples 5 --stochastic --eval

# 跳过重编号（PDB 已为 Chothia 编号）
python design_seq.py 7DK2_AB_C_chothia.pdb --heavy A --light B --config configs/demo_design.yml --skip_renumber
```

### 3.3 BFN: 通用蛋白质设计 (design_protein.py)

按残基位置精确指定设计区域，适用于任意蛋白。

```bash
python design_protein.py <PDB文件> --design "<链:位置>" --config <配置> [选项]
```

**设计区域语法 `--design`：**

| 语法 | 说明 |
|------|------|
| `"A:10-25"` | A链第10到25号残基（连续区间） |
| `"A:10,20,30"` | A链指定位置 |
| `"A:10-25 B:5-15"` | 多链不同区域 |
| `"A:10-25,A:40-50"` | 同链不连续区域 |

**示例：**

```bash
# 单链单区域
python design_protein.py 7DK2_AB_C.pdb --design "A:30-40" --config configs/demo_design.yml --device cpu --eval

# 多链设计
python design_protein.py complex.pdb --design "A:10-25 B:5-15" --config configs/demo_design.yml --num_samples 5 --stochastic --eval

# 保存结果到 JSON
python design_protein.py target.pdb --design "A:1-100" --config configs/demo_design.yml --output result.json
```

### 3.4 ProteinMPNN

```bash
python ProteinMPNN/protein_mpnn_run.py \
    --pdb_path <PDB文件> \
    --pdb_path_chains <链ID> \
    --num_seq_per_target <数量> \
    --sampling_temp "<温度>" \
    --path_to_model_weights ProteinMPNN/vanilla_model_weights \
    --out_folder <输出目录> \
    --save_score 1
```

**常用选项：**

| 选项 | 说明 |
|------|------|
| `--model_name` | 模型版本：`v_48_002`/`v_48_010`/`v_48_020`/`v_48_030` |
| `--seed` | 随机种子（0=随机） |
| `--omit_AAs` | 排除氨基酸，如 `"CX"` 排除 Cys |
| `--use_soluble_model` | 使用可溶蛋白模型 |
| `--ca_only` | 仅 CA 原子模式 |

**示例：**

```bash
# 基本用法
python ProteinMPNN/protein_mpnn_run.py \
    --pdb_path 7DK2_AB_C.pdb --pdb_path_chains "A" \
    --num_seq_per_target 3 --sampling_temp "0.1" --seed 42 \
    --out_folder mpnn_output --save_score 1 \
    --path_to_model_weights ProteinMPNN/vanilla_model_weights

# 排除半胱氨酸
python ProteinMPNN/protein_mpnn_run.py \
    --pdb_path protein.pdb --pdb_path_chains "A" \
    --omit_AAs "C" --num_seq_per_target 3 \
    --path_to_model_weights ProteinMPNN/vanilla_model_weights

# 可溶蛋白专用模型
python ProteinMPNN/protein_mpnn_run.py \
    --pdb_path protein.pdb --pdb_path_chains "A" \
    --use_soluble_model \
    --path_to_model_weights ProteinMPNN/soluble_model_weights
```

### 3.5 ESM-IF

ESM-IF 通过 Python API 调用，无独立 CLI 脚本。在 Web 界面中使用最为便捷。

```python
from esm.pretrained import esm_if1_gvp4_t16_142M_UR50
from esm.inverse_folding import util as if_util

model, _ = esm_if1_gvp4_t16_142M_UR50()
model = model.eval()

coords, native_seq = if_util.load_coords('protein.pdb', 'A')
sampled = model.sample(coords, temperature=0.1)
print(sampled)
```

### 3.6 集成对比测试 (integration_test.py)

一键运行三种工具并生成对比表格。

```bash
# 完整三工具对比
python integration_test.py 7DK2_AB_C.pdb --chain A --region "30:40" --config configs/demo_design.yml --num_samples 3

# 仅部分工具
python integration_test.py protein.pdb --chain A --region "10:25" --skip_esmif
python integration_test.py protein.pdb --chain B --region "5:15" --skip_bfn --skip_mpnn
```

---

## 四、配置文件详解 (app_config.yaml)

修改 `app_config.yaml` 可调整所有工具的默认行为和模型路径。

```yaml
# 服务设置
server:
  host: "127.0.0.1"       # 监听地址（0.0.0.0=允许外部访问）
  port: 7860               # 端口号
  share: false             # Gradio 公网分享
  auto_open: true          # 启动后自动打开浏览器

# 模型路径
models:
  bfn:
    checkpoint: "C:/Users/.../best.pt"   # BFN 模型权重路径
  proteinmpnn:
    weights_dir: "ProteinMPNN/vanilla_model_weights"
    model_name: "v_48_020"
  esmif:
    model_name: "esm_if1_gvp4_t16_142M_UR50"

# 默认参数
bfn_defaults:
  antibody:
    cdrs: ["H_CDR1","H_CDR2","H_CDR3","L_CDR1","L_CDR2","L_CDR3"]
    num_samples: 3
  protein:
    region: "A:10-25"
    num_samples: 3

mpnn_defaults:
  temperature: "0.1"
  seed: 42

esmif_defaults:
  temperature: 0.1
  num_samples: 3

# AlphaFold2 结构预测
alphafold:
  af2:
    venv: "C:\\biological\\AntibodyDesignBFN-main\\venv"
    executable: "colabfold_batch"
  defaults:
    num_models: 3
    num_recycle: 3
  output_dir: "alphafold_results"

# 批量处理
batch:
  output_dir: "batch_results"
  max_workers: 2
  timeout_per_job: 300
```

**修改方式：**
- Web 界面: Tab 4 "设置" 中在线编辑
- 命令行: `python manage.py config --edit`
- 手动: 直接编辑 `app_config.yaml` 文件

---

## 五、工具选择指南

| 需求场景 | 推荐工具 | 理由 |
|----------|----------|------|
| 抗体 CDR 设计 | BFN (抗体模式) | 专门训练于抗体结构 |
| 通用蛋白定点突变 | BFN (通用模式) | 灵活的区域指定，多样候选 |
| 蛋白骨架全序列设计 | ProteinMPNN 或 ESM-IF | 反向折叠专用，恢复率高 |
| 可溶蛋白设计 | ProteinMPNN `--use_soluble_model` | 专门的可溶蛋白权重 |
| 需要序列多样性 | BFN `--stochastic` | 生成式模型天然支持多样采样 |
| 高序列恢复率 | ESM-IF (低温 0.1) | GVP 架构精确编码结构约束 |
| 仅 CA 原子 | ProteinMPNN `--ca_only` | 支持仅 CA 坐标输入 |
| 快速批量设计 | ProteinMPNN | 速度最快（秒级） |
| 从序列预测结构 | AlphaFold2 | ColabFold 本地推理 |
| 结构→设计一键流程 | Web Tab3 → Tab1 | 预测后直接发送到设计 |

---

## 六、结果解读

### 6.1 序列设计结果

| 指标 | 含义 | 评价标准 |
|------|------|----------|
| **PPL (Perplexity)** | 困惑度，模型对该序列的确信程度 | 越低越好（注意：低PPL≠绝对好） |
| **Recovery (恢复率)** | 与原始/天然序列的氨基酸一致率 | 0-100%，越高越接近原始 |
| **Score** | ProteinMPNN 的负对数概率 | 越低越好 |

### 6.2 结构预测结果

| 指标 | 含义 |
|------|------|
| **pLDDT** | 预测局部距离差异测试，0-100分 |
| **pTM** | 预测模板建模分数 |
| **Rank** | ColabFold 按 pLDDT 排序（rank_001 为最优） |

---

## 七、常见问题

### Q1: 如何获取 BFN 模型权重？
权重自动下载到 `~/.cache/huggingface/hub/`。若下载失败，手动从 https://huggingface.co/YueHuLab/AntibodyDesignBFN 下载后修改 `app_config.yaml` 中的 checkpoint 路径。

### Q2: 设计区域编号如何确定？
使用 PDB 文件中的残基编号。可在 Web 界面上传 PDB 后自动显示，或用 PyMOL/Biopython 查看。

### Q3: CPU vs GPU？
- CPU: Web 界面默认使用 CPU，适合小批量设计
- GPU: 修改配置文件中的设备设置，或在 CLI 中使用 `--device cuda`

### Q4: 采样温度如何选？
- 低温 (0.05-0.15): 结果保守，接近原始序列
- 中温 (0.2-0.5): 平衡多样性与合理性
- 高温 (0.5-1.0): 序列多样，但可能不合理

### Q5: AlphaFold2 预测失败？
- 确认 venv 路径配置正确
- 确认 `colabfold_batch` 可执行
- 运行 `python manage.py test` 自检
- 查看 `server.log` 获取详细错误信息

### Q6: Web 界面无法访问？
- 确认服务已启动: `python manage.py status`
- 确认访问地址是 `http://127.0.0.1:7860`（不是 localhost）
- 检查端口是否被占用: 修改 `app_config.yaml` 中的 port

### Q7: ProteinMPNN 报路径错误？
需指定 `--path_to_model_weights ProteinMPNN/vanilla_model_weights`（相对路径）。

---

## 八、项目文件结构

```
AntibodyDesignBFN-main/
├── app.py                        # Gradio Web 应用主程序
├── manage.py                     # 服务管理工具
├── app_config.yaml               # 集中配置文件
├── design_seq.py                 # BFN 抗体 CDR 设计 CLI
├── design_protein.py             # BFN 通用蛋白设计 CLI
├── integration_test.py           # 三工具集成对比测试
├── train.py                      # BFN 训练脚本
├── requirements.txt              # Python 依赖
├── setup.py                      # 包安装配置
├── configs/
│   └── demo_design.yml           # BFN 设计配置
├── antibodydesignbfn/            # BFN 核心模块
│   ├── models/                   # 模型架构
│   ├── datasets/                 # 数据预处理
│   └── utils/                    # 工具函数
├── diffab/                       # DiffAb 模块
├── ProteinMPNN/                  # ProteinMPNN 工具
│   ├── protein_mpnn_run.py       # 推理脚本
│   └── vanilla_model_weights/    # 模型权重
├── checkpoints/                  # 模型检查点
└── data/                         # 示例 PDB 文件
```

---

## 九、快速开始（10 分钟上手）

```bash
# 1. 启动 Web 服务
cd AntibodyDesignBFN-main
python manage.py start --bg

# 2. 浏览器打开 http://127.0.0.1:7860

# 3. 在 Tab 1 上传示例 PDB（项目自带 7DK2_AB_C.pdb）

# 4. 选择工具 → 设参数 → 点击"开始设计"

# 5. 查看结果，复制设计的氨基酸序列

# 6. 尝试 Tab 3 输入 FASTA 序列预测结构

# 7. 将预测结构发送到设计流程（一键操作）
```

**命令行快速测试：**

```bash
source ../venv/Scripts/activate

# 自检
python manage.py test

# BFN 抗体设计
python design_seq.py 7DK2_AB_C.pdb --heavy A --light B --config configs/demo_design.yml --device cpu --eval

# BFN 通用设计
python design_protein.py 7DK2_AB_C.pdb --design "A:30-40" --config configs/demo_design.yml --device cpu --eval

# 三工具对比
python integration_test.py 7DK2_AB_C.pdb --chain A --region "30:40" --config configs/demo_design.yml --num_samples 3
```
