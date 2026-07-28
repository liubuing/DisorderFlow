# DisOrderFlow v5 多构象数据集构建 - 交接文档

> 本文档记录构建运行所需的关键环境、配置、修复与运维知识。
> 换对话模型 / 新会话接手时，先读本文档 + `monitor_v5_progress.py` 即可掌握全貌。

---

## 一、当前目标

为 DisOrderFlow 项目构建 **v5 多构象数据集**：
- 对 1301 个蛋白质（L=54-500）各跑 5 个 AF2 种子，生成 5 套坐标
- Kabsch 对齐 → 逐残基 Cα RMSF → `tanh(RMSF/2)` 得连续 disorder 标签
- 输出：`data/confidence_conformation_v5/confidence_train.lmdb`
- 用途：训练 disorder head（连续逐残基标签，取代二分类广播）

**完成进度**：见 `monitor_v5_progress.py` 输出（目标 1301 条目）

---

## 二、运行环境（关键！）

### 为什么必须用 WSL2
**Windows 原生无法用 GPU 跑 jax** —— jax 在 Windows 上没有任何版本的 cuda12-plugin wheel（jaxlib 有 win 包，但 cuda plugin 没有）。所有 GPU 推理必须走 WSL2。

### 环境：`venv_wsl`（WSL2 Ubuntu-24.04）
- 位置：`<project>/venv_wsl`（Python 3.12，venv 由系统 python3.12 创建）
- 已装包：
  - `jax==0.5.3` + `jax[cuda12]`（jaxlib 0.5.3, jax-cuda12-plugin 0.5.3, jax-cuda12-pjrt 0.5.3）
  - `alphafold-colabfold==2.3.13`
  - `colabfold==1.6.1`
  - `lmdb` (2.2.1)
  - `torch==2.5.1`（CPU 实际用不到，但源 LMDB 的 batch 字典含 torch tensor，unpickle 需要）
  - `nvidia-curand-cu12`（torch 2.5.1 的依赖，jax 没自带）
- 验证命令：
  ```bash
  wsl -d Ubuntu-24.04 bash -lc "cd <wsl-project-path> && source venv_wsl/bin/activate && python -c 'import jax; print(jax.devices())'"
  # 应输出 [CudaDevice(id=0)]
  ```

### 模型权重
- 缓存：`~/.cache/colabfold`（可软链到 Windows 侧已有的 params 缓存）
- 如软链丢失，重建：
  ```bash
  wsl -d Ubuntu-24.04 bash -lc "mkdir -p ~/.cache && ln -sfn <windows-cache-wsl-path> ~/.cache/colabfold"
  ```

---

## 三、关键 Bug 修复（必须保留！）

### alphafold-colabfold 2.3.13 的 `num_residues` bug
- 文件：`venv_wsl/lib/python3.12/site-packages/alphafold/model/modules_multimer.py`
- 第 448 行原代码：`L = num_residues`（变量未定义，NameError）
- 已修复为：`L = num_res`（第 423 行定义的 `num_res = batch['aatype'].shape[0]`）
- 修复命令：
  ```bash
  wsl -d Ubuntu-24.04 bash -lc "sed -i 's/L = num_residues/L = num_res/' /mnt/c/biological/DisorderFlow/venv_wsl/lib/python3.12/site-packages/alphafold/model/modules_multimer.py"
  ```
- **重装 alphafold 后此 patch 会丢失**，需重新打

---

## 四、构建脚本与参数

### 主脚本
`scripts/build/build_conformation_cf.py`
- 固定使用 `alphafold2_multimer_v3 model_1`（`af2_jax_runner.py:271`）
- **不要中途换折叠器**（会破坏 RMSF 标签一致性）
- 当前运行参数：
  ```
  --n_seeds 5 --num_recycle 0 --split train
  --gc_pause_sec 3 --max_consecutive_failures 0 --timeout_minutes 90
  ```

### 运行方式
- 看门狗：`_watchdog_v5.sh`（崩溃自动重启 + 指数退避 15s→300s，10 分钟 3 崩溃则暂停 30 分钟）
- 实际启动：`_run_v5_build.sh`
- 脚本自带 **auto-resume**：扫描 `confidence_train.lmdb` 已有 key，自动从下一个继续

---

## 五、断电应对（每天 02:30 断电）

### 数据安全
- LMDB 默认 `sync=True`，每个蛋白完成后原子提交落盘
- 断电最多丢失正在跑的 1 个蛋白，已完成条目绝对安全

### 开机自启动
1. Windows 启动目录中的 `DisOrderFlow_V5_Build.lnk`
   → 触发 `auto_resume_v5.bat`
2. bat 等 60 秒 → 检查 WSL 就绪 → 检查构建是否已运行 → 启动看门狗

### 手动恢复（自启动失败时）
```powershell
wsl -d Ubuntu-24.04 bash -c "setsid bash <wsl-project-path>/_watchdog_v5.sh > /dev/null 2>&1 < /dev/null &"
```

---

## 六、监控与查询

### 进度监控（随时可跑）
```powershell
python monitor_v5_progress.py
```
或
```bash
wsl -d Ubuntu-24.04 bash -lc "cd <wsl-project-path> && source venv_wsl/bin/activate && python monitor_v5_progress.py"
```

### 关键文件
| 文件 | 作用 |
|---|---|
| `data/confidence_conformation_v5/confidence_train.lmdb` | **产物**（每个 key 是一个蛋白的多构象数据） |
| `_build_v5.log` | 构建日志（含 `[N/1190]` 进度行） |
| `_watchdog_v5.sh` | 看门狗 |
| `_run_v5_build.sh` | 构建启动包装 |
| `auto_resume_v5.bat` | 开机自启动 |
| `monitor_v5_progress.py` | 进度监控 |
| `_check_v5_progress.py` | 简单计数（查看 __len__） |

### 单条目 schema（v5 LMDB 内每条记录）
```python
{
  'pdb_id': str,
  'sequence': str,
  'n_conformations': 5,
  'ca_positions': [np.float32 (N_res,3) × 5],   # 5 套 Cα
  'rmsf': np.float32 (N_res,),                   # 连续 RMSF (Å)
  'disorder_label': np.float32 (N_res,),         # tanh(RMSF/2), 训练目标
  'batch': {原始 BFN batch 字典, 含 torch tensor},
}
```

---

## 七、常见问题排查

### Q: 构建进程不在运行？
```bash
wsl -d Ubuntu-24.04 bash -lc "pgrep -fa build_conformation | head -1"
```
若空，手动启动看门狗（见第五节）。

### Q: 报 `ModuleNotFoundError: No module named 'torch'`？
`venv_wsl` 里 torch 损坏/丢失。重装：
```bash
wsl -d Ubuntu-24.04 bash -lc "cd /mnt/c/biological/DisorderFlow && source venv_wsl/bin/activate && pip install -i https://pypi.org/simple --no-deps torch==2.5.1 && pip install -i https://pypi.org/simple nvidia-curand-cu12"
```

### Q: 报 `NameError: name 'num_residues' is not defined`？
alphafold patch 丢失（重装后），重新打 patch（见第三节）。

### Q: 报 jax 找不到 GPU / `CpuDevice(id=0)`？
检查 `nvidia-smi`（WSL 内）+ jax cuda plugin：
```bash
wsl -d Ubuntu-24.04 bash -lc "nvidia-smi | head -3; cd /mnt/c/biological/DisorderFlow && source venv_wsl/bin/activate && python -c 'import jax; print(jax.devices())'"
```

### Q: 长序列（L>300）崩溃？
正常，看门狗会自动重启续跑。JAX XLA 在长序列易 OOM，已通过 `--num_recycle 0` + 内存卫生缓解。

### Q: 怎么判断数据集构建完成？
`monitor_v5_progress.py` 显示 `进度: 1301/1301 (100.0%)` 且进程退出。

---

## 八、构建完成后

1. 运行校验：
   ```bash
   wsl -d Ubuntu-24.04 bash -lc "cd <wsl-project-path> && source venv_wsl/bin/activate && python _inspect_conformation.py"
   ```
2. 更新训练配置指向 v5：
   `configs/train/bfn_conformation_cuda.yml` 的 `db_path`
3. 删除启动文件夹快捷方式（避免开机再跑）：
   Windows 启动目录中的 `DisOrderFlow_V5_Build.lnk`

### 已配置的自动后处理

`_post_v5_compute.sh` 会等待 v5 主构建退出，然后：

1. 验证条目数必须为 1301。
2. 用 MMseqs2 以 30% identity / 80% coverage 聚类。
3. 生成 `data/confidence_conformation_v5_clustered/` 的 train/val LMDB。
4. 运行跨拆分同源泄漏审计并保存 `split_audit.json`。
5. 仅当 `DISORDERFLOW_FINETUNE` 指向有效 BFN checkpoint 时启动 CUDA 训练；
   缺少 checkpoint 时停止在可审计的数据拆分阶段，不从随机冻结骨干训练。

当前自动后处理日志：`_post_v5_compute.log`。

---

## 九、关键路径速查

| 项 | 值 |
|---|---|
| 项目根 | 当前仓库根目录 |
| WSL 路径 | 由 `wslpath` 从项目根转换 |
| WSL 发行版 | `Ubuntu-24.04` |
| venv | `venv_wsl`（Python 3.12） |
| Windows Python（仅监控用） | 安装了 `lmdb` 的 Python |
| GPU | 支持当前 JAX/CUDA 组合的 NVIDIA GPU |
| 源 LMDB | `data/confidence_unified_v2` (1301 条目, 20GB) |
| 产物 LMDB | `data/confidence_conformation_v5/confidence_train.lmdb` |
| AF2 模型 | `alphafold2_multimer_v3 model_1`（固定，勿换） |
