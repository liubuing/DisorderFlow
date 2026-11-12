# 前端安全审计 — DisorderFlow Web UI

审计对象：`app.py` + `webapp/`（Gradio 6.17.3，Python 3.14.5）
审计日期：2026-11-10
审计方式：源码走查 + 运行时验证（临时实例 127.0.0.1:7861，未对生产实例做写操作）

---

## 0. 威胁模型修正

现有设计假设是"仅本机访问 = 安全边界"。这个假设不成立，原因有三：

1. **所有 44 个回调都是公开 HTTP 端点。** `/config` 中 `dependencies` 共 44 条，全部带 `api_name` 且 `backend_fn=True`。用 `curl` 直打 `POST /gradio_api/call/<handler>` 即可执行任意 handler，无需页面、无需会话、无令牌。已实测 `on_fasta_input` 返回正常业务结果。
2. **配置本身可被这些端点改写**（`do_save_config`），包括 `server.host`。一次调用即可把监听地址改成 `0.0.0.0`，下次启动即对内网开放。
3. **本机不等于可信。** 浏览器里的任意页面、本地其他进程、IDE 插件、端口转发工具（ngrok/frp/VS Code tunnel）都能碰到这个端口。

因此下面按"**未认证远程攻击者**"定级，而不是"本机用户"。

---

## S0 — 严重

### S0-1 未认证 RPC 面：全部 44 个 handler 可直连

**位置**：`webapp/ui.py` 全部 `.click()` / `.change()` / `.upload()` 注册处；`app.py:63-73`

**证据**

```
$ python -c "... config['dependencies']"
dependencies: 44   with api_name: 44   backend_fn: True

$ curl -X POST http://127.0.0.1:7861/gradio_api/call/on_fasta_input \
       -H 'Content-Type: application/json' -d '{"data":[">probe\nMKTAYIAKQR",null]}'
{"event_id":"bd292dda77834924a032a4dd1d108a49"}

$ curl -N .../on_fasta_input/bd292dda...
event: complete
data: [">probe\nMKTAYIAKQR", "probe: 10 aa\n  MKTAYIAKQR"]
```

`launch()` 未传 `auth`（默认 `None`），未设 `allowed_paths` / `blocked_paths` / `max_file_size`。

**影响**：`do_save_config`（写 `app_config.yaml`）、`do_reload_bfn`（重新反序列化检查点）、`run_alphafold_prediction`（起子进程）、`mf_run_af2`（见 S0-3）全部对任何能连上端口的人开放。

**修复**（按性价比排序）

> **实测更正（2026-11-10）**：`api_visibility="private"` **不能阻断原始 HTTP 调用**。
> 在 Gradio 6.17.3 上，`POST /gradio_api/call/<name>` 对 private 事件仍会返回
> `event_id` 并真正执行 handler；`api_name=False` 只是映射成 private。它只隐藏
> 文档与 gradio_client。**唯一有效的边界是绑定地址与鉴权**，下面的 auth 才是关键项。

```python
# 1) 内部 handler 退出文档 / 客户端库（降低便利性，但非安全边界）
btn.click(fn, inputs, outputs, api_visibility="private")

# 2) 真正的修复：回环绑定 + 鉴权（app_v2.py 已实现强制检查）
app.launch(
    auth=(os.environ["DF_USER"], os.environ["DF_PASS"]),   # 或 auth_dependency
    allowed_paths=[str(UPLOAD_DIR)],
    blocked_paths=[str(PROJECT_DIR / ".git"), str(Path.home() / ".ssh")],
    max_file_size="200mb",
    show_error=False,
)

# 3) Blocks 层加临时文件回收
gr.Blocks(title=..., delete_cache=(3600, 3600))
```

`app_v2.py` 的实现：监听非回环地址且未设置 `DF_USER`/`DF_PASS` 时直接拒绝启动（fail closed）；
回环地址下不弹窗（单人本机工具），但文档明确「本机进程视为可信」这一前提。

---

### S0-2 配置可写 → 检查点路径可换 → `torch.load(weights_only=False)` 反序列化 RCE

**位置**：`webapp/state.py:64-90`；`webapp/ui.py:526-533`；同型代码 `manage.py:343,405`、`modules/bfn_loader.py:85`、`modules/design_seq.py:136`

**链路**（每一环都已确认无鉴权）

1. `POST /gradio_api/upload` 上传一个 `.pdb` 文件 → 得到服务器临时路径；
   `gr.File` 只校验扩展名，不校验内容；`torch.load` 不看扩展名。
2. `POST /call/do_save_config`，body 里把 `models.bfn.checkpoint` 指向该路径。
3. `POST /call/do_reload_bfn` → `load_bfn()` → `torch.load(ckpt_path, weights_only=False)` → pickle 反序列化执行。

```python
# webapp/state.py:70
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
```

**修复**

```python
# 1) 只加载张量
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
# 训练配置改为旁挂 JSON/YAML（checkpoint 同目录 <name>.config.json），
# 不再把 Config 对象塞进 pickle。

# 2) 路径白名单
ALLOWED_CKPT_ROOT = (PROJECT_DIR / "logs").resolve()
p = Path(ckpt_path).resolve()
if not p.is_relative_to(ALLOWED_CKPT_ROOT):
    raise ValueError(f"checkpoint outside allowed root: {p}")

# 3) 上传文件做魔数/内容校验（PDB 必须以 ATOM/HETATM/HEADER 开头）
```

若短期内无法改 `weights_only=True`（旧检查点含 Config 对象），至少要做到：
检查点必须落在白名单目录、配置里的 checkpoint 字段不可由 API 修改、且对文件做 SHA-256 记录。

---

### S0-3 `shell=True` 拼接客户端可控字符串

**位置**：`modules/misfolding_pipeline.py:242-256`（`validate_with_af2`）

```python
fasta_path = os.path.join(output_dir, f'{ab_name}_{ag_name}.fasta')
cmd = (
    f'"{colabfold_exe}" --num-models 1 --num-recycle {num_recycle} '
    f'--model-type {model_type} --rank iptm '
    f'"{fasta_path}" "{output_dir}"'
)
result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600, cwd=output_dir)
```

**污染源**：`webapp/ui.py:1428-1434`

```python
output_dir = os.path.join(str(PROJECT_DIR), "misfolding_results", str(disease_key))
result = pipeline.validate_with_af2(seq, ag_seq, output_dir,
                                    ab_name=f'design_{i+1}', ag_name=str(disease_key))
```

`disease_key` 来自 `gr.State`（`_mf_disease_key`）。`gr.State` 在 API 层是**完全透明的普通输入**，客户端可传任意字符串，服务端不做字典校验——`mkb.get_target()` 只在下拉回调里用过，这里没有。

**可利用性**

- **POSIX / WSL 后端**：双引号内的 `$(...)` 与反引号仍会执行，`$()` 是干净的一击 RCE；`os.makedirs` 允许 `"`。
- **Windows 主机**：`"` 是非法文件名字符，`os.makedirs(output_dir)` 会先抛 `OSError`，直接注入被挡住。但同一函数里的 `colabfold_exe` 也来自配置（`cfg['alphafold']['af2']['executable']`），而配置可由 S0-1 的 `do_save_config` 改写：
  `executable = 'colabfold_batch" & calc & "'` → 拼出的命令变成 `"colabfold_batch" & calc & "" --num-models ...` → 执行 `calc`。
  所以 Windows 下是一条"S0-1 → S0-3"的组合链。

**修复**

```python
# 用列表参数，彻底去掉 shell
cmd = [colabfold_exe, "--num-models", "1", "--num-recycle", str(int(num_recycle)),
       "--model-type", model_type, "--rank", "iptm", fasta_path, output_dir]
subprocess.run(cmd, shell=False, ...)

# 并且：名字只允许 [A-Za-z0-9_-]
safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(ag_name))[:64] or "target"
# 且 output_dir 必须 resolve 到 PROJECT_DIR 之下
```

更好的做法是统一走 `runtime_environment.build_colabfold_command()`——`webapp/af2.py:84` 已经在用它，只有 `misfolding_pipeline` 这一处绕过了。

---

## S1 — 高

### S1-1 任意路径读写（路径穿越 + 未校验 State）

**位置**：`webapp/ui.py` 中所有 `gr.State` 承载的路径
`_pdb_path`(271)、`_t5_target_pdb`(560)、`_t5_antibody_pdb`(564)、`_conf_pdb_path`(1051)、`_pipe_pdb_path`(1082)、`_mf_target_pdb`(1130)、`_mf_ab_pdb`(1137)、`_cl_pdb`(1590)

这些 State 只被 `os.path.exists()` 或直接丢给 Biopython / 子进程，没有任何"必须位于上传目录"的约束。API 可直接传 `C:\Users\<u>\.ssh\id_rsa`。

配合 S0-3，`disease_key = r"..\..\..\..\Users\<u>\Desktop\pwn"` 还能把 `misfolding_results/` 之外的目录当作 `output_dir`（Windows 上不含 `"` 时 `makedirs` 会成功），实现任意位置写 FASTA。

**修复**

```python
def resolve_upload(p: str) -> Path:
    path = Path(p).resolve()
    if not path.is_relative_to(UPLOAD_ROOT.resolve()):
        raise ValueError("path escapes upload root")
    return path
```

或者更彻底：**不再用 `gr.State` 传路径**，改用 `gr.File` 类型的输入（Gradio 会做 allowed_paths 校验），或在服务端用一次性 token 映射到真实路径。

---

### S1-2 `show_error=True` + traceback 回显

**位置**：`app.py:72`；`webapp/ui.py:765,1032,1571` 等 `traceback.format_exc()` 直接写进 Textbox。

泄露绝对路径、用户名、Python/库版本、内部函数名；Gradio 的错误页还会回显输入。对研究工具是纯粹的负收益。

**修复**：`show_error=False`；handler 内 `except Exception` 只返回人话 + 错误码，完整栈写 `logs/ui_errors.log`。

---

### S1-3 无临时文件回收、无上传体积上限

**位置**：`app.py:63-73`（`delete_cache` 默认 `None`，`max_file_size` 默认 `None`）

每次上传的 PDB/FASTA 永久留在 Gradio 临时目录；`_save_fasta()` 每次运行新建一个 `NamedTemporaryFile(delete=False)`（`webapp/sequtils.py:273-281`），也从不清理。长期跑会吃满磁盘。

**修复**：`gr.Blocks(delete_cache=(3600, 3600))`、`launch(max_file_size="200mb")`、`_save_fasta` 改为写入 `runs/<run_id>/` 并随 run 生命周期回收。

---

### S1-4 全局单例的并发竞争

**位置**：`webapp/state.py:41-43`（`_bfn_model` / `_bfn_config` / `_esmif_model` / `_app_config`）、`webapp/af2.py:29`（`_predicted_pdb`）

- `do_reload_bfn`（`ui.py:543`）把 `_bfn_model = None` 时，正在采样的回调仍持有旧引用；另一并发回调会重新 `load_bfn()`，`load_bfn` 没有锁，两个线程可能同时加载 → VRAM 翻倍 → OOM。
- `_predicted_pdb` 是进程级全局，两个客户端并发预测会互相覆盖，A 拿到 B 的结构。
- 事件之间没有全局并发限制，`run_design` / `cl_run` / `run_alphafold_prediction` 可以同时起 GPU 任务。

**修复**

```python
_bfn_lock = threading.Lock()
with _bfn_lock:
    if _bfn_model is None: ...
```

`_predicted_pdb` 改为 `gr.State` 或 per-session 字典；`app.queue(default_concurrency_limit=1)`，并对重任务单独设 `concurrency_limit=1`。

---

## S2 — 中

| 编号 | 问题 | 位置 | 修复 |
|---|---|---|---|
| S2-1 | 依赖不锁版本：`gradio>=4.0`，实际跑 6.17.3；`requirements.txt` 全部无 pin | `requirements.txt:14`、`pyproject.toml:52-55` | pin `gradio==6.17.3`；产出 `requirements.lock`；CI 加 `create_ui()` 冒烟测试 |
| S2-2 | 外部字体 `@import` Google Fonts，离线时静默失败并对外发请求（实测 `Inter: unloaded`） | `webapp/ui.py:42` | 删掉，改系统 CJK 字体栈；要品牌字体就本地 self-host |
| S2-3 | 无 CSP / 安全响应头 | `app.py:63` | 反代加头，或 `head=` 注入 meta CSP（`default-src 'self'; connect-src 'self' ws:; style-src 'self' 'unsafe-inline'`） |
| S2-4 | 无运行溯源：设计/预测不记录输入哈希、检查点哈希、配置快照 | 全局 | 见 `redesign-plan.md` 的 run manifest |
| S2-5 | 配置文件编辑器无 schema 校验、无备份、无 diff；存一次坏 YAML 下次启动即挂 | `webapp/ui.py:515-533`、`webapp/state.py:150-160` | pydantic schema + 原子写（tmp→rename）+ `.bak` |

---

## 已排除 / 不成立的点

- `run_alphafold_prediction` 的文件名来自 FASTA header，但已用 `re.sub(r'[^A-Za-z0-9_-]', '_', pid)` 净化（`webapp/af2.py:57-59`）——安全。
- `run_batch` 写盘用的 `Path(pf.name).stem` 不可能含路径分隔符——不构成穿越。
- `run_mpnn` / `run_alphafold_prediction` 用列表传参、`shell=False`——安全，可作为 S0-3 的修复范式。
- `strict_cors` 在 Gradio 6 默认 `True`，纯浏览器侧的跨站 JSON POST 被挡住。但这道防线**不覆盖** curl、本地进程、端口转发和 DNS rebinding，不能替代鉴权。

---

## 修复优先级

| 顺序 | 动作 | 工作量 | 直接消除 |
|---|---|---|---|
| 1 | 全部 handler `api_visibility="private"` | 0.5h | S0-1 主要面 |
| 2 | `auth=` + `allowed_paths` + `max_file_size` + `delete_cache` + `show_error=False` | 0.5h | S0-1、S1-3 |
| 3 | `weights_only=True` + 检查点白名单 | 2h | S0-2 |
| 4 | `validate_with_af2` 改列表参数 + 名字净化 + 输出目录约束 | 1h | S0-3 |
| 5 | 统一 `resolve_upload()` 守卫；State 路径全部过一遍 | 3h | S1-1 |
| 6 | 模型加载加锁、预测结果 per-session | 2h | S1-4 |
| 7 | pin 依赖 + 冒烟测试 | 2h | S2-1 |

前 4 项约半天，可以把"未认证 → RCE"整条链断掉，且不改动任何业务逻辑。
