# 前端重做 — 实现说明（v3，自建前端）

状态：可运行。入口 `app_web.py`，界面在 `webapp/web/static/`。
日期：2026-11-11

## 为什么换掉 Gradio

Gradio 的表单控件（滑块、复选框、下拉、上传框、表格）决定了界面的形态上限。
上一版把 Gradio 涂成深色，控件本身仍是 Gradio 的样子，所以观感停留在"带皮肤的
表单页"。要做出仪器的密度和质感，必须自己控制 DOM。

现在：FastAPI 提供 JSON API，前端是**无框架**的 HTML/CSS/JS。后端业务函数一行没改，
只换表现层。

## 怎么跑

```bash
python app_web.py                  # 默认读 app_config.yaml 的 host/port
python app_web.py --port 7862      # 换个端口
python manage.py start --v2        # 用原来的管理命令
```

跨机部署必须设置鉴权，否则拒绝启动：

```bash
set DF_USER=alice
set DF_PASS=change-me
python app_web.py --host 0.0.0.0
```

测试：

```bash
python -m pytest tests/test_web_api.py -q     # 13 项，约 10 秒
```

## 结构

```
app_web.py                    入口：启动参数、鉴权检查、uvicorn
webapp/
  ops.py                      操作层：8 个科研任务，统一 OpResult
  runs.py                     run manifest（哈希 / 检查点 / 耗时）
  web/
    server.py                 FastAPI：上传、任务、SSE、记录、配置
    static/
      index.html              四个工作区的静态结构
      styles.css              仪器风格样式（无卡片、细线分区）
      app.js                  状态、上传、SSE、表格、快捷键
tests/test_web_api.py         接口契约测试
```

旧 Gradio 界面已删除：`app.py`、`webapp/ui.py`、`webapp/facade.py`。
三个原先 `from app import ...` 的研究模块已改指向真实模块
（`webapp.sequtils` / `webapp.design` / `webapp.state`）。

## 界面结构

```
┌ 顶栏：品牌 / 设计·验证·靶点·记录 / 设备状态 / 设置 ────────────────┐
├ 侧栏 208px ─┬ 主区 ──────────────────────────────────────────────┤
│ 模式        │  标题 + 一句话                                  │
│ · 单结构 1  │  ┌ 输入轨 ────────┐ ┌ 结果画布 ─────────────┐   │
│ · 批量   2  │  │ 细线分区       │ │ 空态：网格 + 运行前检查 │   │
│ · 靶点约束3 │  │ 32px 行高      │ │ 有数据：表格 + 日志     │   │
│ · 疾病靶点4 │  │ 等宽数值       │ │                        │   │
│ 最近运行    │  └────────────────┘ └────────────────────────┘   │
├─────────────┴──────────────────────────────────────────────────┤
│ 底部条：当前操作 · 参数摘要 · 进度 · 耗时 · 运行/取消            │
└─────────────────────────────────────────────────────────────────┘
```

设计选择：

- **无卡片**。用 1px 细线和小节标签（10px 大写 + 字距）分区，减少视觉噪音。
- **紧凑**。行高 32px，控件 28px，标签 12px，数字用等宽字体并右对齐。
- **单一强调色** `#2f6bdb`。语义色只用于状态点。
- **空闲画布**用 34px 网格铺底，左上角是"运行前检查"清单（文件/区域/工具是否就绪），
  让空态也能传达信息。
- **结果表的 Quality 列**渲染成彩色状态点，而不是文字。
- 键盘：`Ctrl/⌘ + Enter` 运行，`1–4` 切模式，`Esc` 取消。

## 后端接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/bootstrap` | 状态 + 知识库 + 最近记录，一次拉全 |
| POST | `/api/upload?name=` | 流式写入，返回不透明 `id` |
| POST | `/api/template` | FASTA → α-螺旋模板 PDB（容忍裸序列） |
| POST | `/api/epitope` | 表位打分 |
| POST | `/api/disease/load` | 加载疾病构象，返回 `upload_id` |
| POST | `/api/run` | `{op, params}`，返回 `job_id` |
| GET | `/api/jobs/{id}/events` | SSE：阶段、百分比、结果 |
| POST | `/api/jobs/{id}/cancel` | 协作式取消（下一次进度回调时生效） |
| GET | `/api/runs` · `/api/runs/{id}` | 记录与 manifest |
| GET | `/api/artifact/{run}/{idx}` | 产物下载，只允许 runs/ 下的文件 |
| GET/PUT | `/api/config` | 配置读取/保存（校验 + 备份 + 原子写） |

八个操作：`design_single` / `design_batch` / `design_target` / `design_disease`
/ `validate_af2` / `validate_confidence` / `validate_iterative` / `validate_loop`。

## 安全模型（相对上一版的变化）

| 项 | Gradio 版 | 自建版 |
|---|---|---|
| 客户端可达面 | 全部 handler 可经 `/call/*` 直连 | 只有 11 个显式路由 |
| 文件路径 | `gr.State` 传原始路径 | 客户端只拿到不透明 token，路径不出服务端 |
| CSRF | 依赖 CORS | 写操作要求 `X-DF: 1`，跨站无法自带 |
| 错误回显 | `show_error=True` 回显栈 | 统一 JSON `detail`，栈只进日志 |
| 绑定地址 | 可配置，无检查 | 非回环且无鉴权 → 拒绝启动 |
| 检查点 | 任意路径 `torch.load` | `logs/` 白名单 + 优先 `weights_only=True` |
| 上传体积 | 无上限 | 200MB，流式写入，超限即删 |

## 实测

- `pytest tests/test_web_api.py` → 13 项通过（上传/模板/表位/记录/配置/路径穿越）
- 浏览器端到端：FASTA → 模板 → 运行 BFN → SSE → 3 条候选 → 产物 → 记录页
  （约 6 秒，模型已加载时）
- 页面无 console error，无 404

## 已知缺口

1. **检查点格式**：内嵌 `easydict.EasyDict`，`weights_only=True` 读不了，代码回退到
   `weights_only=False`（仅限 `logs/` 白名单）。彻底方案是拆旁挂 JSON。
2. **疾病靶点的 AF2 多聚体验证**这一版没有接进界面（上一版有）。需要把上一轮设计结果
   的完整序列带到验证步骤，属于额外状态管理。`ops` 里还没有对应函数。
3. **取消是协作式的**：只在进度回调处生效，单次 BFN 采样中途不会中断。
4. ~~旧 `webapp/ui.py` 未删除~~：已于 2026-11-11 删除（`app.py`、`webapp/ui.py`、
   `webapp/facade.py`），后端同时去掉了 Gradio 依赖。
5. 窄屏只做了单栏降级，未做触控优化。

---

## 生物学表达层（2026-11-11 追加）

界面不再只是"表单 + 表格"，加了四层领域信息：

| 元素 | 数据来源 | 作用 |
|---|---|---|
| **主链 CA 投影** | `/api/structure` 抽取 Cα 坐标，前端按方差最大的两轴投影 | 上传 PDB 后立刻看到折叠形态；设计区域用加粗亮段标出 |
| **残基着色** | 序列列按理化性质分组（疏水/极性/正电/负电/特殊） | 候选序列一眼看出组成偏好 |
| **置信度统计条** | 结果表里的 pLDDT / ipTM / PPL / 恢复率 | 按 AF2 分档配色（≥0.8 绿、≥0.6 黄、其余红） |
| **逐残基色带** | Position + pLDDT 两列 | 置信度评估的残基级剖面，比表格直观 |

实现放在 `webapp/web/static/bio.js`，不持有应用状态，数据由 `app.js` 传入。

## 一次运行失败的定位（2026-11-11）

现象：疾病靶点 + ESM-IF 运行失败，`ValueError: torch_scatter.__spec__ is None`。

原因：`esmif_compat.install_torch_scatter_fallback()` 在 `run_esmif` 与
`load_esmif` 里各调用一次。第一次把兼容模块塞进 `sys.modules`，第二次
`importlib.util.find_spec` 看到该模块 `__spec__` 为 None（早期版本没设），
直接抛 ValueError。

处理：
1. 15:17 的补丁给假模块补了 `__spec__`；
2. 本次进一步做成真正幂等——先看 `sys.modules` 里是否已有可用实现，再对
   `find_spec` 的 `ValueError` 兜底，并能覆盖别处塞进来的残缺 stub；
3. `tests/test_esmif_compat.py` 增加两条回归测试（重复调用、外部 stub）。
