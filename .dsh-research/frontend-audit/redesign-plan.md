# 重做方案 — 架构、信息架构、设计系统

前置结论：**不要现在换 React。** 理由和边界见 §1。
目标不是"更好看的 Gradio"，而是**一个能被别人用的研究工具**。

---

## 1. 技术选型：为什么先不推倒到 React

如果把 `webapp/` 全部换成 React + FastAPI，需要先偿还这些债：

| 现状 | React 化需要补的东西 |
|---|---|
| 44 个 handler 是**闭包**，直接读写组件对象（`gr.update`、`gr.State`） | 全部改写成无状态 REST/WS 契约 |
| 进度靠 `gr.Progress()` 注入 | 自己实现 SSE/WebSocket 任务流 |
| 文件进出靠 Gradio 的临时目录与 `/file=` | 自己实现上传、鉴权、生命周期 |
| 结果渲染靠 `gr.DataFrame` / Textbox | 重写表格、虚拟滚动、导出 |
| 10 个页签的交互状态互相关联 | 引入全局状态管理 |

这是 3–6 周的重写，且**回归风险集中在业务正确性**（BFN 采样、级联阈值、AF2 解析），而不是像素。对一个还在产出论文数据的研究平台，这个交换比不划算。

而视觉问题里，60% 是 CSS 选择器失效（见 `design-audit.md` §1），30% 是布局与 IA，10% 是配色。**前两项都能在 Gradio 里解决，且成本低一个数量级。**

**建议路线**

```
现在 ──► Phase 0 安全热修（半天）
      ──► Phase 1-3 Gradio 重做（1.5–2 周）  ← 80% 的收益
      ──► Phase 4 React（仅当出现多用户/公网/协作需求）
```

Phase 1-3 的产出（信息架构、设计令牌、组件契约、run manifest）**在 React 阶段可以 100% 复用**，不会白做。所以"先 Gradio 后 React"不是妥协，是分期。

**什么时候该直接上 React**：需要账号体系、多人共享结果、公网访问、移动端、或要做成对外产品。如果这是 iBEC 演示或论文附图的门面，Gradio 重做足够。

---

## 2. 信息架构：从 10 个页签到 4 个工作区

### 原则

按**用户任务**切分，不按**实现方式**切分。每个工作区内部用"模式"（segmented control）区分变体，模式切换不改变页面骨架。

### 新结构

```
设计 Design      ← 单结构 / 批量 / 靶点约束 / 疾病靶点   （4 个模式）
验证 Validate    ← AF2 预测 / 置信度评估 / 迭代精修        （3 个模式）
靶点 Targets     ← 疾病库 · 表位分析 · 纳米抗体支架 · 多聚体验证
记录 Runs        ← 运行队列 · 历史 · 溯源清单 · 导出
─────────────────────────────────────────────
设置 Settings    ← 系统状态 / 模型 / 配置        （右上角齿轮，模态，不占导航）
```

映射关系（旧 → 新）

| 旧页签 | 新位置 |
|---|---|
| 单步设计 | 设计 · 单结构 |
| 批量处理 | 设计 · 批量 |
| 靶点设计 | 设计 · 靶点约束 |
| 错误折叠疾病 | 设计 · 疾病靶点 + 靶点工作区（表位/支架） |
| 统一工作流 | 设计 · 单结构（"运行后自动验证"开关） |
| 结构预测 | 验证 · AF2 预测 |
| 置信度评估 | 验证 · 置信度 |
| 迭代精修 | 验证 · 迭代 |
| 闭环优化 | 验证 · 闭环 |
| 设置 | 设置（模态） |
| — | 记录（新增） |

10 → 4，且每个工作区首屏只有一个主按钮。

### 页面骨架（所有工作区一致）

```
┌────────────────────────────────────────────────────────────────────┐
│ DisorderFlow   蛋白质设计与验证平台        ● 服务正常   [设置] [?] │
├────────────────────────────────────────────────────────────────────┤
│ 设计   验证   靶点   记录                                          │
├────────────────────────────────────────────────────────────────────┤
│ 设计 · 单结构                          ┌──────────────────────────┐│
│ ┌────────────────────┐                │ 序列  指标  日志          ││
│ │ 01 输入             │                │ ┌──────────────────────┐ ││
│ │  [拖拽 PDB / 粘贴]  │                │ │ #1 MKTAY…  PPL 3.21  │ ││
│ │  链 A · 128 残基    │                │ │ #2 …                 │ ││
│ ├────────────────────┤                │ └──────────────────────┘ ││
│ │ 02 设计区域         │                │  [复制 FASTA] [导出 CSV] ││
│ │  A:10-25  [可视化]  │                └──────────────────────────┘│
│ ├────────────────────┤                                            │
│ │ 03 工具             │                                            │
│ │  [BFN][MPNN][ESM-IF]│                                            │
│ │  参数（随工具变化）  │                                            │
│ └────────────────────┘                                            │
├────────────────────────────────────────────────────────────────────┤
│ BFN 通用 · 3 序列 · A:10-25        ≈ 40 秒      [ ▶ 开始设计 ]     │ ← sticky
└────────────────────────────────────────────────────────────────────┘
```

关键决定：

1. **左输入 / 右结果**，两栏 `scale=3:4`，高度由内容决定，不再出现 `unequal-height` 的大空洞。
2. **主按钮固定在底部操作条**，始终可见，且条上带"参数摘要 + 预计耗时"——运行前就能确认自己要跑什么。
3. **结果区位置永不变**：序列 / 指标 / 日志三个分段视图，避免"结果在这里、验证在那里"。
4. **步骤编号 01/02/03** 是在告诉用户"这是一个有顺序的流程"，而不是一堆并列表单。

---

## 3. 设计系统

### 3.1 令牌（tokens）

```css
:root {
  /* 表面：3 层足够，不要 5 层 */
  --df-bg:        #0b0d12;
  --df-surface:   #12151c;
  --df-raised:    #191d26;
  --df-border:    #242936;
  --df-border-strong: #323a4b;

  /* 文本：全部通过 AA */
  --df-text:      #e6ebf2;   /* 16.2:1 on bg */
  --df-text-2:    #a9b4c4;   /*  9.3:1 on bg */
  --df-muted:     #7d8896;   /*  5.1:1 on surface */
  --df-disabled:  #4a5361;   /* 仅供禁用态，不承载信息 */

  /* 强调：白字 5.37:1 */
  --df-accent:      #5b5bd6;
  --df-accent-hover:#6e6ee0;
  --df-accent-soft: rgba(91,91,214,.14);

  /* 语义色 */
  --df-ok:    #34d399;
  --df-warn:  #fbbf24;
  --df-error: #f87171;

  /* 间距 4 的倍数 */
  --df-1:4px; --df-2:8px; --df-3:12px; --df-4:16px;
  --df-5:24px; --df-6:32px; --df-7:48px;

  /* 圆角 */
  --df-r-sm:8px; --df-r-md:12px; --df-r-lg:16px;

  /* 字体 */
  --df-font: system-ui, "Segoe UI", "PingFang SC", "Microsoft YaHei UI",
             "Noto Sans SC", sans-serif;
  --df-mono: "JetBrains Mono", "Cascadia Mono", Consolas, monospace;
}
```

要点：

- **删掉 Google Fonts `@import`**（实测未加载，且本地工具不该外连）。中文用系统栈，Latin 要品牌感就用 `gr.LocalFont` 自托管。
- 强调色从 `#7b7bf6` 调到 `#5b5bd6`：白字从 3.5:1 → 5.37:1。
- `--muted` 从 `#687380` 调到 `#7d8896`：3.79:1 → 5.08:1。
- 语义色只用于**状态**，不用于装饰；主要动作只有一个强调色。

### 3.2 字号与层级

| 角色 | 字号/行高 | 字重 |
|---|---|---|
| 页面标题 | 20 / 28 | 600 |
| 卡片标题 | 14 / 20 | 600 |
| 正文/标签 | 13 / 20 | 400–500 |
| 辅助说明 | 12 / 18 | 400，`--df-muted` |
| 数据/序列 | 13 / 22，mono | 400 |

中文正文行高不低于 1.6；不要再用 `0.71em`（≈11px）承载徽章文字。

### 3.3 组件清单（10 个，全部自己拥有 `elem_classes`）

| 组件 | 实现 | 说明 |
|---|---|---|
| `df-card` | `gr.Group(elem_classes="df-card")` | 唯一卡片容器，内边距 16/20，圆角 16 |
| `df-step` | `gr.Markdown` + 序号 | 卡片头：`01 输入` + 一句说明 |
| `df-field` | `gr.Column(elem_classes="df-field")` | 标签 + 控件 + 帮助文本三件套 |
| `df-seg` | `gr.Radio(elem_classes="df-seg")` | 工具切换 / 模式切换，横向分段 |
| `df-run` | `gr.Button(elem_classes="df-run", variant="primary")` | 主按钮，带状态 |
| `df-actionbar` | `gr.Row(elem_classes="df-actionbar")` | 吸底操作条 |
| `df-result` | `gr.Column(elem_classes="df-result")` | 右栏容器 |
| `df-table` | `gr.Dataframe(elem_classes="df-table")` | 粘性表头、等宽序列列 |
| `df-seq` | `gr.Textbox(elem_classes="df-mono")` | 只读序列/日志，mono |
| `df-status` | `gr.Markdown(elem_classes="df-status")` | `● 运行中 00:42` / `✓ 完成` / `✗ 失败` |

### 3.4 硬性实现规则

1. **只样式化你自己声明的 `elem_classes` / `elem_id`。**
   禁止 `.gr-box`、`.gr-form`、`.tab-nav`、`.svelte-*`、`.accordion`。
2. **全局令牌走 `gr.themes.Base().set(...)`**，自定义 CSS 只负责布局。
3. `!important` 总量控制在一张白名单里（建议 ≤10 处），每处写明为什么。
4. 禁止跨 `gr.HTML` 拼接标签（`.td-wrapper` 的教训）。
5. 组件一创建就绑定 handler；未绑定的按钮不许合并。加一条 lint 规则见 §5。
6. 所有数值参数用 `Slider` / `Number(precision=...)`，不用自由文本 `Textbox`。

### 3.5 主题骨架

```python
import gradio as gr

theme = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#f0f0ff", c100="#dcdcfb", c200="#c2c2f7", c300="#a3a3f2",
        c400="#8484ec", c500="#5b5bd6", c600="#4a4ac0", c700="#3a3aa3",
        c800="#2b2b82", c900="#1d1d5e", c950="#101033",
    ),
    font=["system-ui", "Segoe UI", "PingFang SC", "Microsoft YaHei UI", "sans-serif"],
    font_mono=["JetBrains Mono", "Cascadia Mono", "Consolas", "monospace"],
).set(
    body_background_fill="#0b0d12",
    block_background_fill="#12151c",
    block_border_width="1px",
    block_border_color="#242936",
    block_radius="12px",
    block_label_text_weight="600",
    input_background_fill="#0d1017",
    button_primary_background_fill="#5b5bd6",
    button_primary_background_fill_hover="#6e6ee0",
    button_primary_text_color="#ffffff",
    body_text_color="#e6ebf2",
    body_text_color_subdued="#7d8896",
)

with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="DisorderFlow",
               delete_cache=(3600, 3600)) as app:
    ...
```

---

## 4. 状态模型：把"运行"变成一等公民

这是最能提升"专业感"、同时解决溯源问题的一件事。

每个工作区的一次执行 = 一个 **Run**：

```
runs/
└── 20261110-143205_a1b2c3/
    ├── manifest.json      # 见下
    ├── inputs/            # 上传的 PDB/FASTA 副本
    ├── outputs/           # 设计序列、AF2 结构、指标表
    └── ui.log
```

```json
{
  "run_id": "20261110-143205_a1b2c3",
  "workspace": "design",
  "mode": "single",
  "started_at": "2026-11-10T14:32:05+08:00",
  "duration_s": 41.2,
  "status": "completed",
  "inputs": {
    "pdb_sha256": "9f2c…",
    "region": "A:10-25",
    "n_samples": 3,
    "stochastic": false
  },
  "environment": {
    "checkpoint": "logs/…/best.pt",
    "checkpoint_sha256": "1a7b…",
    "config_sha256": "77e0…",
    "device": "cuda",
    "torch": "2.12.0+cu130"
  },
  "artifacts": ["outputs/design.fasta", "outputs/metrics.csv"]
}
```

收益：

- 记录页天然有内容，可回看、可对比、可复现；
- 论文/评审要的溯源直接落到文件，不再靠截图；
- 失败重试有上下文；
- 临时文件有了归属，`delete_cache` 之外还能按 run 清理。

---

## 5. 验收标准（可自动化）

前端改动必须过这 5 条，否则不许合并：

1. **零死选择器**
   脚本解析自定义 CSS 里所有选择器，在渲染后的页面上逐个 `querySelectorAll` 断言 `> 0`。
   （本次审计就是用它抓出 12 个 0 命中选择器。）
2. **对比度 ≥ 4.5:1**
   遍历所有文本节点的 `color` / `background` 计算对比度，失败即报错。
3. **无死按钮**
   静态扫描 `gr.Button(` 创建与 `.click(` / `.then(` 绑定，数量必须配平。
4. **三档视口无大空洞**
   1280 / 1440 / 1920 下，任意两栏高度差 ≤ 120px，首屏无 > 120px 的空列。
5. **首屏三步可达**
   上传 → 选工具 → 运行，三步都在首屏内完成，不需要滚动。

加一个 `tests/test_ui_smoke.py`：`create_ui()` 能构建、组件数符合预期、关键 `elem_classes` 存在。这样 Gradio 升级时会立刻失败，而不是像现在这样静默丢样式。

---

## 6. 实施路线

### Phase 0 — 安全热修（0.5 天，可立即发布）

对应 `security-findings.md` 优先级 1–4：`api_visibility="private"`、`auth=`、`weights_only=True`、`shell=False`。
不碰业务逻辑，不改视觉。

### Phase 1 — 地基（2–3 天）

- `webapp/theme.py`：tokens + `gr.themes.Base`
- `webapp/components.py`：§3.3 的 10 个封装
- `webapp/shell.py`：Navbar + 4 工作区 + 设置模态 + 吸底操作条
- 迁移**设计工作区**（单结构 / 批量 / 靶点约束 / 疾病靶点 四模式）
- `tests/test_ui_smoke.py` + 选择器/对比度检查脚本

产出：一个能跑的设计工作区，样式与结构是新的；其余工作区暂时保留旧页签（渐进迁移，不做大爆炸）。

### Phase 2 — 迁移与状态（3–5 天）

- 迁移 验证 / 靶点 / 记录 三个工作区
- Run 模型 + `runs/` 落盘 + 记录页
- 统一状态机：`idle → running(elapsed, cancel) → done | failed(msg, retry)`
- 消除本审计列出的全部功能缺陷（`refresh_btn`、`ir_btn_stop`、`.value` 误用、`known=[]`、数值框自由文本）

### Phase 3 — 打磨（2 天）

- 键盘：`Ctrl/Cmd+Enter` 运行、`Esc` 取消、`1–4` 切工作区
- 空态/加载骨架/失败重试
- `prefers-reduced-motion`
- 窄屏（<1100）单栏降级：输入在上、结果在下

### Phase 4 — 可选：React

仅当出现多用户/公网需求。此时把 Phase 1-3 定下的工作区、组件、Run 契约直接映射过去；后端把 44 个 handler 收敛成 ~12 个资源端点。

---

## 7. 不要做的事

| 反模式 | 原因 |
|---|---|
| 在现有 CSS 上继续加 `!important` | 选择器已经和 Gradio 版本绑死，只会更难维护 |
| 引入 Tailwind/Bootstrap/MUI 类库 | 与 Gradio 的 theme 系统打架，且两套设计语言 |
| 现在就换 React | 见 §1，成本高一个数量级，收益集中在像素 |
| 保留 10 个平级页签只换皮 | IA 是主要问题，换皮解决不了"我该点哪一个" |
| 把"设置"留在主导航 | 运维入口不该和业务工作区同级 |
| 用 emoji 当图标系统 | 跨平台渲染不一致，无障碍语义为零；换内联 SVG（Lucide） |

---

## 8. 一份最小可用的新骨架（示意）

```python
# webapp/shell.py
import gradio as gr
from webapp.theme import THEME, CSS
from webapp.components import card, step, field, action_bar

NAV = [("设计", "design"), ("验证", "validate"), ("靶点", "targets"), ("记录", "runs")]

def create_app():
    with gr.Blocks(theme=THEME, css=CSS, title="DisorderFlow",
                   delete_cache=(3600, 3600)) as app:
        gr.HTML('<header class="df-topbar">…</header>')

        with gr.Row(elem_classes="df-nav"):
            nav = gr.Radio([n for n, _ in NAV], value="设计",
                           show_label=False, elem_classes="df-nav-radio")

        pages = {}
        with gr.Column(visible=True) as pages["design"]:
            from webapp.workspaces.design import build_design
            build_design()
        for _, key in NAV[1:]:
            with gr.Column(visible=False) as col:
                pages[key] = col
                gr.Markdown(f"### {key} — 迁移中")

        nav.change(lambda v: [gr.update(visible=(k == v)) for k in pages.values()],
                   nav, list(pages.values()), api_visibility="private")

        with gr.Sidebar():        # 可选：设置抽屉
            from webapp.workspaces.settings import build_settings
            build_settings()
    return app
```

---

## 9. 工作量汇总

| 阶段 | 内容 | 人日 |
|---|---|---:|
| Phase 0 | 安全热修 | 0.5 |
| Phase 1 | 主题 + 组件 + 设计工作区 | 2–3 |
| Phase 2 | 其余工作区 + Run 模型 | 3–5 |
| Phase 3 | 打磨 + 无障碍 | 2 |
| **合计** | | **8–11 人日** |

对比 React 重写（3–6 周），且 Phase 0 可以今天就上线。
