# 视觉与交互审计 — DisorderFlow Web UI

审计方式：本地启动临时实例（127.0.0.1:7861，1440×1000 viewport），截图 + 运行时 DOM/CSSOM 测量。
所有数字均为实测，不是目测。

---

## 结论先行

"没有任何美学设计"这个判断**只对了一半**。

`webapp/ui.py:37-227` 里有 190 行、10.3 KB 的完整深色主题——设计意图是清晰的（分层卡片、紫色强调、等宽结果区、圆角 16px）。问题在于：**这套 CSS 有 60% 没有生效**，因为选择器写的是 Gradio 3/4 的类名，而运行时是 Gradio 6.17.3。你看到的"没有设计"，很大一部分是**设计在传输途中丢了**。

同时，有几件事确实是结构性的、改 CSS 救不回来的：

1. 10 个平级页签，其中 6 个是同一个动作（"设计"）的不同包装；
2. 两列布局高度失衡，首屏右侧有约 400px 的纯空白；
3. 结果区永远在折叠线以下；
4. 没有加载态、空态、失败态，旧结果不会清空；
5. 中文界面用了没有 CJK 字形的 Inter，离线还没加载成功。

所以：**该重做的是信息架构和组件层，不是配色。** 配色改 15 行就能达标。

---

## 1. CSS 命中率实测

在页面里逐个 `document.querySelectorAll()` 统计样式表用到的选择器：

| 选择器 | 命中数 | 说明 |
|---|---:|---|
| `.gradio-container` | 1 | ✅ 生效 |
| `.platform-header` | 1 | ✅ 生效 |
| `.platform-footer` | 1 | ✅ 生效 |
| `.gr-group` | 10 | ✅ Gradio 6 仍用这个类 |
| `.tabs > .tabitem` | 2 | ⚠️ 部分生效 |
| **`.tabs > .tab-nav`** | **0** | ❌ 页签导航整套样式失效 |
| **`.tabs > .tab-nav button`** | **0** | ❌ 同上 |
| **`.gr-box`** | **0** | ❌ |
| **`.gr-form`** | **0** | ❌ |
| **`.gr-panel`** | **0** | ❌ |
| **`.card-group`** | **0** | ❌ |
| **`.accordion`** | **0** | ❌ 折叠面板无样式 |
| **`.gr-button`** | **0** | ❌（`button.primary` 靠原生 class 侥幸命中） |
| **`.gr-checkbox`** | **0** | ❌ |
| **`.gr-dropdown`** | **0** | ❌ |
| **`.gr-slider`** | **0** | ❌ |
| **`.gr-accordion`** | **0** | ❌ |
| **`.td-wrapper`** | **0** | ❌ 见 §2 |

Gradio 6 的真实 DOM 是 `.tab-wrapper` + `[role="tablist"]` + `.tab-container`，以及 Svelte 哈希类（`svelte-11gaq1` 等）。所有挂在 Gradio 内部类名上的 `!important` 覆盖都会随小版本升级再次失效。

**这就是最大的技术债**：`!important` 越多，说明越依赖内部实现。重做时必须改成"只样式化我自己声明的 `elem_classes`"。

---

## 2. 一个典型的结构性 bug：`.td-wrapper` 完全失效

```python
# webapp/ui.py:556
gr.HTML('<div class="td-wrapper">')
...
# webapp/ui.py:903
gr.HTML('</div>')  # close .td-wrapper
```

两个 `gr.HTML` 各自渲染成独立的 DOM 容器，浏览器会把第一个 `<div>` 自动闭合。实测 `.td-wrapper` 命中数为 **0**，`ui.py:211-226` 那 16 行"靶点设计页放大布局"全部是死代码。

这类"跨组件拼 HTML"在 Gradio 里永远不成立，重做时应作为硬性禁止项。

---

## 3. 排版

| 项 | 实测 | 问题 |
|---|---|---|
| 字体 | `document.fonts` 中 `Inter: unloaded` ×6 | `@import` 的 Google Fonts 未加载（离线/被墙） |
| 实际字体 | `Inter, "Segoe UI", system-ui, ...` 回退到 Segoe UI | Inter 无 CJK 字形，中文本来也轮不到它 |
| 字号 | 徽章 `0.71em` ≈ 11.4px；结果区 `0.82em` ≈ 13px | 11px 中文低于可读下限 |
| 行高 | 结果区 1.5，正文默认 | 中文正文建议 1.6–1.7 |

**结论**：字体这一层等于没做。中文界面应该直接定义 CJK 字体栈，Latin 字体只在需要时 self-host。

---

## 4. 颜色对比度（WCAG 2.1 AA）

| 组合 | 对比度 | 要求 | 结果 |
|---|---:|---:|:--:|
| `--muted #687380` on `--bg #0c0c14` | **4.04:1** | 4.5 | ❌ |
| `--muted` on `--surface #14141e` | **3.79:1** | 4.5 | ❌ |
| 白字 on `--accent #7b7bf6`（主按钮） | **3.50:1** | 4.5 | ❌ |
| `--text #c2ccd8` on `--bg` | 11.98:1 | 4.5 | ✅ |
| `--heading #dce3ee` on `--section-a` | 14.03:1 | 4.5 | ✅ |
| 结果文字 `#b2baf6` on `#0a0a10` | 10.57:1 | 4.5 | ✅ |

`--muted` 被用在副标题、徽章、表头、页脚——全是小字号，恰好是最需要对比度的地方。主按钮是唯一的高频点击目标，白字 3.5:1 不合格。

**达标替换（已算过）**

```
--muted:  #7d8896   → 5.08:1 on surface   (或 #8b95a5 → 6.04:1)
--accent: #6262e8   → 白字 4.74:1
          或 #5b5bd6 → 白字 5.37:1（更稳）
```

---

## 5. 焦点指示器

```css
/* webapp/ui.py:160-164 */
textarea:focus, input:focus, select:focus {
    outline: none !important;
    box-shadow: inset 0 1px 4px rgba(0,0,0,0.35),
                0 0 0 3px rgba(124,124,248,0.06) !important;   /* 6% 不透明度 */
}
```

6% 不透明度的紫色在深色背景上几乎不可见，同时移除了原生 `outline`。键盘用户无法判断焦点位置，属 WCAG 2.4.7 不达标。Gradio 6 自己的 `:focus-visible` 也被这行覆盖掉了。

**修复**：只对鼠标态去 outline，保留 `:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }`。

---

## 6. 布局

实测首屏（1440×1000，"单步设计"）：

- 左列 `scale=1` 承载"FASTA→PDB 转换 + 上传 PDB + 选择工具"三块，高度约 **1000px**；
- 右列 `scale=2` 只有"工具参数"和"开始"两块，高度约 **450px**；
- 结果是 `.row.unequal-height`，右列下方留下 **~400px 纯空白**（截图中右侧整片空区）；
- 主动作"▶ 开始设计"是整行 100% 宽的大按钮，与上方的参数区割裂；
- 结果卡片在 Row 之外，位于折叠线以下，跑完必须滚屏才看得到。

页签栏：10 个页签，实测 `[role="tablist"]` 宽度 1145px、`flex-wrap: nowrap`，Gradio 用溢出下拉兜底。1440 以下就会开始有页签藏进下拉菜单——导航层级随窗口宽度变化，对"我要找靶点设计在哪"是负帮助。

面板高度：首个 tabpanel 实测 **1339px**，而 CSS 还强制 `.tabs > .tabitem { min-height: 680px }`——既没限制住长的，又给短的留了空。

---

## 7. 交互状态缺失

| 状态 | 现状 | 后果 |
|---|---|---|
| 空态 | 只有 `placeholder="结果将在此显示..."` | 用户不知道下一步该做什么 |
| 加载态 | 只有 `gr.Progress()` 顶部横条 | 长任务（AF2 5–15 min）没有阶段/耗时/ETA |
| 失败态 | 原始 traceback 塞进结果框 | 看不懂，且泄露路径（见 security S1-2） |
| 重跑 | 新结果覆盖旧结果，无历史 | 无法对比两次设计 |
| 取消 | 只有"迭代精修"有个停止按钮，且 `ui.py:926` 创建后**从未绑定** `.click()` | 死按钮 |
| 刷新 | "🔄 刷新状态" `ui.py:505` 创建后**从未绑定** | 死按钮，点了没反应 |

另外 `mpnn_temp`（`ui.py:315`）、`batch_temp`（`ui.py:405`）是自由文本 Textbox，用户输入 `abc` 会一路走到 `float("abc")` 抛异常，再以 traceback 形式回显。应该是 Slider 或带校验的 Number。

---

## 8. 信息架构

现在 10 个页签，按"实现方式"切分，而不是按"用户任务"：

| 页签 | 实际做的事 | 重叠 |
|---|---|---|
| 单步设计 | 单 PDB + BFN/MPNN/ESM-IF | 基线 |
| 批量处理 | 多 PDB + 同上 | 与单步设计同构，只是循环 |
| 结构预测 | AF2 | 独立能力 |
| 设置 | 状态 + 配置编辑 | 运维，不该占一级导航 |
| 靶点设计 | 靶点+表位+约束设计 | = 表位分析 + 单步设计 |
| 迭代精修 | 设计→AF2→筛选循环 | = 单步设计 + 结构预测 + 循环 |
| 置信度评估 | BFN 内部头打分 | 诊断工具 |
| 统一工作流 | 设计→过滤→AF2 | = 单步设计 + 结构预测 |
| 错误折叠疾病 | 疾病库+表位+设计+AF2+精修 | 上述所有 + 知识库 |
| 闭环优化 | 设计→AF2→Rosetta→Pareto | 再一遍 |

**6 个页签在重复"设计"这个动词**，差别只在包装。用户要判断"我该点哪一个"，必须先把 10 个页签都读一遍——这是导航失败，不是审美问题。

科学工作流的真实顺序是：**输入 → 设计 → 验证 → 排序 → 导出**，是一个线性管线，不是 10 个并列工具。

---

## 9. 其他可核实的缺陷

- `webapp/ui.py:1563`：`_mf_epitope_region.value if hasattr(...)` —— `gr.State` 确实有 `.value`，但那是**构建时的初始值 `""`**，不是运行时状态。结果迭代精修拿到空 region，永远走默认值。
- `webapp/ui.py:1241-1247`：注释写着 `# Find disease key from state — we pass it via a trick`，紧接着 `known = []`。"使用文献已知表位"复选框实际是空操作。
- `webapp/ui.py:562`：`_t5_epitope_data` 存了但从未被下游消费。
- `webapp/ui.py:1501`：`mf_run_iterative` 收了 `disease_key` 参数但函数体里没用。
- `app.py:1-14`：文件头 docstring 重复了两遍。
- `webapp/ui.py:1`：`"""app.py split: ui (auto-generated 2026-09-27, bodies verbatim)."""` —— 拆分后每个模块都还在重复 import 整段 20 行的 header，10 个模块里 8 个用不到。

---

## 10. 一句话总结

设计语言是有的，执行是断的；信息架构是真坏的。
重做的正确顺序是：**先修安全 → 再定信息架构 → 最后才是视觉**。反过来做，等于给一个能被 RCE 的界面换壁纸。
