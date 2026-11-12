# DisorderFlow 前端审计与重做方案

日期：2026-11-10 · 分支 `release` · 审计对象 `app.py` + `webapp/`（Gradio 6.17.3 / Python 3.14.5）

## 状态：已重做（v3，自建前端）

主入口 `app_web.py`（FastAPI + 原生 HTML/CSS/JS，仪器风格）。
旧 Gradio 界面（`app.py` / `webapp/ui.py` / `webapp/facade.py`）已于 2026-11-11 删除，
`manage.py start` 现在直接启动 `app_web.py`。
接口测试 `python -m pytest tests/test_web_api.py -q`（13 项）。
细节见 [`IMPLEMENTATION.md`](./IMPLEMENTATION.md)。

## 一句话结论

**安全上有一个必须先堵的洞，视觉上是"设计丢了"而不是"没有设计"，架构上确实该重做。**
建议顺序：先半天修安全 → 再按 4 个工作区重做 Gradio 前端（8–11 人日）→ React 留到有公网/多用户需求时再说。

## 三份文档

| 文档 | 内容 | 关键结论 |
|---|---|---|
| [`security-findings.md`](./security-findings.md) | S0/S1/S2 漏洞清单、证据、修复代码 | 44 个 handler 全部无认证可调；配置可写 + `torch.load(weights_only=False)` 构成 RCE 链；`shell=True` 拼接客户端可控字符串 |
| [`design-audit.md`](./design-audit.md) | 实测 CSS 命中率、对比度、布局、IA | 12 个 CSS 选择器命中数为 0（Gradio 3/4 类名）；主按钮白字 3.5:1 不达标；10 个页签里 6 个重复 |
| [`redesign-plan.md`](./redesign-plan.md) | 选型、信息架构、设计令牌、组件、路线图 | 10 页签 → 4 工作区；只样式化自有 `elem_classes`；Run 模型解决溯源 |
| [`IMPLEMENTATION.md`](./IMPLEMENTATION.md) | 已实施结构、接口、安全模型、已知缺口 | `app_web.py` 可运行，13 项接口测试通过 |

证据：`evidence/initial-screen-1440x1000.png`

## 三个最重要的数字

1. **44 / 44** — 所有后端回调都以公开 HTTP 端点暴露，无鉴权。实测 `curl` 直接调用成功。
2. **12** — 样式表里在 Gradio 6 上命中数为 0 的选择器数量（`.tab-nav`、`.gr-box`、`.gr-form`、`.gr-panel`、`.accordion`、`.gr-button` …）。
3. **3.50:1** — 主按钮白字对比度，低于 WCAG AA 的 4.5:1。

## 立即可以做的四件事（半天内，不改业务逻辑）

```python
# 1. 内部 handler 退出 API
btn.click(fn, inputs, outputs, api_visibility="private")

# 2. 启动参数
app.launch(auth=(USER, PASS), allowed_paths=[UPLOAD_ROOT],
           max_file_size="200mb", show_error=False)

# 3. 检查点只加载张量
torch.load(path, map_location=DEVICE, weights_only=True)

# 4. AF2 命令去掉 shell
subprocess.run([exe, "--num-models", "1", ...], shell=False)
```

前 4 项消除"未认证 → 远程代码执行"整条链。

## 下一步

- 要我直接实施 Phase 0（安全热修）吗？改动集中在 `app.py`、`webapp/state.py`、`modules/misfolding_pipeline.py`，可以不碰 UI。
- 或者先做 Phase 1 的骨架 + 设计工作区，看到实际效果再决定后续。
