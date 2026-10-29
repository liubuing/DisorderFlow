#!/usr/bin/env python
"""
蛋白质序列设计平台 — Gradio Web 界面（管理增强版）

2026-09-27 拆分：实现位于 webapp/ 包（state/sequtils/design/baselines/af2/pipelines/ui）。
本文件只是入口 + 兼容再导出（modules/ 与 scripts/ 里的 `from app import X` 不受影响）。

访问: http://127.0.0.1:7860  （仅本机访问，外部无法连接）
管理: python manage.py start|stop|restart|status|batch|config|test
"""
#!/usr/bin/env python
"""
蛋白质序列设计平台 — Gradio Web 界面（管理增强版）

访问: http://127.0.0.1:7860  （仅本机访问，外部无法连接）
管理: python manage.py start|stop|restart|status|batch|config|test
"""
import os, json, subprocess, tempfile, re, sys, yaml, time
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import gradio as gr
# 将仓库下的 modules/ 目录插入 sys.path 头部，
# 使后续可以按"裸文件名"导入 target_design_helpers、cascade_filter 等平铺布局的研究模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))
import target_design_helpers as tdh
import cascade_filter as cf
import misfolding_knowledge_base as mkb
import misfolding_pipeline as mfp
import closed_loop_scorer as cls
import closed_loop_orchestrator as clo
from runtime_environment import build_colabfold_command, colabfold_environment

# 星号导入 webapp.facade：把实现层的公开 API 再导出到 app 命名空间，
# 保证历史代码中 `from app import X` 的用法在逻辑拆分到 webapp/ 后仍然可用
from webapp.facade import *  # noqa: F401,F403
from webapp.state import load_app_config, load_bfn
from webapp.ui import create_ui

if __name__ == '__main__':
    # 读取 app_config.yaml，获取服务器 host/port、模型路径等运行配置
    cfg = load_app_config()
    sv = cfg['server']

    print("=" * 60)
    print("  Protein Sequence Design Platform v2.0")
    print(f"  http://{sv['host']}:{sv['port']}")
    print(f"  外部访问: {'允许' if sv['host'] != '127.0.0.1' else '禁止（仅本机）'}")
    print("=" * 60)

    # Preload BFN
    try:
        print("  预加载 BFN 模型...")
        load_bfn()
        print("  BFN ✓")
    except Exception as e:
        print(f"  BFN 跳过: {e}")

    print(f"\n  Web 界面: http://{sv['host']}:{sv['port']}")
    print(f"  管理命令: python manage.py --help")
    print()

    # 构建并启动 Gradio 界面；BFN 预加载失败只打印提示、不阻塞 Web 服务
    app, css = create_ui()
    app.launch(
        server_name=sv['host'],
        server_port=sv['port'],
        share=sv.get('share', False),
        inbrowser=sv.get('auto_open', True),
        css=css,
        show_error=True,
    )
