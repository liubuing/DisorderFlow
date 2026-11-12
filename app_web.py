#!/usr/bin/env python
"""DisorderFlow — 自建 Web 前端（FastAPI + 原生 HTML/CSS/JS）。

启动：
    python app_web.py                 # 默认读取 app_config.yaml 的 host/port
    python app_web.py --port 7862
    python app_web.py --no-preload

安全默认值：
- 默认只监听回环地址；监听非回环且未设置 DF_USER / DF_PASS 时拒绝启动；
- 所有写操作要求自定义请求头（阻断跨站简单请求）；
- 上传文件通过服务端签发的不透明 token 访问，响应中不出现绝对路径；
- 检查点只允许从 logs/ 加载（webapp.state.load_checkpoint）。
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "modules"))


def _preload() -> None:
    from webapp import state

    try:
        state.load_bfn()
        print("  BFN 预加载完成", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  BFN 预加载跳过: {type(exc).__name__}: {exc}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="DisorderFlow web UI")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-preload", action="store_true")
    parser.add_argument("--log-level", default="warning")
    args = parser.parse_args()

    import uvicorn

    from webapp import state
    from webapp.web.server import app

    config = state.load_app_config()
    server = config.get("server", {})
    host = args.host or server.get("host", "127.0.0.1")
    port = args.port or server.get("port", 7860)

    loopback = host in ("127.0.0.1", "localhost", "::1")
    if not loopback and not (os.environ.get("DF_USER") and os.environ.get("DF_PASS")):
        print("\n[拒绝启动] 监听非回环地址时必须设置 DF_USER / DF_PASS")
        print("  set DF_USER=user")
        print("  set DF_PASS=change-me")
        raise SystemExit(2)

    print("=" * 62)
    print("  DisorderFlow — 蛋白质设计与验证")
    print(f"  http://{host}:{port}")
    print(f"  鉴权: {'已启用' if os.environ.get('DF_USER') else '关闭（仅本机）'}")
    print("=" * 62)

    if not args.no_preload:
        threading.Thread(target=_preload, name="bfn-preload", daemon=True).start()

    if loopback and server.get("auto_open", True):
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{host}:{port}")).start()

    uvicorn.run(app, host=host, port=port, log_level=args.log_level, access_log=False)


if __name__ == "__main__":
    main()
