"""FastAPI 服务：静态前端 + JSON API + SSE 进度。

与 Gradio 版本的区别：
- 所有文件都通过服务端签发的不透明 token 访问，响应里不出现绝对路径；
- 重任务跑在单线程池里，SSE 推送阶段与百分比，支持协作式取消；
- 默认仅回环；非回环且未配置鉴权时拒绝启动（在 app_web.py 里检查）。
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import secrets
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from webapp import ops, runs as runlog, state

logger = logging.getLogger("disorderflow.web")

PROJECT_DIR = Path(__file__).resolve().parents[2]
WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
UPLOAD_ROOT = PROJECT_DIR / "data" / "web_uploads"
MAX_UPLOAD = 200 * 1024 * 1024


# ── 会话内的文件/任务登记 ───────────────────────────────────────────────────

@dataclass
class Upload:
    id: str
    path: Path
    name: str
    size: int


UPLOADS: dict[str, Upload] = {}


class JobCancelled(Exception):
    pass


@dataclass
class Job:
    id: str
    op: str
    label: str
    status: str = "queued"          # queued | running | done | error | cancelled
    pct: float = 0.0
    stage: str = ""
    started: float = 0.0
    finished: float = 0.0
    result: dict[str, Any] | None = None
    error: str = ""
    run_id: str = ""
    cancel: bool = False

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id, "status": self.status, "pct": round(self.pct, 1),
            "stage": self.stage, "result": self.result, "error": self.error,
            "run_id": self.run_id,
            "elapsed": round((self.finished or time.time()) - self.started, 1)
            if self.started else 0,
        }


JOBS: dict[str, Job] = {}
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="df-job")


def register_upload(path: Path, name: str | None = None) -> Upload:
    uid = uuid.uuid4().hex[:12]
    item = Upload(id=uid, path=Path(path), name=name or Path(path).name,
                  size=Path(path).stat().st_size if Path(path).exists() else 0)
    UPLOADS[uid] = item
    return item


def _upload(uid: str | None) -> Path:
    item = UPLOADS.get(uid or "")
    if not item or not item.path.exists():
        raise HTTPException(400, f"文件不存在或已过期: {uid}")
    return item.path


def _sanitize(name: str) -> str:
    base = Path(name or "upload").name
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base).strip("._") or "upload"
    return base[:120]


# ── 应用 ────────────────────────────────────────────────────────────────────

app = FastAPI(title="DisorderFlow", docs_url=None, redoc_url=None, openapi_url=None)


def _auth_ok(request: Request) -> bool:
    import os

    user = os.environ.get("DF_USER")
    password = os.environ.get("DF_PASS")
    if not user or not password:
        return True
    header = request.headers.get("authorization", "")
    if not header.startswith("Basic "):
        return False
    try:
        raw = base64.b64decode(header[6:]).decode("utf-8")
    except Exception:  # noqa: BLE001
        return False
    given_user, _, given_pass = raw.partition(":")
    return (secrets.compare_digest(given_user, user)
            and secrets.compare_digest(given_pass, password))


@app.middleware("http")
async def _guard(request: Request, call_next):
    path = request.url.path
    if path.startswith("/static") or path == "/":
        return await call_next(request)
    if not _auth_ok(request):
        return JSONResponse({"detail": "需要登录"}, status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="DisorderFlow"'})
    # 阻断跨站简单请求：浏览器跨源无法自带自定义头
    if request.method in ("POST", "PUT", "DELETE") and request.headers.get("x-df") != "1":
        return JSONResponse({"detail": "缺少客户端标识"}, status_code=403)
    return await call_next(request)


# ── 静态资源 ────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html",
                        headers={"Cache-Control": "no-store"})


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── 状态与知识库 ────────────────────────────────────────────────────────────

@app.get("/api/bootstrap")
async def bootstrap():
    import torch

    config = state.load_app_config()
    checkpoint = ""
    try:
        checkpoint = Path(state.get_bfn_ckpt()).name
    except Exception:  # noqa: BLE001
        pass

    return {
        "status": {
            "device": state.DEVICE,
            "cuda": bool(torch.cuda.is_available()),
            "torch": torch.__version__,
            "bfn_ready": state._bfn_model is not None,
            "checkpoint": checkpoint,
        },
        "knowledge": _knowledge(),
        "runs": [_run_row(item) for item in runlog.list_runs(40)],
        "server": config.get("server", {}),
    }


def _knowledge() -> dict[str, Any]:
    import misfolding_knowledge_base as mkb

    diseases = []
    for key, name_cn, name_en in mkb.list_diseases():
        target = mkb.get_target(key) or {}
        diseases.append({
            "key": key, "cn": name_cn, "en": name_en,
            "idp": bool(target.get("idp_flag")),
            "summary": mkb.get_target_summary(key),
            "conformations": [
                {"key": ck, "pdb": cd.get("pdb_id", ""), "type": cd.get("type", ""),
                 "resolution": cd.get("resolution", "")}
                for ck, cd in (mkb.get_conformations(key) or {}).items()
            ],
        })

    scaffolds = []
    for entry in mkb.list_scaffolds():
        detail = mkb.get_scaffold(entry[0]) or {}
        scaffolds.append({
            "key": entry[0], "name": entry[1], "source": entry[2] if len(entry) > 2 else "",
            "length": detail.get("length", ""),
            "cdr": detail.get("cdr_definitions", {}),
            "notes": detail.get("notes", ""),
            "sequence": detail.get("sequence", ""),
        })

    return {"diseases": diseases, "scaffolds": scaffolds}


def _run_row(manifest: dict) -> dict:
    return {
        "run_id": manifest.get("run_id", ""),
        "started_at": manifest.get("started_at", ""),
        "workspace": manifest.get("workspace", ""),
        "mode": manifest.get("mode", ""),
        "status": manifest.get("status", ""),
        "duration_s": manifest.get("duration_s"),
        "artifacts": len(manifest.get("artifacts") or []),
    }


# ── 上传 ────────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload(request: Request, name: str = "upload.bin"):
    safe = _sanitize(name)
    uid = uuid.uuid4().hex[:12]
    folder = UPLOAD_ROOT / uid
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / safe

    size = 0
    try:
        with open(dest, "wb") as handle:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_UPLOAD:
                    handle.close()
                    shutil.rmtree(folder, ignore_errors=True)
                    raise HTTPException(413, "文件超过 200MB 上限")
                handle.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(500, f"写入失败: {exc}") from exc

    item = register_upload(dest, safe)
    chains = []
    if safe.lower().endswith((".pdb", ".ent")) and size < 60 * 1024 * 1024:
        try:
            chains = ops.chain_summary(str(dest))["chains"]
        except Exception:  # noqa: BLE001
            chains = []

    return {"id": item.id, "name": item.name, "size": item.size, "chains": chains}


@app.post("/api/template")
async def template(payload: dict):
    text = payload.get("fasta") or ""
    if not text.strip():
        raise HTTPException(400, "请提供 FASTA 序列")
    # 允许直接粘贴裸序列（没有 > 头）
    if not any(line.startswith(">") for line in text.splitlines()):
        text = ">template" + chr(10) + text
    folder = UPLOAD_ROOT / uuid.uuid4().hex[:12]
    folder.mkdir(parents=True, exist_ok=True)
    try:
        path = ops.write_temp_pdb(text, folder)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc
    item = register_upload(path)
    return {"id": item.id, "name": item.name, "size": item.size,
            "chains": ops.chain_summary(str(path))["chains"]}


@app.post("/api/structure")
async def structure(payload: dict):
    path = _upload(payload.get("upload_id"))
    try:
        return ops.structure_preview(str(path))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/epitope")
async def epitope(payload: dict):
    path = _upload(payload.get("upload_id"))
    chain = payload.get("chain") or "A"
    top_n = int(payload.get("top_n") or 20)
    try:
        return ops.epitope_scan(str(path), chain, top_n)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/disease/load")
async def disease_load(payload: dict):
    import misfolding_knowledge_base as mkb
    import misfolding_pipeline as mfp

    key = payload.get("disease")
    conformation = payload.get("conformation")
    known = {item[0] for item in mkb.list_diseases()}
    if key not in known:
        raise HTTPException(400, "未知的疾病靶点")

    try:
        result = mfp.MisfoldingDesignPipeline().prepare_target(key, conformation)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc
    if not result.get("success"):
        raise HTTPException(400, str(result.get("error") or "加载失败"))

    item = register_upload(Path(result["pdb_path"]))
    return {
        "upload_id": item.id,
        "chain": result["chain_id"],
        "conformation": result.get("conformation_key"),
        "info": mkb.get_target_summary(key),
        "is_idp": bool(result.get("is_idp")),
        "idp_warning": result.get("idp_warning") or "",
    }


# ── 任务 ────────────────────────────────────────────────────────────────────

def _resolver(params: dict) -> dict:
    """把客户端的不透明 token 换成服务端路径；不信任任何原始路径。"""
    resolved = dict(params)
    if params.get("pdb_id"):
        resolved["pdb_path"] = str(_upload(params["pdb_id"]))
    if params.get("target_id"):
        resolved["target_pdb"] = str(_upload(params["target_id"]))
    if params.get("antibody_id"):
        resolved["antibody_pdb"] = str(_upload(params["antibody_id"]))
    if params.get("file_ids"):
        resolved["files"] = [{"path": str(_upload(uid))} for uid in params["file_ids"]]
    return resolved


OPS: dict[str, Callable[[dict, Path, Callable], ops.OpResult]] = {
    "design_single": lambda p, out, prog: ops.design_single(p["pdb_path"], p, out, prog),
    "design_batch": lambda p, out, prog: ops.design_batch(p["files"], p, out, prog),
    "design_target": lambda p, out, prog: ops.design_target(p, out, prog),
    "design_disease": lambda p, out, prog: ops.design_disease(p, out, prog),
    "validate_af2": lambda p, out, prog: ops.validate_af2(p.get("fasta", ""), p, out, prog),
    "validate_confidence": lambda p, out, prog: ops.validate_confidence(
        p["pdb_path"], p, out, prog),
    "validate_iterative": lambda p, out, prog: ops.validate_iterative(
        p["pdb_path"], p, out, prog),
    "validate_loop": lambda p, out, prog: ops.validate_loop(p["pdb_path"], p, out, prog),
}

WORKSPACE = {
    "design_single": ("design", "单结构"),
    "design_batch": ("design", "批量"),
    "design_target": ("design", "靶点约束"),
    "design_disease": ("design", "疾病靶点"),
    "validate_af2": ("validate", "结构预测"),
    "validate_confidence": ("validate", "置信度"),
    "validate_iterative": ("validate", "迭代精修"),
    "validate_loop": ("validate", "闭环优化"),
}


def _execute(job: Job, params: dict) -> None:
    workspace, mode = WORKSPACE[job.op]
    record = runlog.begin(workspace, mode, {
        key: value for key, value in params.items()
        if isinstance(value, (str, int, float, bool))
    })
    job.run_id = record.run_id
    job.status = "running"
    job.started = time.time()

    def progress(value: float = 0, desc: str | None = None, **_kwargs) -> None:
        if job.cancel:
            raise JobCancelled()
        job.pct = max(job.pct, min(99.0, float(value) * 100))
        if desc:
            job.stage = str(desc)

    try:
        outdir = record.outputs
        result = OPS[job.op](params, outdir, progress)
        for artifact in result.artifacts:
            record.record_path(artifact)
        runlog.finish(record, "completed" if result.status != "error" else "failed",
                      error=result.log if result.status == "error" else None,
                      extra={"n_rows": len(result.rows)})

        names = []
        artifacts = record.manifest.get("artifacts") or []
        for index, item in enumerate(artifacts):
            names.append({"name": item.get("name", "artifact"),
                          "url": f"/api/artifact/{record.run_id}/{index}",
                          "size": item.get("size", 0)})
        job.result = result.as_dict([n["name"] for n in names])
        job.result["artifacts"] = names
        job.result["run_id"] = record.run_id
        job.pct = 100
        job.status = "done"
    except JobCancelled:
        runlog.finish(record, "cancelled")
        job.status = "cancelled"
        job.stage = "已取消"
    except Exception as exc:  # noqa: BLE001
        logger.exception("job %s failed", job.id)
        runlog.finish(record, "failed", error=str(exc))
        job.status = "error"
        job.error = f"{type(exc).__name__}: {exc}"
    finally:
        job.finished = time.time()


@app.post("/api/run")
async def run(payload: dict):
    op = payload.get("op")
    if op not in OPS:
        raise HTTPException(400, f"未知操作: {op}")
    params = _resolver(payload.get("params") or {})

    job = Job(id=uuid.uuid4().hex[:12], op=op, label=WORKSPACE[op][1])
    JOBS[job.id] = job
    EXECUTOR.submit(_execute, job, params)
    return {"job_id": job.id, "run_id": job.run_id}


@app.get("/api/jobs/{job_id}")
async def job_state(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    return job.snapshot()


@app.post("/api/jobs/{job_id}/cancel")
async def job_cancel(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    job.cancel = True
    return {"ok": True}


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")

    async def stream():
        last = None
        while True:
            snapshot = job.snapshot()
            payload = json.dumps(snapshot, ensure_ascii=False)
            if payload != last:
                yield f"data: {payload}\n\n"
                last = payload
            if job.status in ("done", "error", "cancelled"):
                yield "event: end\ndata: {}\n\n"
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ── 运行记录与产物 ──────────────────────────────────────────────────────────

@app.get("/api/runs")
async def list_runs():
    return {"runs": [_run_row(item) for item in runlog.list_runs(60)]}


@app.get("/api/runs/{run_id}")
async def read_run(run_id: str):
    manifest = runlog.read_manifest(run_id)
    if not manifest:
        raise HTTPException(404, "记录不存在")
    artifacts = [
        {"name": item.get("name", ""), "size": item.get("size", 0),
         "sha256": (item.get("sha256") or "")[:16],
         "url": f"/api/artifact/{run_id}/{index}"}
        for index, item in enumerate(manifest.get("artifacts") or [])
    ]
    return {"manifest": manifest, "artifacts": artifacts}


@app.get("/api/artifact/{run_id}/{index}")
async def artifact(run_id: str, index: int):
    path = runlog.artifact_path(run_id, index)
    if not path:
        raise HTTPException(404, "产物不存在")
    return FileResponse(path, filename=path.name)


# ── 配置与模型 ──────────────────────────────────────────────────────────────

@app.get("/api/config")
async def get_config():
    return {"path": state.CONFIG_FILE.name,
            "text": state.CONFIG_FILE.read_text(encoding="utf-8")}


@app.put("/api/config")
async def put_config(payload: dict):
    text = payload.get("text") or ""
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return JSONResponse({"detail": f"YAML 解析失败: {exc}"}, status_code=400)

    if not isinstance(parsed, dict):
        return JSONResponse({"detail": "配置顶层必须是映射"}, status_code=400)
    for key in ("server", "models", "alphafold", "bfn_defaults"):
        if key not in parsed:
            return JSONResponse({"detail": f"缺少必需节: {key}"}, status_code=400)

    target = state.CONFIG_FILE
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = target.with_name(f"{target.name}.bak-{stamp}")
    try:
        if target.exists():
            shutil.copy2(target, backup)
        tmp = target.with_suffix(".yaml.tmp")
        tmp.write_text(yaml.dump(parsed, allow_unicode=True, default_flow_style=False),
                       encoding="utf-8")
        tmp.replace(target)
    except OSError as exc:
        return JSONResponse({"detail": f"写入失败: {exc}"}, status_code=500)

    state._app_config = None
    return {"ok": True, "backup": backup.name}


@app.post("/api/models/{name}/load")
async def load_model(name: str):
    try:
        if name == "bfn":
            state.load_bfn()
        elif name == "esmif":
            state.load_esmif()
        else:
            raise HTTPException(400, "未知模型")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc
    return {"ok": True}
