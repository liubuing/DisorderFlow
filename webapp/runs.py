"""Run manifest — 每次执行落一份可复现的记录。

目录结构::

    runs/20261111-093015_a1b2c3/
        manifest.json
        inputs/           上传文件副本
        outputs/          设计序列、AF2 结构等产物

写入失败不抛给调用方：溯源是增值能力，不该成为运行失败的原因。
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("disorderflow.runs")

PROJECT_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_DIR / "runs"
MAX_COPY_BYTES = 50 * 1024 * 1024


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        candidate = Path(value)
    elif isinstance(value, dict):
        candidate = Path(value.get("path") or value.get("name") or "")
    else:
        raw = getattr(value, "path", None) or getattr(value, "name", None)
        candidate = Path(raw) if raw else None
    if not candidate:
        return None
    try:
        return candidate if candidate.exists() else None
    except OSError:
        return None


def config_fingerprint() -> dict[str, Any]:
    try:
        return {"config_sha256": _sha256(PROJECT_DIR / "app_config.yaml")}
    except OSError:
        return {}


def _describe(path: Path) -> dict[str, Any]:
    # manifest 不记录绝对路径：优先 runs/ 相对路径（artifact_path 依赖它），
    # 其次仓库相对路径，仓库外只留文件名
    resolved = path.resolve()
    shown = path.name
    for base in (RUNS_DIR, PROJECT_DIR):
        try:
            shown = str(resolved.relative_to(base.resolve()))
            break
        except (ValueError, OSError):
            continue
    try:
        return {"path": shown, "name": path.name,
                "size": path.stat().st_size, "sha256": _sha256(path)}
    except OSError as exc:
        return {"path": shown, "error": str(exc)}


@dataclass
class Run:
    run_id: str
    run_dir: Path
    manifest: dict[str, Any] = field(default_factory=dict)
    _t0: float = 0.0

    @property
    def outputs(self) -> Path:
        return self.run_dir / "outputs"

    def record_path(self, path: str | Path) -> None:
        candidate = Path(path)
        if candidate.exists():
            self.manifest.setdefault("artifacts", []).append(_describe(candidate))


def begin(workspace: str, mode: str, inputs: dict[str, Any],
          copy_inputs: bool = True) -> Run:
    stamp = datetime.now().astimezone()
    run_id = f"{stamp:%Y%m%d-%H%M%S}_{int(stamp.timestamp() * 1000) % 1000:03d}"
    run_dir = RUNS_DIR / run_id

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "workspace": workspace,
        "mode": mode,
        "status": "running",
        "started_at": stamp.isoformat(timespec="seconds"),
        "inputs": {},
        "environment": config_fingerprint(),
        "artifacts": [],
    }

    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "outputs").mkdir(exist_ok=True)
    except OSError as exc:
        logger.warning("cannot create run dir %s: %s", run_dir, exc)

    for key, value in inputs.items():
        values: Iterable[Any] = value if isinstance(value, (list, tuple)) else [value]
        described = []
        for index, item in enumerate(values):
            source = _as_path(item)
            if source is None:
                described.append(item if isinstance(item, (str, int, float, bool))
                                 else str(item))
                continue
            entry = _describe(source)
            if copy_inputs:
                try:
                    if source.stat().st_size <= MAX_COPY_BYTES:
                        dest_dir = run_dir / "inputs"
                        dest_dir.mkdir(exist_ok=True)
                        dest = dest_dir / f"{index:02d}_{source.name}"
                        shutil.copy2(source, dest)
                        entry["copied_to"] = str(dest.relative_to(run_dir))
                except OSError:
                    pass
            described.append(entry)
        manifest["inputs"][key] = (described if isinstance(value, (list, tuple))
                                   else described[0])

    run = Run(run_id=run_id, run_dir=run_dir, manifest=manifest, _t0=time.time())
    _write(run)
    return run


def finish(run: Run, status: str, *, error: str | None = None,
           extra: dict[str, Any] | None = None) -> None:
    run.manifest.update({
        "status": status,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "duration_s": round(time.time() - run._t0, 2),
    })
    if error:
        run.manifest["error"] = str(error)[:2000]
    if extra:
        run.manifest.setdefault("extra", {}).update(extra)
    _write(run)


def _write(run: Run) -> None:
    try:
        run.run_dir.mkdir(parents=True, exist_ok=True)
        tmp = run.run_dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(run.manifest, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(run.run_dir / "manifest.json")
    except OSError as exc:
        logger.warning("manifest write failed for %s: %s", run.run_id, exc)


def list_runs(limit: int = 60) -> list[dict[str, Any]]:
    if not RUNS_DIR.exists():
        return []
    records: list[dict[str, Any]] = []
    for entry in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not entry.is_dir():
            continue
        manifest = entry / "manifest.json"
        if not manifest.exists():
            continue
        try:
            records.append(json.loads(manifest.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
        if len(records) >= limit:
            break
    return records


def read_manifest(run_id: str) -> dict[str, Any] | None:
    if not run_id or any(c in run_id for c in "/\\.."):
        return None
    try:
        return json.loads((RUNS_DIR / run_id / "manifest.json")
                          .read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def artifact_path(run_id: str, index: int) -> Path | None:
    manifest = read_manifest(run_id)
    if not manifest:
        return None
    artifacts = manifest.get("artifacts") or []
    if not (0 <= index < len(artifacts)):
        return None
    raw = artifacts[index].get("path")
    if not raw:
        return None
    raw_path = Path(raw)
    # manifest 中的 path 是 runs/ 相对路径；兼容旧的绝对路径记录
    path = (raw_path if raw_path.is_absolute() else RUNS_DIR / raw_path).resolve()
    # 只允许从 runs/ 目录下取文件
    if not path.is_relative_to(RUNS_DIR.resolve()):
        return None
    return path if path.exists() else None
