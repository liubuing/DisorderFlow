"""Web 端接口契约测试。

守护三件事：
1. 写操作必须带客户端标识（阻断跨站简单请求）；
2. 上传文件只通过服务端 token 引用，产物只能从 runs/ 目录取；
3. 模板生成与表位打分这些纯 CPU 路径的结果结构稳定。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from webapp import runs as runlog
from webapp.web import server


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "UPLOAD_ROOT", tmp_path / "uploads")
    monkeypatch.setattr(runlog, "RUNS_DIR", tmp_path / "runs")
    (tmp_path / "uploads").mkdir(parents=True, exist_ok=True)
    return TestClient(server.app)


def _post(client, path, payload):
    return client.post(path, json=payload, headers={"X-DF": "1"})


# ── 基础 ────────────────────────────────────────────────────────────────────

def test_index_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "DisorderFlow" in res.text
    assert "/static/app.js" in res.text


def test_static_assets_are_served(client):
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_write_requires_client_header(client):
    res = client.post("/api/template", json={"fasta": ">x\nMKTA"})
    assert res.status_code == 403


def test_unknown_op_is_rejected(client):
    res = _post(client, "/api/run", {"op": "rm_rf", "params": {}})
    assert res.status_code == 400


def test_run_without_upload_is_rejected(client):
    res = _post(client, "/api/run", {"op": "design_single", "params": {"pdb_id": "nope"}})
    assert res.status_code == 400
    assert "文件不存在" in res.json()["detail"]


# ── 模板与表位 ──────────────────────────────────────────────────────────────

def test_template_generation_returns_chains(client):
    res = _post(client, "/api/template",
                {"fasta": ">probe\nMKTAYIAKQRQISFVKSHFSRQDILDLWQ"})
    assert res.status_code == 200
    data = res.json()
    assert data["chains"][0]["id"] == "A"
    assert data["chains"][0]["length"] == 29
    assert data["size"] > 0


def test_template_accepts_bare_sequence(client):
    """没有 > 头的裸序列也应可用（用户常常只粘贴氨基酸）。"""
    res = _post(client, "/api/template", {"fasta": "MKTAYIAKQRQISFVK"})
    assert res.status_code == 200
    assert res.json()["chains"][0]["length"] == 16


def test_epitope_scan_returns_rows(client):
    made = _post(client, "/api/template",
                 {"fasta": ">probe\nMKTAYIAKQRQISFVKSHFSRQDILDLWQ"}).json()
    res = _post(client, "/api/epitope",
                {"upload_id": made["id"], "chain": "A", "top_n": 5})
    assert res.status_code == 200
    data = res.json()
    assert data["columns"] == ["残基号", "残基", "综合分", "相对 SASA", "亲水性", "凸出度"]
    assert data["rows"]
    assert all(isinstance(row[2], float) and row[2] > 0 for row in data["rows"])
    assert data["region"].startswith("A:")


def test_epitope_unknown_upload(client):
    res = _post(client, "/api/epitope", {"upload_id": "deadbeef", "chain": "A"})
    assert res.status_code == 400


# ── 记录与产物 ──────────────────────────────────────────────────────────────

def test_artifact_path_traversal_is_blocked():
    assert runlog.artifact_path("../../etc/passwd", 0) is None
    assert runlog.artifact_path("valid", -1) is None
    assert runlog.artifact_path("", 0) is None


def test_run_manifest_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", tmp_path / "runs")
    sample = tmp_path / "input.pdb"
    sample.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n")

    record = runlog.begin("design", "单结构", {"pdb": sample})
    output = record.outputs / "design.fasta"
    output.write_text(">x\nMKTA\n")
    record.record_path(output)
    runlog.finish(record, "completed", extra={"n_rows": 1})

    data = json.loads((record.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert data["status"] == "completed"
    assert data["inputs"]["pdb"]["sha256"]
    assert runlog.artifact_path(record.run_id, 0) == output.resolve()

    listed = runlog.list_runs()
    assert listed and listed[0]["run_id"] == record.run_id


# ── 配置 ────────────────────────────────────────────────────────────────────

def test_config_rejects_broken_yaml(client):
    res = client.put("/api/config", json={"text": "a: [unclosed"},
                     headers={"X-DF": "1"})
    assert res.status_code == 400


def test_config_rejects_missing_sections(client):
    res = client.put("/api/config", json={"text": "server: {host: 127.0.0.1, port: 1}"},
                     headers={"X-DF": "1"})
    assert res.status_code == 400
    assert "缺少必需节" in res.json()["detail"]
