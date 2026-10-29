#!/usr/bin/env python
"""Plan or run a cadence-guarded expanded antibody-IDP RCSB metadata query."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
# RCSB 搜索 API v2; 上次获取记录(含 retrieved_at_utc)用于节奏门控与增量日期
ENDPOINT = "https://search.rcsb.org/rcsbsearch/v2/query"
LAST_ACQUISITION = (
    ROOT / "data/successor_v3_rcsb_snapshot_2026_08_15/acquisition.json"
)
OUTPUT_ROOT = ROOT / "reviewer_outputs/idp_ensemble_expanded_development_v2"
# 查询词: 抗体形式术语 与 IDP(固有无序蛋白) 术语, 两组求交集(AND of ORs)
ANTIBODY_TERMS = [
    "antibody", "Fab", "VHH", "scFv", "nanobody", "single-domain antibody"
]
IDP_TERMS = [
    "amyloid beta", "tau", "alpha-synuclein", "prion protein", "huntingtin",
    "TDP-43", "fused in sarcoma", "islet amyloid polypeptide", "amylin",
]


def canonical_json(value):
    # 键排序+紧凑分隔符的规范化 JSON, 保证同一查询的哈希稳定可复现
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def terminal(term):
    # RCSB 查询树的全文检索单节点
    return {
        "type": "terminal",
        "service": "full_text",
        "parameters": {"value": term},
    }


def query_spec(release_date):
    # 查询结构: (抗体词 OR) AND (IDP 词 OR) AND (初始发布日期 >= 增量截止日);
    # 只取实验结构、紧凑返回、按 rcsb_id 升序, 单页最多 10000 条
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "group", "logical_operator": "or",
                    "nodes": [terminal(term) for term in ANTIBODY_TERMS],
                },
                {
                    "type": "group", "logical_operator": "or",
                    "nodes": [terminal(term) for term in IDP_TERMS],
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_accession_info.initial_release_date",
                        "operator": "greater_or_equal",
                        "value": release_date,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "results_content_type": ["experimental"],
            "results_verbosity": "compact",
            "sort": [{"sort_by": "rcsb_id", "direction": "asc"}],
            "paginate": {"start": 0, "rows": 10000},
        },
    }


def parse_utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


def sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


def build_plan(config_path, now=None):
    # 生成查询计划: 7 天节奏门控(距上次获取不足 7 天则 gate 关闭);
    # 增量截止日期取上次获取日, 只检索其之后新发布的条目
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    acquisition = json.loads(LAST_ACQUISITION.read_text(encoding="ascii"))
    last_retrieved = parse_utc(acquisition["retrieved_at_utc"])
    next_permitted = last_retrieved + timedelta(days=7)
    release_date = last_retrieved.date().isoformat()
    spec = query_spec(release_date)
    stamp = now.strftime("%Y_%m_%d")
    output = OUTPUT_ROOT / f"rcsb_expanded_snapshot_{stamp}.json"
    return {
        "schema_version": 1,
        "status": (
            "eligible_to_execute" if now >= next_permitted else "cadence_gate_closed"
        ),
        "classification": "expanded_antibody_idp_metadata_query_plan",
        "now_utc": now.isoformat().replace("+00:00", "Z"),
        "last_snapshot_retrieved_at_utc": acquisition["retrieved_at_utc"],
        "next_permitted_at_utc": next_permitted.isoformat().replace("+00:00", "Z"),
        "eligible_to_execute": now >= next_permitted,
        "config": str(config_path.relative_to(ROOT)),
        "final_v1_isolation": config["isolation"],
        "query_spec": spec,
        "query_sha256": sha256_bytes(canonical_json(spec).encode("ascii")),
        "output": str(output),
        "access_boundary": (
            "RCSB entry-ID metadata only; every returned entry becomes exposed "
            "development data and is ineligible for final_confirmation_v1"
        ),
    }


def execute(plan, opener=urllib.request.urlopen):
    # 执行查询: 节奏门未开或目标快照已存在时拒绝; POST 规范化查询体,
    # 提取条目 ID(排序大写去重)并连同响应哈希一并冻结落盘
    if not plan["eligible_to_execute"]:
        raise RuntimeError(
            f"Cadence gate closed until {plan['next_permitted_at_utc']}"
        )
    output = Path(plan["output"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite RCSB expanded snapshot: {output}")
    body = canonical_json(plan["query_spec"]).encode("ascii")
    request = urllib.request.Request(
        ENDPOINT,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "DisorderFlow-expanded-IDP-development-v2/1",
        },
        method="POST",
    )
    with opener(request, timeout=120) as response:
        content = response.read()
    payload = json.loads(content)
    ids = [
        row if isinstance(row, str) else row.get("identifier")
        for row in payload.get("result_set", [])
    ]
    if any(not entry_id for entry_id in ids):
        raise ValueError("RCSB response includes a missing entry ID")
    result = {
        **{key: value for key, value in plan.items() if key != "status"},
        "status": "expanded_rcsb_metadata_snapshot_acquired",
        "endpoint": ENDPOINT,
        "http_response_sha256": sha256_bytes(content),
        "entry_count": len(ids),
        "entry_ids": sorted(str(entry_id).upper() for entry_id in ids),
        "claim_boundary": (
            "Expanded retrospective development discovery only; no returned "
            "entry is untouched or final-confirmation eligible"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/benchmarks/idp_ensemble_expanded_development_v2.yml"),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = build_plan(ROOT / args.config)
    result = execute(plan) if args.execute else plan
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
