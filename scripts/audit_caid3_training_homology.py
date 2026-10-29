#!/usr/bin/env python
"""Audit CAID3 against the exact balanced-v4 training sequence records."""

# 本脚本在用 CAID3 外部基准打分之前，先审计其与本项目训练集
# （balanced-v4 训练序列记录，存储于 lookup pickle + LMDB）的序列同源性：
# 用 MMseqs2 easy-search 将 CAID3 序列对训练集搜索，按恒等度/覆盖率阈值
# 判定同源（需排除）与独立（可参与外评）目标，并结合聚类清单区分
# "独立但簇号未解析"与"可用于确认性评估"的目标。
# 输出：同源审计 JSON 与训练集 UniRef50 簇号清单。

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path

import lmdb

ROOT = Path(__file__).resolve().parents[1]


# 清洗序列：转大写并仅保留 20 种标准氨基酸，其余字符一律剔除
def clean_sequence(sequence):
    return "".join(residue for residue in sequence.upper() if residue in "ACDEFGHIKLMNPQRSTVWY")


# 读取 FASTA：以 ">" 后第一个空白前的字段作为记录 ID，拼接多行序列
def read_fasta(path):
    records = []
    target_id = None
    sequence = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            if target_id is not None:
                records.append({"id": target_id, "sequence": "".join(sequence)})
            target_id = line[1:].split()[0]
            sequence = []
        elif line:
            sequence.append(line)
    if target_id is not None:
        records.append({"id": target_id, "sequence": "".join(sequence)})
    return records


# 加载训练集记录：先读 lookup pickle 获得记录 ID 清单，再从只读 LMDB
# 逐条取回并清洗序列；lookup 中有而 LMDB 缺失的记录视为致命错误。
def load_training(lookup_path, lmdb_path):
    envelope = pickle.loads(Path(lookup_path).read_bytes())
    record_ids = sorted(envelope["profiles"])
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, subdir=True)
    records = []
    missing = []
    with env.begin() as transaction:
        for record_id in record_ids:
            raw = transaction.get(str(record_id).encode("ascii"))
            if raw is None:
                missing.append(record_id)
                continue
            record = pickle.loads(raw)
            sequence = clean_sequence(str(record.get("sequence", "")))
            if sequence:
                records.append({"id": str(record_id), "sequence": sequence})
    env.close()
    if missing:
        raise ValueError(f"Training LMDB is missing lookup records: {missing[:10]}")
    return records, envelope


# 以 ASCII + LF 换行写出 FASTA，保证检索输入字节级可复现
def write_fasta(path, records):
    with Path(path).open("w", encoding="ascii", newline="\n") as handle:
        for record in records:
            handle.write(f">{record['id']}\n{clean_sequence(record['sequence'])}\n")


# 主入口：读 CAID3 与训练集 → MMseqs2 搜索 → 按命中划分同源/独立目标 →
# 结合簇清单给出可评估目标 → 写审计 JSON 与训练簇号清单。
# 搜索参数：min-seq-id 0.3、coverage 0.8（cov-mode 0 按查询覆盖），
# 与输出 threshold 字段一致，用作同源性排除的统一阈值。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caid", default=str(ROOT / "data/caid3/disorder_nox/sequences.fasta"))
    parser.add_argument(
        "--lookup", default=str(ROOT / "data/disorder_supervision/train_afdb_balanced_v3.pkl")
    )
    parser.add_argument("--lmdb", default=str(ROOT / "data/confidence_disprot_afdb_v1"))
    parser.add_argument("--mmseqs", default="mmseqs")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--out", default=str(ROOT / "data/caid3/disorder_nox_training_homology.json")
    )
    parser.add_argument(
        "--training-clusters-out",
        default=str(ROOT / "data/caid3/balanced_v4_training_clusters.txt"),
    )
    parser.add_argument(
        "--cluster-manifest",
        default=str(ROOT / "data/caid3/disorder_nox_clusters.json"),
    )
    args = parser.parse_args()
    if shutil.which(args.mmseqs) is None:
        raise FileNotFoundError(f"MMseqs2 not found: {args.mmseqs}")
    caid = read_fasta(args.caid)
    training, envelope = load_training(args.lookup, args.lmdb)
    with tempfile.TemporaryDirectory(prefix="caid3_homology_") as temporary:
        temporary = Path(temporary)
        query = temporary / "caid3.fasta"
        target = temporary / "balanced_v4_train.fasta"
        result = temporary / "hits.tsv"
        write_fasta(query, caid)
        write_fasta(target, training)
        command = [
            args.mmseqs,
            "easy-search",
            str(query),
            str(target),
            str(result),
            str(temporary / "work"),
            "--min-seq-id",
            "0.3",
            "-c",
            "0.8",
            "--cov-mode",
            "0",
            "--format-output",
            "query,target,fident,qcov,tcov,evalue",
            "--threads",
            str(args.threads),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(completed.stderr[-4000:])
        hits = {}
        if result.exists():
            for line in result.read_text().splitlines():
                query_id, target_id, identity, qcov, tcov, evalue = line.split("\t")
                hits.setdefault(query_id, []).append(
                    {
                        "training_record": target_id,
                        "identity_percent": float(identity),
                        "query_coverage": float(qcov),
                        "target_coverage": float(tcov),
                        "evalue": float(evalue),
                    }
                )
    independent = sorted(record["id"] for record in caid if record["id"] not in hits)
    cluster_manifest = json.loads(Path(args.cluster_manifest).read_text())
    unresolved = sorted(
        target_id
        for target_id in independent
        if str(cluster_manifest.get(target_id, "unresolved:")).startswith("unresolved:")
    )
    eligible = sorted(set(independent) - set(unresolved))
    training_clusters = sorted(
        {
            str(item.get("cluster_id"))
            for item in envelope["provenance"].values()
            if item.get("cluster_id")
        }
    )
    output = {
        "schema_version": 1,
        "status": "sequence_homology_audited_before_external_scoring",
        "threshold": {"minimum_identity": 0.3, "coverage": 0.8, "coverage_mode": 0},
        "caid_targets": len(caid),
        "training_records": len(training),
        "training_uniref50_clusters": len(training_clusters),
        "training_cluster_ids": training_clusters,
        "homology_excluded": len(hits),
        "homology_independent": len(independent),
        "independent_ids": independent,
        "unresolved_cluster_ids": unresolved,
        "eligible_ids": eligible,
        "hits": dict(sorted(hits.items())),
        "inputs": {
            "caid_sha256": hashlib.sha256(Path(args.caid).read_bytes()).hexdigest(),
            "lookup_sha256": hashlib.sha256(Path(args.lookup).read_bytes()).hexdigest(),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    Path(args.training_clusters_out).write_text(
        "\n".join(training_clusters) + "\n", encoding="ascii"
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in output.items()
                if key not in {"training_cluster_ids", "independent_ids", "eligible_ids", "hits"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
