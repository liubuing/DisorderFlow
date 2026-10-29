#!/usr/bin/env python3
"""Build an instance-keyed authoritative IMGT structural parent cache."""

import argparse
import csv
import gzip
import io
import json
import pickle
import sys
import tarfile
from pathlib import Path

import lmdb
import torch
from Bio.PDB import MMCIFParser


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from disorderflow.datasets.statecontrast_structural import (
    assign_official_imgt_cdrs,
    require_full_antigen_mapping,
)
from disorderflow.data_factory import sha256_file
from disorderflow.utils.protein.parsers import parse_biopython_structure


def chain_ids(value):
    # 把 SAbDab2 的链字段（'/' 分隔，'+' 表示多聚抗原）解析为干净的链 ID 列表。
    return [item.strip() for item in str(value).split("/") if item.strip() and item.strip() != "+"]


def observed_records(records_dir, split, limit):
    # 从 factory 分片记录中筛出指定 split、operation 为 observed_bound 的记录。
    # 参数:
    #   records_dir: factory 记录目录（含 manifest.json 与 gzip 分片）。
    #   split: 目标划分（'train' 或 'validation'）。
    #   limit: 最多收集多少条；None 表示不设上限。
    # 返回: (manifest, records) 二元组。
    manifest = json.loads((records_dir / "manifest.json").read_text(encoding="utf-8"))
    records = []
    for shard in manifest["shards"]:
        with gzip.open(records_dir / shard["path"], "rt", encoding="ascii") as handle:
            for line in handle:
                record = json.loads(line)
                if record["split"] != split or record["lineage"]["operation"] != "observed_bound":
                    continue
                records.append(record)
                if limit is not None and len(records) >= limit:
                    return manifest, records
    return manifest, records


def merge_chains(chains):
    # 把多条抗原链合并为一条记录：张量沿残基维拼接、列表相接、标量取第一条。
    # 抗原可能是多聚体；下游只需要一条统一的抗原对象，不关心链边界。
    if len(chains) == 1:
        return chains[0]
    merged = {}
    for key in chains[0]:
        values = [chain[key] for chain in chains]
        if isinstance(values[0], torch.Tensor):
            merged[key] = torch.cat(values, dim=0)
        elif isinstance(values[0], list):
            merged[key] = sum(values, start=[])
        else:
            merged[key] = values[0]
    return merged


def parse_parent(source, row, record):
    # 解析单个 mmcif 并构建带 IMGT 官方注释的结构父本。
    # 参数:
    #   source: tar 归档中该 instance 的 cif 文件流。
    #   row: SAbDab2 官方 abag_split.csv 中的一行（声明重/轻/抗原链）。
    #   record: factory 记录，提供官方 CDR 序列与抗原序列。
    # 返回: 含 heavy/light/antigen 链张量与 factory_mapping（IMGT 方案、CDR 索引、
    #   抗原索引映射）的字典，直接 pickle 进 LMDB。
    # 校验（任一失败即记为该 instance 的构建失败）:
    #   - 声明的重链、轻链、抗原链必须真实存在于结构中；
    #   - 抗体链与抗原链角色不得重叠；
    #   - 所有重原子坐标必须有限（防 NaN 污染缓存）。
    text = io.StringIO(source.read().decode("utf-8"))
    model = MMCIFParser(QUIET=True).get_structure(row["INSTANCE"], text)[0]
    heavy_id = row.get("Hchain")
    light_id = row.get("Lchain")
    antigen_ids = chain_ids(row.get("agchains", ""))
    if not heavy_id or not light_id or heavy_id not in model or light_id not in model:
        raise ValueError("Declared heavy/light chains are absent")
    if not antigen_ids or any(chain_id not in model for chain_id in antigen_ids):
        raise ValueError("Declared antigen chains are absent")
    if len({heavy_id, light_id, *antigen_ids}) != 2 + len(antigen_ids):
        raise ValueError("Antibody and antigen chain roles overlap")

    heavy, _ = parse_biopython_structure(model[heavy_id])
    light, _ = parse_biopython_structure(model[light_id])
    antigen_parts = [parse_biopython_structure(model[chain_id])[0] for chain_id in antigen_ids]
    antigen = merge_chains(antigen_parts)

    antibody = record["antibody"]
    heavy_cdrs = {name: antibody["cdrs"][name] for name in ("H1", "H2", "H3")}
    light_cdrs = {name: antibody["cdrs"][name] for name in ("L1", "L2", "L3")}
    heavy_spans = {name: antibody["cdr_spans"][name] for name in heavy_cdrs}
    light_spans = {name: antibody["cdr_spans"][name] for name in light_cdrs}
    cdr_indices = {}
    cdr_indices.update(assign_official_imgt_cdrs(
        heavy, antibody["vh"], heavy_cdrs, heavy_spans))
    cdr_indices.update(assign_official_imgt_cdrs(
        light, antibody["vl"], light_cdrs, light_spans))
    antigen["cdr_flag"] = torch.zeros_like(antigen["aa"])
    antigen_indices = require_full_antigen_mapping(antigen, record["antigen"]["sequence"])

    for chain in (heavy, light, antigen):
        if not torch.isfinite(chain["pos_heavyatom"]).all():
            raise ValueError("Non-finite atom coordinates")
    return {
        "id": row["INSTANCE"],
        "pdb_id": row["PDB_ID"],
        "official_instance": row["INSTANCE"],
        "heavy": heavy,
        "light": light,
        "antigen": antigen,
        "factory_mapping": {
            "scheme": "IMGT",
            "source": "SAbDab2 official abag_split.csv",
            "cdr_indices": cdr_indices,
            "antigen_indices": antigen_indices,
        },
    }


def main():
    # 构建 instance 键控的权威 IMGT 结构父本缓存（LMDB）。
    # 流程: 选定 observed_bound 记录 → 在 SAbDab2 归档中按 INSTANCE 匹配
    # 官方链声明 → 流式解析 cif（r|gz 流式遍历，避免整包解压）→ 逐个
    # 解析并以小写 instance 为键写入 LMDB。
    # 稳健性设计:
    #   - candidate-limit/max-success 限制规模（默认最多 64 候选 / 8 成功，
    #     冒烟规模），达到 max-success 即提前终止；
    #   - progress.json 记录成功/失败清单，--resume 可续跑且校验 factory
    #     schema 一致；失败默认跳过，--retry-failures 清空失败表重试；
    #   - 每写 50 个归档成员落一次进度，中断后不丢已完成工作。
    # 报告含两个输出文件的 SHA-256；零成功则整体失败。
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-dir", default="data/statecontrast_factory_v2_1")
    parser.add_argument("--archive", default="data/sabdab2_current/splits.tar.gz")
    parser.add_argument("--output", default="data/statecontrast_structural_smoke/parents.lmdb")
    parser.add_argument("--split", default="train", choices=("train", "validation"))
    parser.add_argument("--candidate-limit", type=int, default=64)
    parser.add_argument("--max-success", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()

    records_dir = ROOT / args.records_dir
    archive = ROOT / args.archive
    output = ROOT / args.output
    ids_path = Path(str(output) + "-ids")
    progress_path = output.with_suffix(".progress.json")
    if not args.resume and (output.exists() or ids_path.exists()):
        raise FileExistsError(f"Refusing to overwrite structural cache: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest, parents = observed_records(records_dir, args.split, args.candidate_limit)
    by_instance = {record["provenance"]["source_record_id"]: record for record in parents}

    with tarfile.open(archive, "r:gz") as tar:
        split_file = tar.extractfile("splits_final/abag_split.csv")
        if split_file is None:
            raise FileNotFoundError("SAbDab2 abag_split.csv is absent")
        rows_by_key = {}
        for row in csv.DictReader(io.TextIOWrapper(split_file, encoding="utf-8")):
            key = row["INSTANCE"].casefold()
            if key in by_instance:
                rows_by_key.setdefault(key, row)
        rows = {row["INSTANCE"]: row for row in rows_by_key.values()}

    env = lmdb.open(str(output), subdir=False, map_size=4 * 1024**3)
    with env.begin() as txn:
        successful = sorted(key.decode() for key, _ in txn.cursor())
    failures = {}
    if args.resume and progress_path.exists():
        previous_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if previous_progress.get("factory_schema") != manifest["schema_version"]:
            raise ValueError("Resume progress schema does not match current factory")
        failures = dict(previous_progress.get("failures", {}))
    if args.retry_failures:
        failures = {}

    def write_progress(status="running"):
        ids_path.write_bytes(pickle.dumps(successful))
        progress = {
            "status": status,
            "factory_schema": manifest["schema_version"],
            "split": args.split,
            "candidate_limit": args.candidate_limit,
            "requested_successes": args.max_success,
            "successful_parents": len(successful),
            "failed_parents": len(failures),
            "parent_ids": successful,
            "failures": failures,
            "output": str(output.relative_to(ROOT)).replace("\\", "/"),
        }
        progress_path.write_text(
            json.dumps(progress, indent=2, sort_keys=True), encoding="utf-8")
        return progress

    write_progress()
    completed = set(successful)
    if not args.retry_failures:
        completed.update(key.casefold() for key in failures)
    wanted = {
        f"splits_final/{instance}.cif": (instance, rows.get(instance))
        for instance in rows if instance.casefold() not in completed
    }
    examined = 0
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if len(successful) >= args.max_success:
                break
            selected = wanted.get(member.name)
            if selected is None:
                continue
            examined += 1
            instance, row = selected
            source = tar.extractfile(member)
            if source is None or row is None:
                failures[instance] = "missing archive row or CIF stream"
                continue
            try:
                parent = parse_parent(source, row, by_instance[instance.casefold()])
                with env.begin(write=True) as txn:
                    if txn.get(instance.casefold().encode()) is not None:
                        raise ValueError("duplicate structural parent key")
                    txn.put(instance.casefold().encode(), pickle.dumps(parent))
                successful.append(instance.casefold())
                successful.sort()
            except Exception as error:
                failures[instance] = f"{type(error).__name__}: {error}"
            if examined % 50 == 0:
                write_progress()
                print(json.dumps({
                    "examined_this_run": examined,
                    "successful_parents": len(successful),
                    "failed_parents": len(failures),
                }), flush=True)
    env.close()
    report = write_progress("pass" if successful else "fail")
    report.update({
        "examined_this_run": examined,
        "parent_lmdb_sha256": sha256_file(output),
        "parent_ids_sha256": sha256_file(ids_path),
    })
    output.with_suffix(".build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not successful:
        raise RuntimeError("No authoritative structural parents were built")


if __name__ == "__main__":
    main()
