#!/usr/bin/env python
"""Audit H3 localization and pose availability for the admitted v3 cohort."""

# 目的：对已准入（admitted）的 v3 候选队列做生成前就绪审计：用 Anarcii 抗体编号
# （转 Chothia 方案）定位每条重链的 CDR-H3 区间与序列，并统计每个组分可用的
# 已接受姿态数。仅当全部组分 H3 定位成功且姿态数 >=2 时判定 generation_ready。
# 输出仅表达"生成就绪性"，不包含任何候选或性能结论（见 claim_boundary）。

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from anarcii import Anarcii
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1

# (93, 102)：Chothia 编号方案中 CDR-H3 的起止残基号，用于从编号结果切出 H3。
CHOTHIA_H3 = (93, 102)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chain_sequence(chain):
    return "".join(
        seq1(residue.resname)
        for residue in chain
        if "CA" in residue and is_aa(residue, standard=True)
    )


def h3_records(sequences):
    # 抗体编号：Anarcii 以 accuracy 模式编号后统一转到 Chothia 方案，
    # 输出每条链的 H3 零基索引与序列；编号失败或无 H3 残基则 ready=False。
    model = Anarcii(
        seq_type="antibody", mode="accuracy", batch_size=16,
        cpu=True, ncpu=1, verbose=False,
    )
    numbered = model.number(sequences)
    numbered = model.to_scheme("chothia")
    output = {}
    for key, row in numbered.items():
        if not row or row.get("error"):
            output[key] = {"ready": False, "error": row.get("error") if row else "no result"}
            continue
        query_index = int(row["query_start"])
        indices, sequence = [], []
        for (position, _insertion), aa in row["numbering"]:
            if aa == "-":
                continue
            if CHOTHIA_H3[0] <= int(position) <= CHOTHIA_H3[1]:
                indices.append(query_index)
                sequence.append(aa)
            query_index += 1
        output[key] = {
            "ready": bool(indices),
            "chain_type": row["chain_type"],
            "h3_indices_zero_based": indices,
            "h3_sequence": "".join(sequence),
        }
    return output


def audit(admission_path, coordinate_dir, output, pose_manifest=None):
    # 主审计：拒绝覆盖已有输出；准入状态必须是 untouched_cohort_admitted；
    # 提取抗体链序列做 H3 编号，结合（可选的）姿态清单统计每组分姿态可用性。
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite generation audit: {output}")
    admission = json.loads(admission_path.read_text(encoding="ascii"))
    if admission.get("status") != "untouched_cohort_admitted":
        raise RuntimeError("Cohort is not admitted")
    poses_by_component = {}
    if pose_manifest is not None:
        manifest = json.loads(Path(pose_manifest).read_text(encoding="ascii"))
        for row in manifest.get("components", []):
            poses_by_component[row["component_id"]] = [
                pose for pose in row.get("poses", [])
                if pose.get("accepted") is True
            ]
    parser = MMCIFParser(QUIET=True)
    sequences, chain_rows = {}, {}
    for component in admission["components"]:
        component_id = component["component_id"]
        path = coordinate_dir / f"{component_id}.cif"
        model = next(iter(parser.get_structure(component_id, str(path))))
        candidates = []
        for chain_id in component["antibody_chains"]:
            sequence = chain_sequence(model[chain_id])
            if sequence:
                key = f"{component_id}:{chain_id}"
                sequences[key] = sequence
                candidates.append(key)
        chain_rows[component_id] = {"path": str(path), "candidate_keys": candidates}
    numbered = h3_records(sequences)
    rows = []
    for component in admission["components"]:
        component_id = component["component_id"]
        candidates = [
            (key, numbered[key])
            for key in chain_rows[component_id]["candidate_keys"]
            if numbered[key].get("ready")
            and numbered[key].get("chain_type") == "H"
        ]
        selected = candidates[0] if candidates else (None, {})
        if poses_by_component:
            accepted = poses_by_component.get(component_id, [])
            pose_paths = [pose["path"] for pose in accepted]
        else:
            pose_paths = [chain_rows[component_id]["path"]]
        rows.append({
            "component_id": component_id,
            "heavy_chain": selected[0].split(":", 1)[1] if selected[0] else None,
            "h3_ready": bool(candidates),
            "h3_sequence": selected[1].get("h3_sequence"),
            "h3_indices_zero_based": selected[1].get("h3_indices_zero_based", []),
            "pose_paths": pose_paths,
            "pose_count": len(pose_paths),
            "multi_pose_ready": len(pose_paths) >= 2,
        })
    h3_count = sum(row["h3_ready"] for row in rows)
    pose_count = sum(row["multi_pose_ready"] for row in rows)
    payload = {
        "schema_version": 1,
        "status": "generation_ready" if h3_count == len(rows) and pose_count == len(rows) else "generation_blocked",
        "classification": "v3_candidate_generation_readiness_audit",
        "admission": str(admission_path),
        "admission_sha256": sha256(admission_path),
        "pose_manifest": str(pose_manifest) if pose_manifest else None,
        "pose_manifest_sha256": sha256(pose_manifest) if pose_manifest else None,
        "component_count": len(rows),
        "h3_ready_components": h3_count,
        "multi_pose_ready_components": pose_count,
        "components": rows,
        "decision": "generate_candidates" if h3_count == len(rows) and pose_count == len(rows) else "acquire_or_generate_frozen_pose_ensembles_before_candidates",
        "claim_boundary": "generation readiness only; no candidate or performance result",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--pose-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.admission, args.coordinates, args.output, args.pose_manifest)
    print(json.dumps({key: value for key, value in result.items() if key != "components"}, indent=2))


if __name__ == "__main__":
    main()
