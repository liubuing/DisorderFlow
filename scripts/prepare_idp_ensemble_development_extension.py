#!/usr/bin/env python
"""Prepare and audit the historical-structure development-extension pose panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml
from Bio.PDB import PDBIO, PDBParser, Select
from Bio.SeqUtils import seq1
from openmm.app import PDBFile
from pdbfixer import PDBFixer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "modules"))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_idp_ensemble_expansion import (  # noqa: E402
    audit_pose,
    generate_peptide_only_ensemble,
    prepare_source_conformers,
    transfer_pose,
    write_epitope_model,
)

from modules.ensemble_pose_transfer import (  # noqa: E402
    fixed_paratope_contact_map,
    pose_geometry_audit,
    refine_pose_local,
    resolve_pose_clashes,
)
from modules.state_contact_scorer import extract_contact_map  # noqa: E402

# 修饰残基到标准氨基酸的映射（硒代蛋氨酸 MSE->M、焦谷氨酸 PCA->E、磷酸化 S/T/Y）
MODIFIED_RESIDUES = {"MSE": "M", "PCA": "E", "SEP": "S", "TPO": "T", "PTR": "Y"}
# 修饰残基重命名为标准三字母残基名
PTM_RENAME = {"SEP": "SER", "TPO": "THR", "PTR": "TYR", "MSE": "MET"}
# 重命名后各标准残基允许保留的原子集合（不在集合内的原子行被剔除）
STANDARD_ATOMS = {
    "SER": {"N", "CA", "C", "O", "CB", "OG"},
    "THR": {"N", "CA", "C", "O", "CB", "OG1", "CG2"},
    "TYR": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"},
    "MET": {"N", "CA", "C", "O", "CB", "CG", "SD", "CE"},
}
# MSE 的硒原子重命名为甲硫氨酸的硫原子
MSE_ATOM_RENAME = {"SE": "SD"}


def sha256(path):
    # 文件 SHA256，用于冻结清单与产物血缘校验
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sequence_residues(chain):
    # 提取链上含 CA 的标准残基，返回 (序列, 残基列表)；修饰残基先归一化
    residues = []
    sequence = []
    for residue in chain:
        if "CA" not in residue:
            continue
        aa = MODIFIED_RESIDUES.get(
            residue.resname, seq1(residue.resname, custom_map={"MSE": "M"})
        )
        if aa in "ACDEFGHIKLMNPQRSTVWY":
            residues.append(residue)
            sequence.append(aa)
    return "".join(sequence), residues


class ReferenceSelect(Select):
    """Bio.PDB 选择器：仅保留抗体链与抗原链上的表位残基。"""

    def __init__(self, antibody_chains, antigen_chain, antigen_residue_ids):
        self.antibody_chains = set(antibody_chains)
        self.antigen_chain = antigen_chain
        self.antigen_residue_ids = set(antigen_residue_ids)

    def accept_chain(self, chain):
        return chain.id in self.antibody_chains or chain.id == self.antigen_chain

    def accept_residue(self, residue):
        return residue.get_parent().id in self.antibody_chains or (
            residue.get_parent().id == self.antigen_chain
            and residue.id in self.antigen_residue_ids
        )


class EpitopeSelect(Select):
    """Bio.PDB 选择器：仅保留指定链上的表位残基（用于写表位模型）。"""

    def __init__(self, chain_id, residue_ids):
        self.chain_id = chain_id
        self.residue_ids = set(residue_ids)

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        return residue.id in self.residue_ids


def rename_ptm_residues(path, chain_id, residue_ids, mapping):
    # 就地重命名表位残基（如磷酸化 S/T/Y -> SER/THR/TYR），并剔除非标准原子行
    lines = Path(path).read_text(encoding="ascii").splitlines(keepends=True)
    kept = []
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            if line[21] == chain_id:
                resid = (int(line[22:26]), line[26:27].strip())
                if resid in residue_ids:
                    name = line[17:20].strip()
                    if name in mapping:
                        replacement = mapping[name].ljust(3)
                        line = f"{line[:17]}{replacement}{line[20:]}"
                        atom_name = line[12:16].strip()
                        if atom_name not in STANDARD_ATOMS.get(mapping[name], set()):
                            continue
        kept.append(line)
    Path(path).write_text("".join(kept), encoding="ascii")


def normalize_mse_residues(path):
    # 将 MSE/PCA 残基名统一改写为 MET/GLU
    lines = Path(path).read_text(encoding="ascii").splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith(("ATOM  ", "HETATM")):
            resname = line[17:20].strip()
            if resname == "MSE":
                lines[index] = f"{line[:17]}MET{line[20:]}"
            elif resname == "PCA":
                lines[index] = f"{line[:17]}GLU{line[20:]}"
    Path(path).write_text("".join(lines), encoding="ascii")


def parse_ptm_mapping(spec):
    # 将配置中的 PTM 归一化描述（dict/字符串/二元组列表）统一解析为映射
    if not spec:
        return {}
    if isinstance(spec, dict):
        return dict(spec)
    mapping = {}
    for item in spec:
        if isinstance(item, dict):
            mapping.update(item)
        elif isinstance(item, str) and ":" in item:
            key, value = item.split(":", 1)
            mapping[key.strip()] = value.strip()
        elif len(item) == 2:
            mapping[item[0]] = item[1]
        else:
            raise ValueError(f"Unsupported PTM normalization spec: {item!r}")
    return mapping


def trim_reference(component, output_path):
    """裁剪参考结构：仅保留抗体链 + 抗原链表位残基，并做修饰残基归一化。"""
    source = ROOT / component["reference_pdb"]
    parser = PDBParser(QUIET=True)
    model = next(iter(parser.get_structure(component["component_id"], str(source))))
    sequence, residues = sequence_residues(model[component["antigen_chain"]])
    start = sequence.find(component["epitope"])
    if start < 0:
        raise ValueError(
            f"{component['component_id']} lacks observed epitope {component['epitope']}"
        )
    epitope_residues = residues[start:start + len(component["epitope"])]
    antibody_chains = [component["heavy_chain"]]
    if component.get("light_chain"):
        antibody_chains.append(component["light_chain"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(model)
    io.save(str(output_path), ReferenceSelect(
        antibody_chains,
        component["antigen_chain"],
        [residue.id for residue in epitope_residues],
    ))
    normalize_mse_residues(output_path)
    mapping = parse_ptm_mapping(component.get("ptm_normalization", []))
    rename_ptm_residues(
        output_path,
        component["antigen_chain"],
        {(residue.id[1], residue.id[2].strip()) for residue in epitope_residues},
        mapping,
    )
    return {
        "path": output_path.relative_to(ROOT).as_posix(),
        "sha256": sha256(output_path),
        "antibody_chains": antibody_chains,
        "epitope_residue_ids": [
            [residue.id[0], residue.id[1], residue.id[2].strip()]
            for residue in epitope_residues
        ],
    }


def prepare_component(component, config, output_root):
    """单组件预处理：裁剪参考 -> 生成/转移 pose -> 逐 pose 审计 -> 按上限挑选合格 pose。"""
    component_dir = output_root / component["component_id"]
    trimmed = trim_reference(component, component_dir / "trimmed_reference.pdb")
    template = ROOT / trimmed["path"]
    template_map = extract_contact_map(str(template), peptide_chain=component["antigen_chain"])
    if template_map["peptide_sequence"] != component["epitope"]:
        raise ValueError(f"{component['component_id']} template epitope mismatch")
    poses = []
    source_rows = []
    if component["ensemble"]["type"] == "synthetic_local_openmm":
        import random

        random.seed(20260811)  # 固定采样种子，保证合成系综可复现
        from generate_h3_peptide_ensemble import peptide_geometry
        from openmm.app import PDBFile

        reference_topology, reference_positions = (
            PDBFile(str(template)).getTopology(),
            PDBFile(str(template)).getPositions(),
        )
        reference_geometry = peptide_geometry(
            reference_topology, reference_positions, component["antigen_chain"]
        )
        settings = {
            **config["sampling"],
            "quality_control": {
                **config["sampling"]["quality_control"],
                "max_ca_geometry_outliers": int(
                    config["sampling"]["quality_control"]["max_ca_geometry_outliers"]
                ) + int(reference_geometry["ca_geometry_outliers"]),
                "max_cn_geometry_outliers": int(
                    config["sampling"]["quality_control"]["max_cn_geometry_outliers"]
                ) + int(reference_geometry["cn_geometry_outliers"]),
            },
        }
        if config["sampling"].get("platform") == "CPU":
            from openmm import Platform

            Platform.getPlatformByName("CPU").setPropertyDefaultValue(
                "Threads", str(int(config["sampling"].get("cpu_threads", 1)))
            )
        component_openmm = {
            **component,
            "ensemble": {**component["ensemble"], "sampling": settings},
        }
        from openmm import Platform as OpenMMPlatform

        ensemble = generate_peptide_only_ensemble(
            template,
            component_openmm,
            component_dir,
            hydrogens_platform=OpenMMPlatform.getPlatformByName("Reference"),
        )
        accepted = [row for row in ensemble["conformers"] if row["status"] == "accepted"]
        for rank, row in enumerate(accepted):
            pose = component_dir / "poses" / f"pose_{rank}.pdb"
            transfer = transfer_pose(
                template,
                row["pdb"],
                pose,
                component["antigen_chain"],
                component["antigen_chain"],
                component["heavy_chain"],
                component.get("light_chain") or "",
                1,
            )
            source_rows.append({
                "source_model": row["seed"],
                "source_pdb": str(row["pdb"]),
                "source_sha256": row["sha256"],
            })
            poses.append(audit_pose(
                component, pose, template_map, config["requirements"], transfer=transfer
            ))
    else:
        conformers = prepare_source_conformers(
            component, component_dir / "source_conformers"
        )
        for rank, (model_index, source, _) in enumerate(conformers):
            pose = component_dir / "poses" / f"pose_{rank}.pdb"
            transfer = transfer_pose(
                template,
                source,
                pose,
                component["antigen_chain"],
                component["ensemble"]["chain"],
                component["heavy_chain"],
                component.get("light_chain") or "",
                1,
            )
            source_rows.append({
                "source_model": model_index,
                "source_pdb": source.relative_to(ROOT).as_posix(),
                "source_sha256": sha256(source),
            })
            poses.append(audit_pose(
                component, pose, template_map, config["requirements"], transfer=transfer
            ))

    cap = int(config["requirements"]["selected_pose_conformers_per_component"])
    passing = [row for row in poses if row["status"] == "pass"]
    # sensitivity-only 组件允许通过数即所需数；正式组件要求达到完整上限
    if bool(component.get("sensitivity_only", False)):
        required = max(1, min(cap, len(passing)))
    else:
        required = cap
    selected = passing[:cap]
    selected_paths = {row["pose_pdb"] for row in selected}
    for row in poses:
        row["selected_for_panel"] = row["pose_pdb"] in selected_paths
    return {
        **component,
        "trimmed_reference_pdb": trimmed["path"],
        "trimmed_reference_sha256": trimmed["sha256"],
        "ensemble_provenance": component["ensemble"]["provenance"],
        "source_conformers": source_rows,
        "poses": poses,
        "passing_poses": len(passing),
        "selected_poses": len(selected),
        "required_poses": required,
        "status": "pass" if len(selected) == required else "fail",
    }


def main():
    # 主流程：校验冻结清单血缘 -> 逐组件准备 pose -> 增量写准备报告 -> 通过后冻结 pose manifest
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/benchmarks/idp_ensemble_development_extension_prep_v1.yml",
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = ROOT / config["component_manifest"]
    if sha256(manifest_path) != config["component_manifest_sha256"]:
        raise ValueError("Frozen component manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_by_id = {row["component_id"]: row for row in manifest["components"]}
    for component in config["components"]:
        frozen = manifest_by_id[component["component_id"]]
        reference = ROOT / component["reference_pdb"]
        if sha256(reference) != component["reference_sha256"]:
            raise ValueError(f"Reference hash mismatch: {component['component_id']}")
        if component["h3_positions_1_indexed"] != frozen["h3_positions_1_indexed"]:
            raise ValueError("H3 position drift vs frozen manifest")
        if component["native_h3"] != frozen["native_h3"]:
            raise ValueError("Native H3 drift vs frozen manifest")
        for key in ("heavy_chain", "light_chain", "antigen_chain", "epitope"):
            if component[key] != frozen[key]:
                raise ValueError(f"{key} drift vs frozen manifest")
        if frozen.get("ptm_normalization") and not component.get("ptm_normalization"):
            raise ValueError("PTM normalization missing for frozen component")
        if bool(frozen.get("sensitivity_only")) != bool(component.get("sensitivity_only", False)):
            raise ValueError("Sensitivity-only flag drift vs frozen manifest")
    for component in config["components"]:
        if component["ensemble"]["type"] == "pdb_files":
            for relative in component["ensemble"]["paths"]:
                if not (ROOT / relative).exists():
                    raise ValueError(f"Missing source conformer: {relative}")
    output_path = ROOT / config["output"]["preparation"]
    output_root = output_path.parent / "work"
    rows = []
    for component in config["components"]:
        row = prepare_component(component, config, output_root)
        rows.append(row)
        output = {
            "schema_version": 1,
            "status": "pass" if all(item["status"] == "pass" for item in rows)
            and len(rows) == len(config["components"]) else "partial",
            "classification": config["classification"],
            "config": args.config,
            "config_sha256": sha256(config_path),
            "components": rows,
            "claim_boundary": config["claim_boundary"],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
        print(
            f"Extension preparation {component['component_id']}: "
            f"status={row['status']} selected={row['selected_poses']}",
            flush=True,
        )
    if output["status"] == "pass":
        pose_manifest_path = ROOT / config["output"]["component_manifest"]
        pose_manifest = {
            "schema_version": 1,
            "status": "frozen_after_pose_gate_before_candidate_generation",
            "classification": config["classification"],
            "preparation": output_path.relative_to(ROOT).as_posix(),
            "preparation_sha256": sha256(output_path),
            "components": [
                {
                    "component_id": row["component_id"],
                    "target": manifest_by_id[row["component_id"]]["target"],
                    "disease": manifest_by_id[row["component_id"]]["disease"],
                    "antibody": manifest_by_id[row["component_id"]]["antibody"],
                    "reference_path": row["trimmed_reference_pdb"],
                    "reference_sha256": row["trimmed_reference_sha256"],
                    "heavy_chain": row["heavy_chain"],
                    "light_chain": row.get("light_chain"),
                    "antigen_chain": row["antigen_chain"],
                    "epitope": row["epitope"],
                    "h3_positions_1_indexed": row["h3_positions_1_indexed"],
                    "native_h3": row["native_h3"],
                    "ensemble_provenance": row["ensemble_provenance"],
                    "sensitivity_only": bool(
                        manifest_by_id[row["component_id"]].get("sensitivity_only", False)
                    ),
                    "ptm_normalization": manifest_by_id[row["component_id"]].get(
                        "ptm_normalization"
                    ),
                    "pose_limitation": manifest_by_id[row["component_id"]].get(
                        "pose_limitation"
                    ),
                    "pose_paths": [
                        pose["pose_pdb"] for pose in row["poses"]
                        if pose["selected_for_panel"]
                    ],
                }
                for row in rows
            ],
            "claim_boundary": config["claim_boundary"],
        }
        pose_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        pose_manifest_path.write_text(json.dumps(pose_manifest, indent=2) + "\n", encoding="ascii")
    if output["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
