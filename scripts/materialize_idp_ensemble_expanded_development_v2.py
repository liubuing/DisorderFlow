#!/usr/bin/env python
"""Materialize and audit the balanced expanded-IDP development coordinates."""

# 本脚本为平衡的扩展 IDP 开发面板下载并审计坐标（mmCIF）：
# 从 RCSB 拉取面板中全部 PDB 条目（记录 SHA-256 与字节大小），
# 逐一解析并检查链配置：重链 ≥90 个标准残基、轻链（若有）≥80 个、
# 抗原链至少有可观测残基。任一环节失败会清空输出目录重新抛错；
# future_confirmation_eligible_count 固定为 0，强调仅开发用途。

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    # 计算文件 SHA-256，用于面板来源校验
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# 统计链中带 CA 原子的标准氨基酸残基数
def residue_count(chain):
    return sum("CA" in residue and is_aa(residue, standard=True) for residue in chain)


# 物化主流程：输出目录/审计文件已存在即拒绝 → 校验面板的"隔离标记"
# （future_confirmation_eligible_count 必须为 0）→ 逐 PDB 下载（超时 120 秒）
# → 写坐标清单 → 解析并做逐成分链完整性审计 → 失败时整体回滚目录
def materialize(panel_path, output_dir, audit_path, opener=urllib.request.urlopen):
    if output_dir.exists() or audit_path.exists():
        raise FileExistsError("Refusing to overwrite expanded development coordinates")
    panel = json.loads(panel_path.read_text(encoding="ascii"))
    if panel.get("future_confirmation_eligible_count") != 0:
        raise RuntimeError("Expanded panel isolation marker is missing")
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        for pdb_id in sorted({row["pdb_id"] for row in panel["components"]}):
            url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "DisorderFlow-expanded-IDP-development-v2/1"},
            )
            with opener(request, timeout=120) as response:
                content = response.read()
            path = output_dir / f"{pdb_id.upper()}.cif"
            path.write_bytes(content)
            records.append({
                "pdb_id": pdb_id,
                "path": str(path),
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
                "source_url": url,
            })
        manifest = {
            "schema_version": 1,
            "status": "expanded_development_coordinates_materialized",
            "classification": "retrospective_expanded_idp_development_coordinates",
            "panel": str(panel_path),
            "panel_sha256": sha256(panel_path),
            "coordinate_count": len(records),
            "records": records,
            "future_confirmation_eligible_count": 0,
            "claim_boundary": panel["claim_boundary"],
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="ascii"
        )

        parser = MMCIFParser(QUIET=True)
        coordinate_paths = {
            row["pdb_id"]: Path(row["path"]) for row in records
        }
        component_rows = []
        for component in panel["components"]:
            model = next(iter(parser.get_structure(
                component["component_id"],
                str(coordinate_paths[component["pdb_id"]]),
            )))
            available = {chain.id: chain for chain in model}
            heavy = component["heavy_chain"]
            light = component["light_chain"]
            antigens = component["antigen_chains"]
            missing = [
                chain for chain in [heavy, light, *antigens]
                if chain and chain not in available
            ]
            heavy_count = residue_count(available[heavy]) if heavy in available else 0
            light_count = residue_count(available[light]) if light in available else None
            antigen_counts = {
                chain: residue_count(available[chain]) if chain in available else 0
                for chain in antigens
            }
            checks = {
                "configured_chains_present": not missing,
                "heavy_chain_minimum_90_residues": heavy_count >= 90,
                "light_chain_minimum_80_or_absent": (
                    light is None or (light_count or 0) >= 80
                ),
                "antigen_has_observed_residues": any(
                    count > 0 for count in antigen_counts.values()
                ),
            }
            component_rows.append({
                "component_id": component["component_id"],
                "pdb_id": component["pdb_id"],
                "target": component["target"],
                "lineage_proxy": component["lineage_proxy"],
                "missing_configured_chains": missing,
                "heavy_residue_count": heavy_count,
                "light_residue_count": light_count,
                "antigen_residue_counts": antigen_counts,
                "checks": checks,
                "coordinate_ready": all(checks.values()),
            })
        audit = {
            "schema_version": 1,
            "status": "expanded_coordinate_audit_complete",
            "classification": "retrospective_expanded_idp_development_audit",
            "component_count": len(component_rows),
            "coordinate_ready_count": sum(
                row["coordinate_ready"] for row in component_rows
            ),
            "blocked_count": sum(
                not row["coordinate_ready"] for row in component_rows
            ),
            "components": component_rows,
            "future_confirmation_eligible_count": 0,
            "decision": "use_coordinate_ready_components_for_development_only",
            "claim_boundary": panel["claim_boundary"],
        }
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="ascii")
        return manifest, audit
    except Exception:
        for path in sorted(output_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        output_dir.rmdir()
        raise


def main():
    # 主入口：面板/坐标目录/审计路径均有默认值，完成后打印就绪与阻断计数
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel", type=Path,
        default=Path(
            "reviewer_outputs/idp_ensemble_expanded_development_v2/"
            "balanced_panel.json"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(
            "reviewer_outputs/idp_ensemble_expanded_development_v2/coordinates"
        ),
    )
    parser.add_argument(
        "--audit", type=Path,
        default=Path(
            "reviewer_outputs/idp_ensemble_expanded_development_v2/"
            "coordinate_audit.json"
        ),
    )
    args = parser.parse_args()
    manifest, audit = materialize(
        ROOT / args.panel, ROOT / args.output_dir, ROOT / args.audit
    )
    print(json.dumps({
        "status": manifest["status"],
        "coordinates": manifest["coordinate_count"],
        "components": audit["component_count"],
        "coordinate_ready": audit["coordinate_ready_count"],
        "blocked": audit["blocked_count"],
        "future_confirmation_eligible": audit[
            "future_confirmation_eligible_count"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
