#!/usr/bin/env python
"""Materialize and audit the balanced expanded-IDP development coordinates."""

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
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def residue_count(chain):
    return sum("CA" in residue and is_aa(residue, standard=True) for residue in chain)


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
