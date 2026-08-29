#!/usr/bin/env python
"""Materialize same-lineage target donor coordinates for expanded development."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def donor_ids(audit):
    return sorted({
        pdb_id
        for component in audit["components"]
        if component["experimental_pose_source_ready"]
        for pdb_id in component["same_lineage_target_pdb_ids"]
    })


def materialize(audit_path, output_dir, opener=urllib.request.urlopen):
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite donor coordinates: {output_dir}")
    audit = json.loads(audit_path.read_text(encoding="ascii"))
    selected_ids = donor_ids(audit)
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        for pdb_id in selected_ids:
            url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "DisorderFlow-expanded-IDP-donors-v2/1"},
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
        payload = {
            "schema_version": 1,
            "status": "expanded_experimental_pose_donors_materialized",
            "classification": "retrospective_expanded_idp_development_coordinates",
            "pose_source_audit": str(audit_path),
            "pose_source_audit_sha256": sha256(audit_path),
            "donor_coordinate_count": len(records),
            "records": records,
            "future_confirmation_eligible_count": 0,
            "claim_boundary": audit["claim_boundary"],
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="ascii"
        )
        return payload
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
        "--audit", type=Path,
        default=Path(
            "reviewer_outputs/idp_ensemble_expanded_development_v2/"
            "pose_source_audit.json"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(
            "reviewer_outputs/idp_ensemble_expanded_development_v2/"
            "pose_donor_coordinates"
        ),
    )
    args = parser.parse_args()
    result = materialize(ROOT / args.audit, ROOT / args.output_dir)
    print(json.dumps({
        "status": result["status"],
        "donor_coordinates": result["donor_coordinate_count"],
        "future_confirmation_eligible": result[
            "future_confirmation_eligible_count"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
