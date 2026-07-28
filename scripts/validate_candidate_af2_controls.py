#!/usr/bin/env python3
"""Validate the candidate AF2 backend on experimentally resolved anti-A-beta controls."""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "modules"))

from build_design_variant_dataset import _batch_af2_wsl  # noqa: E402
from idp_antibody_design import _extract_sequence_from_pdb  # noqa: E402
from state_contact_scorer import ABETA42  # noqa: E402


def run_control(name, pdb_path, antibody_chains, peptide, output_dir):
    antibody = ":".join(
        _extract_sequence_from_pdb(pdb_path, chain) for chain in antibody_chains)
    conditions = [("native_fragment", peptide), ("full_abeta42", ABETA42)]
    records = []
    for label, epitope in conditions:
        result = _batch_af2_wsl(
            [antibody], epitope, 3, warmup_seq=antibody,
            output_dir=output_dir / f"{name}_{label}")[0]
        plddt = np.asarray(result.get("plddt_seq", []), dtype=np.float64)
        antibody_length = sum(len(chain) for chain in antibody.split(":"))
        records.append({
            "control": name,
            "condition": label,
            "epitope": epitope,
            "success": bool(result and result.get("success")),
            "iptm": float(result.get("iptm", 0.0)),
            "ptm": float(result.get("ptm", 0.0)),
            "antibody_plddt": float(plddt[:antibody_length].mean()),
            "epitope_plddt": float(plddt[antibody_length:].mean()),
            "interface_pae": float(result.get("interface_pae", 0.0)),
        })
    return records


def main():
    output_dir = PROJECT_ROOT / "results/v5_1_candidates/abeta42_local/af2_controls"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    records.extend(run_control(
        "4HIX", "data/anti_abeta_refs/4HIX.pdb", ["H", "L"], "DAEFRH", output_dir))
    records.extend(run_control(
        "5CSZ", "data/anti_abeta_refs/5CSZ.pdb", ["A", "B"], "DAEFRHDSGY", output_dir))
    native = [record for record in records if record["condition"] == "native_fragment"]
    calibrated = all(
        record["iptm"] >= 0.25 and record["antibody_plddt"] >= 0.60
        and record["interface_pae"] <= 20.0 for record in native)
    report = {
        "backend": "AlphaFold2-Multimer v3, JAX-native single-sequence, 3 recycles",
        "positive_controls": ["4HIX", "5CSZ"],
        "calibrated_for_candidate_gating": calibrated,
        "records": records,
    }
    (output_dir / "control_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
