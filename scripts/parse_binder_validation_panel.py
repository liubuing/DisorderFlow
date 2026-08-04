#!/usr/bin/env python3
"""Parse multi-seed ColabFold outputs for the 3STB binder validation panel."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def interface_pae(score: dict, chain_lengths: list[int]) -> float | None:
    if len(chain_lengths) != 2 or "pae" not in score:
        return None
    split = chain_lengths[0]
    pae = np.asarray(score["pae"], dtype=np.float64)
    cross = np.concatenate((pae[:split, split:].ravel(), pae[split:, :split].ravel()))
    return float(cross.mean()) if cross.size else None


def residue_contacts(pdb_path: Path, cutoff: float = 5.0) -> int:
    residues: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
    for line in pdb_path.read_text(encoding="ascii", errors="ignore").splitlines():
        if not line.startswith("ATOM"):
            continue
        chain = line[21].strip() or "_"
        residue = line[22:27].strip()
        atom = line[12:16].strip()
        if atom.startswith("H"):
            continue
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        residues.setdefault((chain, residue), []).append(xyz)
    chains = sorted({key[0] for key in residues})
    if len(chains) < 2:
        return 0
    left = [(key, atoms) for key, atoms in residues.items() if key[0] == chains[0]]
    right = [(key, atoms) for key, atoms in residues.items() if key[0] == chains[1]]
    cutoff_sq = cutoff * cutoff
    contacts = 0
    for _, atoms_a in left:
        found = False
        for _, atoms_b in right:
            if any(sum((a[i] - b[i]) ** 2 for i in range(3)) <= cutoff_sq for a in atoms_a for b in atoms_b):
                contacts += 1
                found = True
            if found:
                break
    return contacts


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def parse_arm(directory: Path, name: str, chain_lengths: list[int]) -> dict:
    score_files = sorted(directory.glob(f"{name}_scores_*.json"))
    records = []
    for score_path in score_files:
        score = json.loads(score_path.read_text(encoding="utf-8"))
        pdb_name = score_path.name.replace("_scores_", "_unrelaxed_").replace(".json", ".pdb")
        pdb_path = directory / pdb_name
        record = {
            "score_json": str(score_path.relative_to(ROOT)),
            "mean_plddt": float(np.mean(score["plddt"])),
            "ptm": float(score["ptm"]),
        }
        if "iptm" in score:
            record["iptm"] = float(score["iptm"])
        pae_value = interface_pae(score, chain_lengths)
        if pae_value is not None:
            record["interface_pae"] = pae_value
        if pdb_path.exists() and len(chain_lengths) == 2:
            record["cross_chain_contact_residues"] = residue_contacts(pdb_path)
            record["structure_pdb"] = str(pdb_path.relative_to(ROOT))
        records.append(record)
    output = {"n_predictions": len(records), "records": records}
    for metric in ("mean_plddt", "ptm", "iptm", "interface_pae", "cross_chain_contact_residues"):
        values = [float(record[metric]) for record in records if metric in record and math.isfinite(float(record[metric]))]
        if values:
            output[metric] = summarize(values)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", default="results/binder_validation/3stb_panel_v1")
    args = parser.parse_args()
    panel_dir = ROOT / args.panel_dir
    manifest = json.loads((panel_dir / "manifest.json").read_text(encoding="utf-8"))
    gates = manifest["gates"]
    rows = []
    for source in manifest["records"]:
        name = source["name"]
        monomer = parse_arm(panel_dir / "monomer_results", name, [source["vhh_length"]])
        multimer = parse_arm(panel_dir / "multimer_results", name, source["multimer_chain_lengths"])
        monomer_plddt = monomer.get("mean_plddt", {}).get("mean")
        multimer_iptm = multimer.get("iptm", {}).get("mean")
        multimer_pae = multimer.get("interface_pae", {}).get("mean")
        rows.append({
            **source,
            "monomer": monomer,
            "multimer": multimer,
            "monomer_gate": monomer_plddt is not None and monomer_plddt >= gates["monomer_mean_plddt_min"],
            "multimer_gate": (
                multimer_iptm is not None and multimer_iptm >= gates["multimer_mean_iptm_min"]
                and multimer_pae is not None and multimer_pae <= gates["multimer_mean_interface_pae_max"]
            ),
        })
    by_name = {row["name"]: row for row in rows}
    native = by_name["3STB_native"]["multimer"]
    scrambled = by_name["3STB_native_cdr_scrambled"]["multimer"]
    native_iptm = native.get("iptm", {}).get("mean")
    scrambled_iptm = scrambled.get("iptm", {}).get("mean")
    report = {
        "panel_version": manifest["panel_version"],
        "run_contract": manifest["colabfold_contract"],
        "gates": gates,
        "assessment": {
            "monomer_foldability": "supported" if all(row["monomer_gate"] for row in rows) else "mixed",
            "multimer_absolute_gate": "passed" if any(row["multimer_gate"] for row in rows) else "failed",
            "native_minus_scrambled_mean_iptm": (
                native_iptm - scrambled_iptm
                if native_iptm is not None and scrambled_iptm is not None else None
            ),
            "multimer_use": "weak_rejection_gate_only",
        },
        "records": rows,
    }
    (panel_dir / "panel_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "panel": report["panel_version"],
        "complete_monomer": sum(row["monomer"]["n_predictions"] >= 5 for row in rows),
        "complete_multimer": sum(row["multimer"]["n_predictions"] >= 5 for row in rows),
        "monomer_pass": sum(row["monomer_gate"] for row in rows),
        "multimer_pass": sum(row["multimer_gate"] for row in rows),
    }, indent=2))


if __name__ == "__main__":
    main()
