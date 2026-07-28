#!/usr/bin/env python
"""Audit experimental off-states and apply a candidate contrastive gate."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "modules"))

from ensemble_pose_transfer import ca_coordinates, fixed_paratope_contact_map, kabsch_transform  # noqa: E402
from state_contact_scorer import AA3_TO_1, extract_contact_map, score_sequence_on_contact_map  # noqa: E402


OFF_STATES = {
    "tau": [
        {
            "pdb_id": "5O3L", "chain": "A", "method": "cryo-EM",
            "state": "Alzheimer paired helical filament",
        },
        {
            "pdb_id": "6QJH", "chain": "A", "method": "cryo-EM",
            "state": "heparin-induced 2N4R tau snake filament",
        },
    ],
    "alpha_synuclein": [
        {
            "pdb_id": "1XQ8", "chain": "A", "method": "solution NMR minimized average",
            "state": "micelle-bound average conformation",
        },
        {
            "pdb_id": "2N0A", "chain": "A", "method": "solid-state NMR",
            "state": "pathogenic full-length fibril",
        },
        {
            "pdb_id": "8B9V", "chain": "A", "method": "X-ray",
            "state": "positive antibody complex incorrectly listed as off-state",
            "exclude_reason": "same positive reference complex used by the design pipeline",
        },
    ],
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def chain_sequence(path, chain_id):
    residues = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM  ") or line[21:22].strip() != chain_id:
            continue
        key = (line[22:27], line[17:20])
        if key in seen or line[17:20].strip() not in AA3_TO_1:
            continue
        seen.add(key)
        residues.append(AA3_TO_1[line[17:20].strip()])
    return "".join(residues)


def project_assembly(template, off_path, output, target):
    template_sequence, template_ca = ca_coordinates(template, target["reference_peptide_chain"])
    off_sequence, off_ca = ca_coordinates(off_path, target["off_chain"])
    start = off_sequence.find(template_sequence)
    if start < 0:
        raise ValueError(f"{off_path.stem} lacks exact epitope {template_sequence}")
    rotation, translation, rmsd = kabsch_transform(
        off_ca[start:start + len(template_sequence)], template_ca
    )
    antibody_chains = {target["heavy_chain"], target["light_chain"]}
    antigen_ids = [
        chain for chain in "PQRSTUVWXYZABCDEFGHIJKLMNO" if chain not in antibody_chains
    ]
    source_chains = []
    for line in off_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("ATOM  "):
            chain = line[21:22].strip()
            if chain and chain not in source_chains:
                source_chains.append(chain)
    chain_map = {chain: antigen_ids[index] for index, chain in enumerate(source_chains)}
    lines = []
    serial = 1
    for line in template.read_text(encoding="utf-8").splitlines():
        if line.startswith("ATOM  ") and line[21:22].strip() in antibody_chains:
            lines.append(f"{line[:6]}{serial:5d}{line[11:]}")
            serial += 1
    for line in off_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM  "):
            continue
        source_chain = line[21:22].strip()
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        x, y, z = xyz @ rotation + translation
        rewritten = (
            f"{line[:6]}{serial:5d}{line[11:21]}{chain_map[source_chain]}"
            f"{line[22:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
        )
        lines.append(rewritten)
        serial += 1
    lines.append("END")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="ascii")
    return {
        "epitope_fit_rmsd": rmsd,
        "epitope_start_index": start + 1,
        "primary_antigen_chain": chain_map[target["off_chain"]],
        "antigen_chain_map": chain_map,
    }


def atom_records(path, chains):
    rows = []
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.startswith("ATOM  ") or line[21:22].strip() not in chains:
            continue
        element = line[76:78].strip() or line[12:16].strip()[0]
        if element.upper() == "H":
            continue
        rows.append({
            "chain": line[21:22].strip(),
            "resid": int(line[22:26]),
            "coord": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
        })
    return rows


def projection_geometry(path, antibody_chains, antigen_chains, primary_chain, epitope_resids):
    antibody = atom_records(path, set(antibody_chains))
    antigen = atom_records(path, set(antigen_chains))
    distances = np.linalg.norm(
        np.asarray([row["coord"] for row in antibody])[:, None, :]
        - np.asarray([row["coord"] for row in antigen])[None, :, :], axis=-1,
    )
    epitope = [
        row for row in antigen if row["chain"] == primary_chain and row["resid"] in epitope_resids
    ]
    neighbors = [
        row for row in antigen
        if not (row["chain"] == primary_chain and row["resid"] in epitope_resids)
    ]
    epitope_neighbor_pairs = 0
    if epitope and neighbors:
        neighbor_distances = np.linalg.norm(
            np.asarray([row["coord"] for row in epitope])[:, None, :]
            - np.asarray([row["coord"] for row in neighbors])[None, :, :], axis=-1,
        )
        epitope_neighbor_pairs = int((neighbor_distances < 5.0).sum())
    return {
        "minimum_antibody_antigen_distance": float(distances.min()),
        "severe_clash_pairs_lt_1_5A": int((distances < 1.5).sum()),
        "close_pairs_lt_5A": int((distances < 5.0).sum()),
        "epitope_environment_pairs_lt_5A": epitope_neighbor_pairs,
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    manifest_path = PROJECT_ROOT / "outputs/non_abeta_idp_ensemble_prep_v1/ensemble_manifest.json"
    candidate_path = PROJECT_ROOT / "outputs/non_abeta_idp_ensemble_design_v1/ensemble_designed_candidates.csv"
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    with open(candidate_path, newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))
    targets = {row["target"]: row for row in manifest["targets"]}
    out_dir = PROJECT_ROOT / "outputs/non_abeta_idp_offstate_contrast_v1"
    pose_dir = PROJECT_ROOT / "data/non_abeta_idp/offstate_projected_v1"
    state_rows = []
    maps = defaultdict(list)

    for target_name, entries in OFF_STATES.items():
        target = targets[target_name]
        template = PROJECT_ROOT / target["trimmed_reference_pdb"]
        template_map = extract_contact_map(
            str(template), peptide_chain=target["reference_peptide_chain"]
        )
        for entry in entries:
            base = {
                "target": target_name,
                "pdb_id": entry["pdb_id"],
                "state": entry["state"],
                "method": entry["method"],
            }
            if entry.get("exclude_reason"):
                state_rows.append({
                    **base, "status": "excluded", "reason": entry["exclude_reason"],
                    "exact_epitope": False, "contact_scoring_status": "excluded",
                })
                continue
            pdb_path = PROJECT_ROOT / f"data/non_abeta_idp/{entry['pdb_id']}.pdb"
            sequence = chain_sequence(pdb_path, entry["chain"])
            exact = target["epitope"] in sequence
            if not exact:
                state_rows.append({
                    **base, "pdb": pdb_path.relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": sha256(pdb_path), "exact_epitope": False,
                    "status": "provenance_only", "reason": "exact design epitope absent",
                    "contact_scoring_status": "not_comparable",
                })
                continue
            output = pose_dir / target_name / f"{entry['pdb_id']}_projected_complex.pdb"
            projected = project_assembly(template, pdb_path, output, {
                **target, "off_chain": entry["chain"],
            })
            primary = projected["primary_antigen_chain"]
            primary_residues = []
            primary_sequence = []
            for line in output.read_text(encoding="ascii").splitlines():
                if not line.startswith("ATOM  ") or line[21:22].strip() != primary:
                    continue
                key = int(line[22:26])
                if key not in primary_residues:
                    primary_residues.append(key)
                    primary_sequence.append(AA3_TO_1[line[17:20].strip()])
            sequence_string = "".join(primary_sequence)
            start = sequence_string.find(target["epitope"])
            epitope_resids = set(primary_residues[start:start + len(target["epitope"])])
            geometry = projection_geometry(
                output, [target["heavy_chain"], target["light_chain"]],
                projected["antigen_chain_map"].values(), primary, epitope_resids,
            )
            score_map = fixed_paratope_contact_map(output, template_map, peptide_chain=primary)
            native_score = score_sequence_on_contact_map(
                template_map["paratope_sequence"], score_map
            )["state_contact_score"]
            sterically_blocked = geometry["severe_clash_pairs_lt_1_5A"] > 0
            state_rows.append({
                **base,
                "pdb": pdb_path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256(pdb_path),
                "exact_epitope": True,
                "projected_pose": output.relative_to(PROJECT_ROOT).as_posix(),
                "epitope_fit_rmsd": projected["epitope_fit_rmsd"],
                **geometry,
                "native_projected_score": native_score,
                "status": "geometry_ready",
                "reason": "projected pose is a geometric control, not an experimental negative complex",
                "contact_scoring_status": (
                    "sterically_blocked" if sterically_blocked else "projected_pose_scoreable"
                ),
            })
            maps[target_name].append((entry["pdb_id"], score_map, native_score, sterically_blocked))

    candidate_rows = []
    for candidate in candidates:
        target_maps = maps[candidate["target"]]
        score_rows = []
        for pdb_id, contact_map, native_score, blocked in target_maps:
            score = score_sequence_on_contact_map(candidate["sequence"], contact_map)["state_contact_score"]
            score_rows.append({
                "pdb_id": pdb_id, "score": score, "native_score": native_score,
                "delta_native": score - native_score, "sterically_blocked": blocked,
            })
        comparable = [row for row in score_rows if not row["sterically_blocked"]]
        max_delta = max((row["delta_native"] for row in comparable), default=None)
        contrast_pass = max_delta is not None and max_delta <= 0.0
        candidate_rows.append({
            **candidate,
            "offstate_comparable_count": len(comparable),
            "offstate_blocked_count": sum(row["sterically_blocked"] for row in score_rows),
            "offstate_max_delta_native": max_delta,
            "offstate_scores": json.dumps(score_rows, separators=(",", ":")),
            "offstate_contrast_status": (
                "pass" if contrast_pass else
                "sequence_gate_not_applicable_geometry_blocked" if not comparable else "review"
            ),
        })

    by_target = {}
    for target_name in targets:
        rows = [row for row in candidate_rows if row["target"] == target_name]
        states = [row for row in state_rows if row["target"] == target_name]
        by_target[target_name] = {
            "experimental_states_audited": len(states),
            "exact_epitope_states": sum(bool(row.get("exact_epitope")) for row in states),
            "projected_scoreable_states": sum(
                row.get("contact_scoring_status") == "projected_pose_scoreable" for row in states
            ),
            "sterically_blocked_states": sum(
                row.get("contact_scoring_status") == "sterically_blocked" for row in states
            ),
            "candidate_pass": sum(row["offstate_contrast_status"] == "pass" for row in rows),
            "candidate_geometry_blocked": sum(
                row["offstate_contrast_status"] == "sequence_gate_not_applicable_geometry_blocked"
                for row in rows
            ),
            "candidate_total": len(rows),
        }
    report = {
        "schema_version": "nonabeta.offstate_contrast.v1",
        "status": "pass" if all(
            row["projected_scoreable_states"] >= 1 and row["candidate_pass"] >= 2
            for row in by_target.values()
        ) else "partial",
        "by_target": by_target,
        "state_audit": state_rows,
        "claim_boundary": (
            "Experimental off-state geometries with template-projected antibody poses. "
            "These are geometry and relative-score controls, not experimental non-binding labels."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "offstate_candidate_gate.csv", candidate_rows)
    with open(out_dir / "offstate_audit.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Non-A-beta off-state audit: status={report['status']} {by_target}")


if __name__ == "__main__":
    main()
