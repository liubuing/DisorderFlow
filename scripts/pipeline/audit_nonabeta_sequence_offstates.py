#!/usr/bin/env python
"""Pre-registered topology/sequence off-state screen for non-A-beta IDP candidates."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "modules"))
RETRIEVAL_DATE = "2026-07-28"

from ensemble_pose_transfer import fixed_paratope_contact_map  # noqa: E402
from state_contact_scorer import extract_contact_map, score_sequence_on_contact_map  # noqa: E402


SEQUENCE_SOURCES = {
    "MAPT": ("P10636", "KHVPGGGSV"),
    "MAP2": ("P11137", "HHVPGGGNV"),
    "MAP4": ("P27816", "KHVPGGGNV"),
    "SNCA": ("P37840", "EGILEDMPVD"),
    "SNCB": ("Q16143", "EGESYEDPPQ"),
    "SNCG": ("O76070", "EASKEKEEVA"),
}

STATES = {
    "tau": [
        ("target", "MAPT_target", "KHVPGGGSV", "P10636", "target"),
        ("adjacent_repeat", "MAPT_repeat_R1", "KHQPGGGKV", "P10636", "computational"),
        ("adjacent_repeat", "MAPT_repeat_R3", "HHKPGGGQV", "P10636", "computational"),
        ("adjacent_repeat", "MAPT_repeat_R4", "THVPGGGNK", "P10636", "computational"),
        ("homolog", "MAP2_repeat", "HHVPGGGNV", "P11137", "computational"),
        ("homolog", "MAP4_repeat", "KHVPGGGNV", "P27816", "computational"),
        ("scrambled", "composition_scramble", "GVPGSKVGH", "derived", "computational"),
        ("ptm", "tau_PTM_panel", "", "experimental_plan", "experimental_only"),
    ],
    "alpha_synuclein": [
        ("target", "SNCA_target", "EGILEDMPVD", "P37840", "target"),
        ("adjacent", "SNCA_left_shift", "QEGILEDMPV", "P37840", "computational"),
        ("adjacent", "SNCA_right_shift", "GILEDMPVDP", "P37840", "computational"),
        ("adjacent", "SNCA_downstream", "DMPVDPDNEA", "P37840", "computational"),
        ("homolog", "SNCB_best_local", "EGESYEDPPQ", "Q16143", "computational"),
        ("homolog", "SNCG_best_local", "EASKEKEEVA", "O76070", "computational"),
        ("scrambled", "composition_scramble", "DMPVEGILDE", "derived", "computational"),
        ("ptm", "SNCA_PTM_panel", "", "experimental_plan", "experimental_only"),
    ],
}


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def state_map(contact_map, sequence):
    if len(sequence) != len(contact_map["peptide_sequence"]):
        raise ValueError(f"State length {len(sequence)} != template peptide {len(contact_map['peptide_sequence'])}")
    return {
        **contact_map,
        "peptide_sequence": sequence,
        "contacts": [replace(contact, epitope_aa=sequence[contact.epitope_index]) for contact in contact_map["contacts"]],
    }


def main():
    manifest = json.load(open(ROOT / "outputs/non_abeta_idp_ensemble_prep_v1/ensemble_manifest.json", encoding="utf-8"))
    audit = json.load(open(ROOT / "outputs/non_abeta_idp_pose_panels_v1/pose_panel_audit.json", encoding="utf-8"))
    candidate_files = [
        ROOT / "outputs/non_abeta_idp_ensemble_design_v1/ensemble_designed_candidates.csv",
        ROOT / "outputs/non_abeta_idp_ensemble_redesign_v2/ensemble_designed_candidates.csv",
    ]
    candidates = {}
    for path in candidate_files:
        for row in load_csv(path):
            candidates[row["candidate_uid"]] = {**row, "candidate_source": path.parent.name}
    target_cfg = {row["target"]: row for row in manifest["targets"]}
    state_rows = []
    score_rows = []
    gates = {
        "min_target_delta_mean": 0.0,
        "min_specificity_gap_delta": 0.0,
        "max_offstate_delta_vs_native": 0.0,
        "required_computational_offstates": 2,
        "semantics": "Pre-registered topology/sequence screen; not structural or experimental specificity.",
    }
    for target_name, states in STATES.items():
        cfg = target_cfg[target_name]
        template_map = extract_contact_map(
            str(ROOT / cfg["trimmed_reference_pdb"]), peptide_chain=cfg["reference_peptide_chain"]
        )
        pose_maps = [
            fixed_paratope_contact_map(ROOT / row["pose_pdb"], template_map)
            for row in audit["entries"]
            if row["target"] == target_name and row.get("selected_for_panel")
        ]
        native = template_map["paratope_sequence"]
        computational_states = [state for state in states if state[4] == "computational"]
        for kind, state_id, sequence, accession, mode in states:
            if kind == "homolog":
                alignment_method = (
                    "ungapped sliding-window positional identity against the design epitope; "
                    "highest-identity local window retained"
                )
            elif kind in {"adjacent", "adjacent_repeat", "target"}:
                alignment_method = "literal window selection from the accession sequence"
            elif kind == "scrambled":
                alignment_method = "deterministic composition-matched permutation of the design epitope"
            else:
                alignment_method = "not_applicable_experimental_definition"
            state_rows.append({
                "target": target_name, "state_id": state_id, "state_kind": kind,
                "sequence": sequence, "sequence_length": len(sequence), "source_accession": accession,
                "retrieval_date": RETRIEVAL_DATE if accession not in {"derived", "experimental_plan"} else "",
                "source_window": sequence,
                "alignment_method": alignment_method,
                "evaluation_mode": mode,
                "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest() if sequence else "",
                "claim_boundary": (
                    "same positive-state contact topology with substituted sequence"
                    if mode == "computational" else
                    "requires explicit modified antigen preparation and experiment" if mode == "experimental_only"
                    else "positive target"
                ),
            })
        native_target = np.mean([
            score_sequence_on_contact_map(native, contact_map)["state_contact_score"] for contact_map in pose_maps
        ])
        native_off = {
            state_id: float(np.mean([
                score_sequence_on_contact_map(native, state_map(contact_map, sequence))["state_contact_score"]
                for contact_map in pose_maps
            ]))
            for _, state_id, sequence, _, _ in computational_states
        }
        native_gap = native_target - max(native_off.values())
        target_candidates = [row for row in candidates.values() if row["target"] == target_name]
        for candidate in target_candidates:
            target_values = [
                score_sequence_on_contact_map(candidate["sequence"], contact_map)["state_contact_score"]
                for contact_map in pose_maps
            ]
            off_scores = {}
            off_deltas = {}
            for _, state_id, sequence, _, _ in computational_states:
                values = [
                    score_sequence_on_contact_map(candidate["sequence"], state_map(contact_map, sequence))["state_contact_score"]
                    for contact_map in pose_maps
                ]
                off_scores[state_id] = float(np.mean(values))
                off_deltas[state_id] = off_scores[state_id] - native_off[state_id]
            target_mean = float(np.mean(target_values))
            target_delta = target_mean - native_target
            gap = target_mean - max(off_scores.values())
            gap_delta = gap - native_gap
            passes = (
                target_delta >= gates["min_target_delta_mean"]
                and gap_delta >= gates["min_specificity_gap_delta"]
                and max(off_deltas.values()) <= gates["max_offstate_delta_vs_native"]
                and len(off_scores) >= gates["required_computational_offstates"]
            )
            score_rows.append({
                "candidate_uid": candidate["candidate_uid"], "target": target_name,
                "candidate_source": candidate["candidate_source"],
                "n_mutations": candidate["n_mutations"], "applied_mutations": candidate["applied_mutations"],
                "target_score_mean": round(target_mean, 5), "target_delta_native": round(target_delta, 5),
                "specificity_gap": round(gap, 5), "native_specificity_gap": round(native_gap, 5),
                "specificity_gap_delta": round(gap_delta, 5),
                "max_offstate_delta_native": round(max(off_deltas.values()), 5),
                "offstate_scores": json.dumps(off_scores, sort_keys=True, separators=(",", ":")),
                "offstate_deltas_native": json.dumps(off_deltas, sort_keys=True, separators=(",", ":")),
                "topology_sequence_status": "pass" if passes else "review",
                "heavy_sequence": candidate["heavy_sequence"], "light_sequence": candidate["light_sequence"],
            })
    out = ROOT / "outputs/non_abeta_idp_sequence_offstate_v1"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("sequence_state_panel.csv", state_rows), ("candidate_sequence_contrast.csv", score_rows)):
        with open(out / name, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    by_target = {
        target: {
            "candidates": sum(row["target"] == target for row in score_rows),
            "passing": sum(row["target"] == target and row["topology_sequence_status"] == "pass" for row in score_rows),
            "computational_states": sum(row["target"] == target and row["evaluation_mode"] == "computational" for row in state_rows),
            "experimental_only_states": sum(row["target"] == target and row["evaluation_mode"] == "experimental_only" for row in state_rows),
        }
        for target in STATES
    }
    summary = {
        "schema_version": "nonabeta.sequence_offstate.v1", "status": "complete",
        "gates": gates, "by_target": by_target,
        "sequence_sources": {key: {"accession": value[0], "reference_window": value[1]} for key, value in SEQUENCE_SOURCES.items()},
        "retrieval_date": RETRIEVAL_DATE,
        "source_policy": (
            "UniProt accession sequence; exact window and selection method are recorded per state. "
            "Coordinates are not asserted where isoform numbering is ambiguous."
        ),
        "claim_boundary": (
            "Candidate-specific paratope sequences were scored against substituted off-state sequences on the "
            "positive contact topology. Pass is a triage result, not a structural or experimental specificity claim."
        ),
    }
    json.dump(summary, open(out / "sequence_offstate_summary.json", "w", encoding="utf-8"), indent=2)
    print(f"Sequence off-state screen: {by_target}")


if __name__ == "__main__":
    main()
