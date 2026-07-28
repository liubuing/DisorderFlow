#!/usr/bin/env python
"""Generate candidates with explicit positive-versus-sequence-off-state optimization."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "modules"))
sys.path.insert(0, str(ROOT / "scripts/pipeline"))

from audit_nonabeta_sequence_offstates import STATES, state_map  # noqa: E402
from build_contact_guided_mutation_plans import infer_chain_roles, region_label  # noqa: E402
from design_abeta_midregion_ensemble import build_full_chain, ensemble_metrics  # noqa: E402
from ensemble_pose_transfer import fixed_paratope_contact_map  # noqa: E402
from score_shortlist_developability import score_sequence  # noqa: E402
from state_contact_scorer import (  # noqa: E402
    _hotspot_positions, _position_allowed_residues, extract_contact_map,
    score_sequence_on_contact_map,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beam-width", type=int, default=1000)
    parser.add_argument("--max-mutations", type=int, default=4)
    parser.add_argument("--top-per-target", type=int, default=20)
    parser.add_argument("--out", default="outputs/non_abeta_idp_sequence_contrastive_design_v1")
    return parser.parse_args()


def contrast_metrics(sequence, positive_maps, off_maps, native_positive, native_off):
    positive = np.asarray([
        score_sequence_on_contact_map(sequence, contact_map)["state_contact_score"]
        for contact_map in positive_maps
    ])
    off = {
        state_id: float(np.mean([
            score_sequence_on_contact_map(sequence, contact_map)["state_contact_score"]
            for contact_map in maps
        ]))
        for state_id, maps in off_maps.items()
    }
    target_mean = float(positive.mean())
    target_delta = target_mean - float(native_positive.mean())
    off_deltas = {state_id: value - native_off[state_id] for state_id, value in off.items()}
    specificity_gap = target_mean - max(off.values())
    native_gap = float(native_positive.mean()) - max(native_off.values())
    gap_delta = specificity_gap - native_gap
    target_ensemble = ensemble_metrics(sequence, positive_maps, native_positive)
    max_off_delta = max(off_deltas.values())
    objective = (
        target_ensemble["native_delta_p25"]
        + 0.75 * gap_delta
        - 1.5 * max(0.0, max_off_delta)
    )
    return {
        **target_ensemble,
        "target_delta_mean": target_delta,
        "specificity_gap": specificity_gap,
        "native_specificity_gap": native_gap,
        "specificity_gap_delta": gap_delta,
        "max_offstate_delta_native": max_off_delta,
        "offstate_scores": off,
        "offstate_deltas_native": off_deltas,
        "contrastive_objective": objective,
    }


def main():
    args = parse_args()
    manifest = json.load(open(ROOT / "outputs/non_abeta_idp_ensemble_prep_v1/ensemble_manifest.json", encoding="utf-8"))
    pose_audit = json.load(open(ROOT / "outputs/non_abeta_idp_pose_panels_v1/pose_panel_audit.json", encoding="utf-8"))
    selected = []
    plans = []
    summaries = []
    for target in manifest["targets"]:
        target_name = target["target"]
        reference = {
            "pdb": target["reference_id"], "complex_pdb": target["trimmed_reference_pdb"],
            "peptide_chain": target["reference_peptide_chain"],
        }
        template_map = extract_contact_map(
            str(ROOT / reference["complex_pdb"]), peptide_chain=reference["peptide_chain"]
        )
        positive_maps = [
            fixed_paratope_contact_map(ROOT / row["pose_pdb"], template_map)
            for row in pose_audit["entries"]
            if row["target"] == target_name and row.get("selected_for_panel")
        ]
        native = template_map["paratope_sequence"]
        native_positive = np.asarray([
            score_sequence_on_contact_map(native, contact_map)["state_contact_score"]
            for contact_map in positive_maps
        ])
        computational_states = [state for state in STATES[target_name] if state[4] == "computational"]
        off_maps = {
            state_id: [state_map(contact_map, sequence) for contact_map in positive_maps]
            for _, state_id, sequence, _, _ in computational_states
        }
        native_off = {
            state_id: float(np.mean([
                score_sequence_on_contact_map(native, contact_map)["state_contact_score"]
                for contact_map in maps
            ]))
            for state_id, maps in off_maps.items()
        }
        roles = infer_chain_roles(template_map)
        mutable = [
            index for index, residue in enumerate(template_map["paratope_residues"])
            if region_label(roles.get(residue["chain"].upper(), residue["chain"].upper()), residue["resid"])
            != "framework_or_unknown"
        ]
        hotspots = _hotspot_positions(template_map)
        options = {
            index: [aa for aa in _position_allowed_residues(template_map, index, hotspots) if aa != native[index]]
            for index in mutable
        }
        cache = {native: contrast_metrics(native, positive_maps, off_maps, native_positive, native_off)}
        beam = [native]
        all_sequences = {native}
        for _depth in range(1, args.max_mutations + 1):
            expanded = set()
            for parent in beam:
                occupied = {index for index, (old, new) in enumerate(zip(native, parent)) if old != new}
                for index in mutable:
                    if index in occupied:
                        continue
                    for amino_acid in options[index]:
                        expanded.add(parent[:index] + amino_acid + parent[index + 1:])
            for sequence in expanded:
                if sequence not in cache:
                    cache[sequence] = contrast_metrics(sequence, positive_maps, off_maps, native_positive, native_off)
            beam = sorted(
                expanded,
                key=lambda sequence: (
                    cache[sequence]["contrastive_objective"],
                    cache[sequence]["specificity_gap_delta"],
                    -cache[sequence]["max_offstate_delta_native"],
                ), reverse=True,
            )[:args.beam_width]
            all_sequences.update(beam)

        rows = []
        for sequence in all_sequences - {native}:
            metrics = cache[sequence]
            uid = f"{target['reference_id']}_{hashlib.sha256(sequence.encode('ascii')).hexdigest()[:12]}"
            heavy, light, applied, warnings, plan = build_full_chain(reference, template_map, sequence, uid)
            parent_heavy, parent_light, _, _, _ = build_full_chain(reference, template_map, native, "native")
            parent_flags = set(filter(None, (score_sequence(parent_heavy)["flags"] + ";" + score_sequence(parent_light)["flags"]).split(";")))
            candidate_flags = set(filter(None, (score_sequence(heavy)["flags"] + ";" + score_sequence(light)["flags"]).split(";")))
            introduced = candidate_flags - parent_flags
            risk = max(score_sequence(heavy)["sequence_risk"], score_sequence(light)["sequence_risk"])
            n_mutations = sum(a != b for a, b in zip(native, sequence))
            target_floor = -0.005 if target_name == "tau" else 0.0
            passes = (
                metrics["target_delta_mean"] >= 0.0
                and metrics["native_delta_p25"] >= target_floor
                and metrics["specificity_gap_delta"] >= 0.0
                and metrics["max_offstate_delta_native"] <= 0.0
                and risk <= 0.55 and not introduced and not warnings
            )
            rows.append({
                "candidate_uid": uid, "target": target_name, "reference_pdb": target["reference_id"],
                "sequence": sequence, "native_paratope": native, "n_mutations": n_mutations,
                "applied_mutations": ";".join(applied),
                **{key: value for key, value in metrics.items() if key not in {"offstate_scores", "offstate_deltas_native"}},
                "offstate_scores": json.dumps(metrics["offstate_scores"], sort_keys=True, separators=(",", ":")),
                "offstate_deltas_native": json.dumps(metrics["offstate_deltas_native"], sort_keys=True, separators=(",", ":")),
                "full_chain_risk": risk, "introduced_flags": ";".join(sorted(introduced)),
                "heavy_sequence": heavy, "light_sequence": light, "warnings": ";".join(warnings),
                "passes_contrastive_gate": passes,
            })
            if passes:
                plans.extend(plan)
        rows.sort(key=lambda row: (
            row["passes_contrastive_gate"], row["contrastive_objective"],
            row["specificity_gap_delta"], -row["n_mutations"],
        ), reverse=True)
        chosen = [row for row in rows if row["passes_contrastive_gate"]][:args.top_per_target]
        for rank, row in enumerate(chosen, 1):
            row["contrastive_rank"] = rank
        selected.extend(chosen)
        selected_uids = {row["candidate_uid"] for row in chosen}
        plans = [row for row in plans if row["candidate_uid"] in selected_uids or row["candidate_uid"] in {r["candidate_uid"] for r in selected}]
        summaries.append({
            "target": target_name, "evaluated": len(all_sequences) - 1,
            "passing": sum(row["passes_contrastive_gate"] for row in rows), "selected": len(chosen),
            "max_mutations": args.max_mutations, "beam_width": args.beam_width,
        })
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    if selected:
        with open(out / "contrastive_candidates.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selected[0]), extrasaction="ignore")
            writer.writeheader(); writer.writerows(selected)
        with open(out / "contrastive_candidates.fasta", "w", encoding="ascii") as handle:
            for row in selected:
                handle.write(f">{row['candidate_uid']}|{row['target']}|VH\n{row['heavy_sequence']}\n")
                handle.write(f">{row['candidate_uid']}|{row['target']}|VL\n{row['light_sequence']}\n")
    if plans:
        with open(out / "contrastive_mutation_plan.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(plans[0]))
            writer.writeheader(); writer.writerows(plans)
    summary = {
        "schema_version": "nonabeta.sequence_contrastive_design.v1",
        "status": "pass" if all(row["selected"] >= 1 for row in summaries) else "no_hit",
        "pre_registered_gates": {
            "target_delta_mean_min": 0.0, "specificity_gap_delta_min": 0.0,
            "max_offstate_delta_native_max": 0.0,
            "native_delta_p25_min": {"tau": -0.005, "alpha_synuclein": 0.0},
        },
        "by_target": summaries,
        "claim_boundary": "Sequence/topology contrastive design; structural and experimental gates remain required.",
    }
    json.dump(summary, open(out / "contrastive_design_summary.json", "w", encoding="utf-8"), indent=2)
    print(f"Contrastive design: {summary}")


if __name__ == "__main__":
    main()
