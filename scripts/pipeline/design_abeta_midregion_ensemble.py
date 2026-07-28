#!/usr/bin/env python
"""Direct ensemble-conditioned design for A-beta middle-epitope antibodies."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "modules"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "pipeline"))

from build_contact_guided_mutation_plans import infer_chain_roles, region_label  # noqa: E402
from ensemble_pose_transfer import fixed_paratope_contact_map  # noqa: E402
from export_full_chain_constructs import apply_candidate_mutations, contact_chain_roles, pdb_chain_sequences  # noqa: E402
from score_shortlist_developability import score_sequence  # noqa: E402
from state_contact_scorer import (  # noqa: E402
    _hotspot_positions,
    _position_allowed_residues,
    extract_contact_map,
    score_sequence_on_contact_map,
)


TARGETS = ("4XXD", "5VZY")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/statecontrast/abeta_position_effect_v3_4ref_loro.yml")
    parser.add_argument("--pose-audit", default="outputs/abeta_multiconf_pose_panel_v1/pose_panel_audit.json")
    parser.add_argument("--out", default="outputs/abeta_midregion_ensemble_design_v1")
    parser.add_argument("--beam-width", type=int, default=250)
    parser.add_argument("--max-mutations", type=int, default=4)
    parser.add_argument("--top-per-reference", type=int, default=20)
    parser.add_argument("--max-full-chain-risk", type=float, default=0.55)
    return parser.parse_args()


def ensemble_metrics(sequence, maps, native_scores):
    values = np.asarray([
        score_sequence_on_contact_map(sequence, contact_map)["state_contact_score"]
        for contact_map in maps
    ])
    deltas = values - native_scores
    robust = (
        0.45 * np.percentile(values, 25)
        + 0.25 * values.min()
        + 0.20 * values.mean()
        - 0.10 * values.std()
    )
    return {
        "ensemble_min": float(values.min()),
        "ensemble_p25": float(np.percentile(values, 25)),
        "ensemble_mean": float(values.mean()),
        "ensemble_std": float(values.std()),
        "native_delta_min": float(deltas.min()),
        "native_delta_p25": float(np.percentile(deltas, 25)),
        "native_delta_mean": float(deltas.mean()),
        "multiconf_robust_score": float(robust),
        "per_conformer_scores": values.tolist(),
    }


def mutation_rows(template_map, roles, native, sequence, reference_id, candidate_uid):
    rows = []
    for index, (old, new) in enumerate(zip(native, sequence)):
        if old == new:
            continue
        residue = template_map["paratope_residues"][index]
        role = roles.get(residue["chain"].upper(), residue["chain"].upper())
        rows.append({
            "reference_pdb": reference_id,
            "candidate_uid": candidate_uid,
            "paratope_index": index + 1,
            "chain": residue["chain"],
            "chain_role": role,
            "resid": residue["resid"],
            "region_label": region_label(role, residue["resid"]),
            "native_aa": old,
            "mutant_aa": new,
            "mutation": f"{residue['chain']}:{residue['resid']}:{old}>{new}",
        })
    return rows


def build_full_chain(reference, template_map, sequence, candidate_uid):
    roles = infer_chain_roles(template_map)
    chain_res = pdb_chain_sequences(reference["complex_pdb"])
    plan = mutation_rows(
        template_map, roles, template_map["paratope_sequence"], sequence,
        reference["pdb"], candidate_uid,
    )
    mutated, applied, warnings = apply_candidate_mutations(chain_res, plan)
    role_to_chain = contact_chain_roles(template_map, roles)
    heavy_chain = role_to_chain.get("H")
    light_chain = role_to_chain.get("L")
    heavy = mutated.get(heavy_chain, "")
    light = mutated.get(light_chain, "")
    return heavy, light, applied, warnings, plan


def design_reference(reference, pose_rows, args):
    reference_id = reference["pdb"]
    template_map = extract_contact_map(
        str(PROJECT_ROOT / reference["complex_pdb"]), peptide_chain=reference["peptide_chain"]
    )
    maps = [
        fixed_paratope_contact_map(PROJECT_ROOT / row["pose_pdb"], template_map)
        for row in pose_rows if row["reference_pdb"] == reference_id and row["status"] == "pass"
    ]
    if len(maps) != 5:
        raise ValueError(f"{reference_id} requires five passing poses, found {len(maps)}")
    native = template_map["paratope_sequence"]
    native_scores = np.asarray([
        score_sequence_on_contact_map(native, contact_map)["state_contact_score"]
        for contact_map in maps
    ])
    roles = infer_chain_roles(template_map)
    mutable = [
        index for index, residue in enumerate(template_map["paratope_residues"])
        if region_label(
            roles.get(residue["chain"].upper(), residue["chain"].upper()), residue["resid"]
        ) != "framework_or_unknown"
    ]
    hotspots = _hotspot_positions(template_map)
    options = {
        index: [aa for aa in _position_allowed_residues(template_map, index, hotspots) if aa != native[index]]
        for index in mutable
    }

    cache = {native: ensemble_metrics(native, maps, native_scores)}
    beam = [native]
    all_sequences = {native}
    for depth in range(1, args.max_mutations + 1):
        expanded = set()
        for parent in beam:
            current_mutations = {index for index, (old, new) in enumerate(zip(native, parent)) if old != new}
            for index in mutable:
                if index in current_mutations:
                    continue
                for amino_acid in options[index]:
                    sequence = parent[:index] + amino_acid + parent[index + 1:]
                    expanded.add(sequence)
        for sequence in expanded:
            if sequence not in cache:
                cache[sequence] = ensemble_metrics(sequence, maps, native_scores)
        beam = sorted(
            expanded,
            key=lambda sequence: (
                cache[sequence]["native_delta_p25"],
                cache[sequence]["native_delta_min"],
                cache[sequence]["multiconf_robust_score"],
            ),
            reverse=True,
        )[:args.beam_width]
        all_sequences.update(beam)

    rows = []
    plans = []
    for sequence in all_sequences - {native}:
        metrics = cache[sequence]
        candidate_uid = f"{reference_id}_{hashlib.sha256(sequence.encode('ascii')).hexdigest()[:12]}"
        heavy, light, applied, warnings, plan = build_full_chain(
            reference, template_map, sequence, candidate_uid
        )
        heavy_score = score_sequence(heavy)
        light_score = score_sequence(light)
        parent_heavy, parent_light, _, _, _ = build_full_chain(
            reference, template_map, native, f"{reference_id}_native"
        )
        parent_flags = set(filter(None, (
            score_sequence(parent_heavy)["flags"] + ";" + score_sequence(parent_light)["flags"]
        ).split(";")))
        candidate_flags = set(filter(None, (heavy_score["flags"] + ";" + light_score["flags"]).split(";")))
        introduced_flags = sorted(candidate_flags - parent_flags)
        risk = max(heavy_score["sequence_risk"], light_score["sequence_risk"])
        n_mutations = sum(old != new for old, new in zip(native, sequence))
        min_delta_threshold = getattr(args, "min_native_delta", -0.01)
        min_p25_threshold = getattr(args, "min_native_delta_p25", 0.0)
        passes = (
            metrics["native_delta_p25"] >= min_p25_threshold
            and metrics["native_delta_min"] >= min_delta_threshold
            and risk <= args.max_full_chain_risk
            and not introduced_flags
            and not warnings
            and all(row["region_label"] != "framework_or_unknown" for row in plan)
        )
        row = {
            "candidate_uid": candidate_uid,
            "reference_pdb": reference_id,
            "sequence": sequence,
            "native_paratope": native,
            "n_mutations": n_mutations,
            "applied_mutations": ";".join(applied),
            **metrics,
            "full_chain_risk": risk,
            "inherited_flags": ";".join(sorted(parent_flags)),
            "introduced_flags": ";".join(introduced_flags),
            "heavy_chain": role_chain(template_map, roles, "H"),
            "light_chain": role_chain(template_map, roles, "L"),
            "heavy_sequence": heavy,
            "light_sequence": light,
            "warnings": ";".join(warnings),
            "passes_design_gate": passes,
            "generator": "direct_multiconf_beam_v1",
        }
        rows.append(row)
        plans.extend(plan)
    rows.sort(key=lambda row: (
        row["passes_design_gate"], row["native_delta_p25"], row["native_delta_min"],
        row["multiconf_robust_score"], -row["n_mutations"],
    ), reverse=True)
    selected = [row for row in rows if row["passes_design_gate"]][:args.top_per_reference]
    for rank, row in enumerate(selected, 1):
        row["ensemble_design_rank"] = rank
    selected_uids = {row["candidate_uid"] for row in selected}
    return selected, [row for row in plans if row["candidate_uid"] in selected_uids], {
        "reference_pdb": reference_id,
        "mutable_cdr_contact_positions": len(mutable),
        "beam_width": args.beam_width,
        "max_mutations": args.max_mutations,
        "evaluated_beam_sequences": len(all_sequences) - 1,
        "passing": sum(row["passes_design_gate"] for row in rows),
        "selected": len(selected),
    }


def role_chain(contact_map, roles, target_role):
    return contact_chain_roles(contact_map, roles).get(target_role, "")


def main():
    args = parse_args()
    with open(PROJECT_ROOT / args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with open(PROJECT_ROOT / args.pose_audit, encoding="utf-8") as handle:
        audit = json.load(handle)
    if audit["status"] != "pass" or audit["passed"] != audit["poses"]:
        raise SystemExit("Strict pose panel must pass all poses")
    references = {row["pdb"]: row for row in config["references"]}
    selected = []
    plans = []
    summaries = []
    for reference_id in TARGETS:
        ref_rows, ref_plans, summary = design_reference(
            references[reference_id], audit["entries"], args
        )
        selected.extend(ref_rows)
        plans.extend(ref_plans)
        summaries.append(summary)

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(selected[0]) if selected else []
    with open(out_dir / "ensemble_designed_candidates.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    with open(out_dir / "ensemble_designed_mutation_plan.csv", "w", newline="", encoding="utf-8") as handle:
        fields_plan = list(plans[0]) if plans else []
        writer = csv.DictWriter(handle, fieldnames=fields_plan)
        writer.writeheader()
        writer.writerows(plans)
    with open(out_dir / "ensemble_designed_full_chains.fasta", "w", encoding="ascii") as handle:
        for row in selected:
            handle.write(f">{row['candidate_uid']}|{row['reference_pdb']}|heavy\n{row['heavy_sequence']}\n")
            handle.write(f">{row['candidate_uid']}|{row['reference_pdb']}|light\n{row['light_sequence']}\n")
    report = {
        "schema_version": "abeta.midregion_ensemble_design.v1",
        "status": "pass" if all(row["selected"] >= 2 for row in summaries) else "partial",
        "design_semantics": "direct beam search on five strict poses; CDR-like contact positions only",
        "references": summaries,
        "selected_total": len(selected),
        "claim_boundary": "computational sequence prioritization; no binding claim",
    }
    with open(out_dir / "ensemble_design_summary.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with open(out_dir / "ensemble_design_report.md", "w", encoding="utf-8") as handle:
        handle.write("# A-beta Middle-Region Ensemble-Conditioned Design\n\n")
        handle.write(f"Status: `{report['status']}`; selected: {len(selected)}.\n\n")
        handle.write("| Ref | Rank | UID | Mutations | Delta P25 | Delta min | Robust | Full-chain risk |\n")
        handle.write("|---|---:|---|---:|---:|---:|---:|---:|\n")
        for row in selected:
            handle.write(
                f"| {row['reference_pdb']} | {row['ensemble_design_rank']} | {row['candidate_uid']} | "
                f"{row['n_mutations']} | {row['native_delta_p25']:.4f} | "
                f"{row['native_delta_min']:.4f} | {row['multiconf_robust_score']:.4f} | "
                f"{row['full_chain_risk']:.4f} |\n"
            )
    print(f"Middle-region ensemble design: selected={len(selected)} status={report['status']}")
    if report["status"] != "pass":
        raise SystemExit("Middle-region design gate did not pass")


if __name__ == "__main__":
    main()
