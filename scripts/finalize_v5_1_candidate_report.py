#!/usr/bin/env python3
"""Finalize the corrected v5.1 candidate decision and hypothesis queue."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/v5_1_candidates"
ABETA42 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    generation = load(BASE / "abeta42/generation_report.json")
    de_novo_af2 = load(BASE / "abeta42/af2_screen.json")
    local_rerank = load(BASE / "abeta42_local/local_rerank_report.json")
    local_af2 = load(BASE / "abeta42_local/af2_screen.json")
    controls = load(BASE / "abeta42_local/af2_controls/control_report.json")
    control_by_reference = {
        record["control"]: record
        for record in controls["records"] if record["condition"] == "full_abeta42"
    }

    candidates = []
    for record in local_af2["records"]:
        control = control_by_reference[record["reference_pdb"]]
        iptm_delta = record["af2_iptm"] - control["iptm"]
        interface_pae_delta = record["af2_interface_pae"] - control["interface_pae"]
        plddt_delta = record["af2_antibody_plddt"] - control["antibody_plddt"]
        relative_improvement = iptm_delta > 0 and interface_pae_delta < 0
        record = {
            **record,
            "control_iptm": control["iptm"],
            "control_interface_pae": control["interface_pae"],
            "iptm_delta_vs_parent_control": iptm_delta,
            "interface_pae_delta_vs_parent_control": interface_pae_delta,
            "antibody_plddt_delta_vs_parent_control": plddt_delta,
            "relative_side_evidence": "improved" if relative_improvement else "not_improved",
            "decision_status": "hypothesis_only_not_synthesis_ready",
        }
        record["decision_score"] = (
            float(record["corrected_rank_score"])
            + 0.30 * iptm_delta
            - 0.01 * interface_pae_delta
        )
        candidates.append(record)
    candidates.sort(
        key=lambda record: (
            record["relative_side_evidence"] == "improved", record["decision_score"]),
        reverse=True)
    queue = candidates[:6]
    for rank, record in enumerate(queue, 1):
        record["hypothesis_rank"] = rank

    final_status = "no_synthesis_ready_candidate"
    local_status = "retained_as_hypothesis_queue"
    structure_decision = {
        "status": "invalid_for_decision",
        "reason": "Experimentally resolved 4HIX and 5CSZ positive controls fail the same pLDDT/interface-PAE gates",
        "calibrated": controls["calibrated_for_candidate_gating"],
    }
    required_before_synthesis = [
        "calibrated three-chain structure predictor or template-aware AF2 control pass",
        "side-chain rebuild and interface repack",
        "orthogonal developability review",
        "binding and monomer/oligomer/fibril specificity experiments",
    ]
    calibrated_path = BASE / "abeta42_colabfold_candidates/calibrated_screen.json"
    if calibrated_path.exists():
        calibrated_screen = load(calibrated_path)
        if calibrated_screen.get("calibrated"):
            old_by_id = {record["construct_id"]: record for record in candidates}
            queue = []
            for screened in calibrated_screen["records"]:
                merged = {**old_by_id[screened["construct_id"]], **screened}
                for key in list(merged):
                    if key.startswith("af2_") or key in {
                        "control_iptm", "control_interface_pae",
                        "iptm_delta_vs_parent_control",
                        "interface_pae_delta_vs_parent_control",
                        "antibody_plddt_delta_vs_parent_control",
                    }:
                        merged.pop(key)
                merged["hypothesis_rank"] = screened["panel_rank"]
                merged["decision_status"] = "structure_screen_pass_not_synthesis_ready"
                queue.append(merged)
            candidates = queue
            final_status = "structure_screen_pass_not_synthesis_ready"
            local_status = "calibrated_structure_screen_pass"
            structure_decision = {
                "status": "calibrated_pass",
                "calibrated": True,
                "controls": calibrated_screen["controls"],
                "candidates_tested": calibrated_screen["n_candidates"],
                "candidates_passed": calibrated_screen["n_structure_pass"],
            }
            required_before_synthesis = [
                "side-chain rebuild and interface repack",
                "orthogonal developability review",
                "binding and monomer/oligomer/fibril specificity experiments",
            ]

    relaxation_path = BASE / "abeta42_relaxed/relaxation_report.json"
    developability_path = ROOT / "outputs/synthesis_candidate_developability_nglyco_rescued_v2/shortlist_developability.csv"
    if calibrated_path.exists() and relaxation_path.exists() and developability_path.exists():
        relaxation = load(relaxation_path)
        relaxation_by_id = {record["construct_id"]: record for record in relaxation["records"]}
        with developability_path.open(newline="", encoding="utf-8") as handle:
            developability_by_id = {
                record["construct_id"]: record for record in csv.DictReader(handle)
            }
        finalized = []
        for record in candidates:
            relax_record = relaxation_by_id.get(record["construct_id"], {})
            developability_record = developability_by_id.get(record["construct_id"], {})
            record["relax_pass"] = bool(relax_record.get("relax_pass"))
            record["relaxed_pdb"] = relax_record.get("relaxed_pdb")
            record["relax_ca_rmsd_angstrom"] = relax_record.get("ca_rmsd_angstrom")
            record["developability_status"] = developability_record.get("developability_status")
            record["developability_risk"] = developability_record.get("combined_developability_risk")
            record["synthesis_panel_pass"] = (
                record.get("structure_pass", False)
                and record.get("full_construct_sequence_exact", False)
                and record["relax_pass"]
                and record["developability_status"] == "developability_pass"
            )
            if record["synthesis_panel_pass"]:
                finalized.append(record)
        finalized.sort(key=lambda record: record["panel_score"], reverse=True)
        for rank, record in enumerate(finalized, 1):
            record["hypothesis_rank"] = rank
            record["decision_status"] = "experimental_synthesis_panel_ready"
        queue = finalized
        final_status = "experimental_synthesis_panel_ready"
        required_before_synthesis = [
            "manual construct and chain-format review",
            "expression-vector selection and codon optimization",
            "include native 5CSZ positive control and an isotype negative control",
        ]
        structure_decision["relaxation"] = {
            "tested": relaxation["n_completed"],
            "passed": relaxation["n_pass"],
        }
        structure_decision["final_exact_sequence_panel"] = len(queue)

    biological_screen_path = BASE / "abeta42_biological_constructs/biological_screen.json"
    biological_relax_path = BASE / "abeta42_biological_constructs/relaxed/relaxation_report.json"
    expression_package_path = BASE / "abeta42_biological_constructs/expression_package/expression_package.json"
    if biological_screen_path.exists() and biological_relax_path.exists() and expression_package_path.exists():
        biological_screen = load(biological_screen_path)
        biological_relax = load(biological_relax_path)
        expression_package = load(expression_package_path)
        old_by_id = {record["construct_id"]: record for record in candidates}
        screen_by_id = {record["construct_id"]: record for record in biological_screen["records"]}
        relax_by_id = {record["construct_id"]: record for record in biological_relax["records"]}
        biological_panel = []
        for rank, construct in enumerate(expression_package["constructs"], 1):
            construct_id = construct["construct_id"]
            screen_record = screen_by_id[construct_id]
            relax_record = relax_by_id[construct_id]
            if not screen_record["structure_pass"] or not relax_record["relax_pass"]:
                raise ValueError(f"Biological construct gate failed for {construct_id}")
            merged = {
                **old_by_id[construct_id],
                **screen_record,
                "hypothesis_rank": rank,
                "decision_status": "expression_package_ready_pending_manual_vector_choices",
                "full_construct_sequence_exact": True,
                "relax_pass": True,
                "relaxed_pdb": relax_record["relaxed_pdb"],
                "relax_ca_rmsd_angstrom": relax_record["ca_rmsd_angstrom"],
                "synthesis_panel_pass": True,
            }
            biological_panel.append(merged)
        queue = biological_panel
        candidates = biological_panel
        final_status = "expression_package_ready_pending_manual_vector_choices"
        local_status = "complete_seqres_structure_and_relaxation_pass"
        required_before_synthesis = expression_package["required_manual_decisions"]
        structure_decision = {
            "status": "complete_seqres_calibrated_pass",
            "sequence_contract": biological_screen["sequence_contract"],
            "parent_calibrated": biological_screen["parent_calibrated"],
            "parent": biological_screen["parent"],
            "candidates_tested": biological_screen["n_candidates"],
            "candidates_passed": biological_screen["n_structure_pass"],
            "relaxation_tested": biological_relax["n_completed"],
            "relaxation_passed": biological_relax["n_pass"],
        }

    output_dir = BASE / "abeta42_final"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": final_status,
        "checkpoint": generation["checkpoint"],
        "target": "A-beta42",
        "target_sequence": ABETA42,
        "decisions": {
            "de_novo_generation": {
                "status": "rejected",
                "reason": "30-32 mutations across 33 designed CDR residues and 0/10 AF2 absolute passes",
                "generated": generation["generation"]["factual_candidates"],
                "quality_pass": generation["generation"]["quality_pass"],
                "af2_tested": de_novo_af2["n_candidates"],
            },
            "local_variant_rerank": {
                "status": local_status,
                "scored": local_rerank["n_scored"],
                "mean_condition_advantage": local_rerank["mean_condition_advantage"],
                "af2_tested": local_af2["n_candidates"],
            },
            "af2_absolute_gate": structure_decision,
        },
        "hypothesis_queue_count": len(queue),
        "synthesis_panel_count": len(queue) if final_status in {
            "experimental_synthesis_panel_ready",
            "expression_package_ready_pending_manual_vector_choices",
        } else 0,
        "relative_side_evidence_improved": sum(
            record["relative_side_evidence"] == "improved" for record in candidates),
        "required_before_synthesis": required_before_synthesis,
        "required_after_synthesis": [
            "expression and monodispersity QC",
            "binding affinity against A-beta monomer, oligomer, and fibril",
            "off-target and state-specificity controls",
        ],
        "hypothesis_queue": queue,
        "synthesis_panel": queue if final_status in {
            "experimental_synthesis_panel_ready",
            "expression_package_ready_pending_manual_vector_choices",
        } else [],
        "all_local_candidates": candidates,
        "evidence": {
            "generation": "results/v5_1_candidates/abeta42/generation_report.json",
            "de_novo_af2": "results/v5_1_candidates/abeta42/af2_screen.json",
            "local_rerank": "results/v5_1_candidates/abeta42_local/local_rerank_report.json",
            "local_af2": "results/v5_1_candidates/abeta42_local/af2_screen.json",
            "af2_controls": "results/v5_1_candidates/abeta42_local/af2_controls/control_report.json",
            "calibrated_colabfold": "results/v5_1_candidates/abeta42_colabfold_candidates/calibrated_screen.json",
            "relaxation": "results/v5_1_candidates/abeta42_relaxed/relaxation_report.json",
            "developability": "outputs/synthesis_candidate_developability_nglyco_rescued_v2/shortlist_developability.csv",
            "biological_constructs": "results/v5_1_candidates/abeta42_biological_constructs/biological_construct_report.json",
            "biological_screen": "results/v5_1_candidates/abeta42_biological_constructs/biological_screen.json",
            "biological_relaxation": "results/v5_1_candidates/abeta42_biological_constructs/relaxed/relaxation_report.json",
            "expression_package": "results/v5_1_candidates/abeta42_biological_constructs/expression_package/expression_package.json",
        },
    }
    (output_dir / "candidate_decision.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    fields = [
        "hypothesis_rank", "construct_id", "reference_pdb", "candidate_id", "n_mutations",
        "mutations", "condition_advantage", "contact_retention", "antibody_plddt", "iptm",
        "iptm_delta_vs_parent", "interface_pae", "interface_pae_delta_vs_parent",
        "structure_pass", "panel_score", "decision_status", "structure_pdb",
        "full_construct_sequence_exact", "relax_pass", "relaxed_pdb",
        "relax_ca_rmsd_angstrom", "developability_status", "developability_risk",
        "synthesis_panel_pass",
    ]
    with (output_dir / "candidate_hypothesis_queue.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(queue)
    with (output_dir / "candidate_hypothesis_queue.fasta").open("w", encoding="ascii") as handle:
        for record in queue:
            handle.write(
                f">rank={record['hypothesis_rank']}|{record['construct_id']}|status={record['decision_status']}\n")
            handle.write(
                f"{record['heavy_sequence']}:{record['light_sequence']}:"
                f"{record.get('epitope_sequence', ABETA42)}\n")
    print(json.dumps({
        "status": report["status"],
        "hypothesis_queue_count": len(queue),
        "relative_side_evidence_improved": report["relative_side_evidence_improved"],
    }, indent=2))


if __name__ == "__main__":
    main()
