#!/usr/bin/env python3
"""Create expression and A-beta state-specificity assay planning artifacts."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/v5_1_candidates/abeta42_biological_constructs/expression_package"
ISOTYPE_CONTROL = "human_IgG1_kappa_Fab_isotype_control"


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    package = json.loads((BASE / "expression_package.json").read_text(encoding="utf-8"))
    candidates = [record["construct_id"] for record in package["constructs"]]

    expression_conditions = [
        *[(candidate, f"{candidate}_heavy", "candidate") for candidate in candidates],
        ("5CSZ_complete_parent", "5CSZ_parent_heavy", "positive_control"),
        (ISOTYPE_CONTROL, "external_complete_Fab_reagent", "negative_control"),
        ("mock_transfection", "none", "process_blank"),
    ]
    expression_rows = []
    wells = [f"{row}{column}" for row in "ABCD" for column in range(1, 7)]
    index = 0
    for sample, heavy, role in expression_conditions:
        for replicate in range(1, 4):
            expression_rows.append({
                "well": wells[index],
                "sample": sample,
                "role": role,
                "heavy_plasmid": heavy,
                "light_plasmid": "5CSZ_common_light" if role not in ("negative_control", "process_blank") else (
                    "included_in_external_complete_Fab_reagent" if role == "negative_control" else "none"),
                "heavy_light_molar_ratio": "1:1" if role != "process_blank" else "NA",
                "biological_replicate": replicate,
            })
            index += 1
    write_csv(
        BASE / "expression_plate_24well.csv",
        ["well", "sample", "role", "heavy_plasmid", "light_plasmid", "heavy_light_molar_ratio", "biological_replicate"],
        expression_rows,
    )

    analytes = [
        ("A_beta_1_42_monomer", "target_state"),
        ("A_beta_1_42_oligomer", "target_state"),
        ("A_beta_1_42_fibril", "target_state"),
        ("A_beta_1_11_peptide", "epitope_positive"),
        ("scrambled_A_beta_1_42", "sequence_negative"),
        ("BSA", "matrix_negative"),
    ]
    samples = candidates + ["5CSZ_complete_parent", ISOTYPE_CONTROL]
    binding_rows = []
    for sample in samples:
        for analyte, role in analytes:
            binding_rows.append({
                "sample": sample,
                "analyte": analyte,
                "analyte_role": role,
                "primary_readout": "BLI_kinetics" if analyte != "A_beta_1_42_fibril" else "ELISA_binding_curve",
                "replicates": 3,
                "required_outputs": "response;KD_or_EC50;replicate_CV;fit_quality",
            })
    write_csv(
        BASE / "binding_specificity_matrix.csv",
        ["sample", "analyte", "analyte_role", "primary_readout", "replicates", "required_outputs"],
        binding_rows,
    )

    criteria = {
        "status": "pre_registered_experimental_plan",
        "expression_qc": {
            "biological_replicates": 3,
            "required": [
                "detectable secreted Fab in all biological replicates",
                "expected heavy/light bands under reducing conditions",
                "predominant assembled Fab under non-reducing conditions",
                "SEC monomer fraction >= 0.90",
            ],
            "report_without_hard_gate": ["yield_mg_L", "thermal_transition", "DLS_polydispersity"],
        },
        "assay_validity": {
            "parent_positive_control": "5CSZ complete parent must bind A-beta1-11 and at least one A-beta1-42 state",
            "negative_control": "commercial human IgG1/kappa Fab isotype response must remain below the assay-specific detection threshold",
            "replicate_cv_max": 0.20,
            "blank_subtraction_required": True,
        },
        "candidate_decision": {
            "binding_pass": "reproducible concentration-dependent signal above isotype on A-beta1-11 and at least one A-beta1-42 state",
            "state_preference_report": "oligomer/monomer and fibril/monomer response ratios with confidence intervals",
            "specificity_pass": "scrambled peptide and BSA signals remain below 20% of the strongest A-beta state signal",
            "advance_rule": "expression_qc pass AND binding_pass AND specificity_pass; rank by state preference only after these gates",
        },
        "interpretation_limits": [
            "Computational disorder conditioning does not establish experimental state specificity.",
            "A-beta preparation state must be independently characterized before comparing candidates.",
            "Affinity and state-preference claims require orthogonal confirmation.",
        ],
    }
    (BASE / "experimental_acceptance_criteria.json").write_text(
        json.dumps(criteria, indent=2), encoding="utf-8")

    summary = {
        "status": "expression_and_binding_plan_ready",
        "candidate_fabs": candidates,
        "unique_candidate_plasmids": package["plasmids"],
        "expression_wells": len(expression_rows),
        "binding_conditions": len(binding_rows),
        "controls": ["5CSZ_complete_parent", ISOTYPE_CONTROL, "mock_transfection"],
        "artifacts": {
            "expression_plate": "expression_plate_24well.csv",
            "binding_matrix": "binding_specificity_matrix.csv",
            "acceptance_criteria": "experimental_acceptance_criteria.json",
        },
    }
    (BASE / "experimental_plan_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
