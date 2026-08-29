#!/usr/bin/env python
"""Build a review-only 3D6 scFv expression and blinded BLI handoff package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

import yaml
from Bio.Seq import Seq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CODONS = {
    "A": "GCC", "C": "TGC", "D": "GAC", "E": "GAG", "F": "TTC",
    "G": "GGC", "H": "CAC", "I": "ATC", "K": "AAG", "L": "CTG",
    "M": "ATG", "N": "AAC", "P": "CCC", "Q": "CAG", "R": "CGC",
    "S": "AGC", "T": "ACC", "V": "GTG", "W": "TGG", "Y": "TAC",
}
FORBIDDEN_DNA = {
    "BsaI": ("GGTCTC", "GAGACC"),
    "BsmBI": ("CGTCTC", "GAGACG"),
    "SapI": ("GCTCTTC", "GAAGAGC"),
    "polyadenylation": ("AATAAA",),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sequence_sha256(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def load_boundary_module(path):
    spec = importlib.util.spec_from_file_location("variable_boundaries", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shuffled_h3(native, seed):
    rng = random.Random(seed)
    values = list(native)
    for _attempt in range(100):
        rng.shuffle(values)
        sequence = "".join(values)
        if sequence != native:
            return sequence
    raise ValueError("Could not construct H3 composition shuffle")


def replace_h3(heavy, h3):
    if len(h3) != 12:
        raise ValueError("The frozen 3D6 H3 must contain 12 residues")
    return heavy[:95] + h3 + heavy[107:]


def reverse_translate_reference(protein):
    dna = "".join(CODONS[amino_acid] for amino_acid in protein)
    if str(Seq(dna).translate()) != protein:
        raise ValueError("Reference CDS translation mismatch")
    return dna


def dna_audit(dna):
    hits = {
        name: [pattern for pattern in patterns if pattern in dna]
        for name, patterns in FORBIDDEN_DNA.items()
    }
    hits = {name: patterns for name, patterns in hits.items() if patterns}
    homopolymers = sorted(set(re.findall(r"A{7,}|C{7,}|G{7,}|T{7,}", dna)))
    gc = 100.0 * (dna.count("G") + dna.count("C")) / len(dna)
    return {"gc_percent": gc, "forbidden_pattern_hits": hits, "homopolymers_gt_6": homopolymers}


def build_entities(config, analysis):
    if len(analysis["shortlist"]) != int(config["input"]["expected_candidates"]):
        raise ValueError("Expression shortlist count differs from frozen contract")
    first = analysis["shortlist"][0]
    native_h3 = "VRYDHYSGSSDY"
    entities = []
    for row in analysis["shortlist"]:
        entities.append({
            "construct_id": row["entity_id"],
            "role": "candidate",
            "h3_sequence": row["h3_sequence"],
            "heavy_sequence": row["heavy_sequence"],
            "light_sequence": row["light_sequence"],
            "computational_rank": row["final_rank"],
        })
    native_heavy = replace_h3(first["heavy_sequence"], native_h3)
    entities.extend([
        {
            "construct_id": config["construct"]["native_control_id"],
            "role": "positive_control",
            "h3_sequence": native_h3,
            "heavy_sequence": native_heavy,
            "light_sequence": first["light_sequence"],
            "computational_rank": None,
        },
        {
            "construct_id": config["construct"]["shuffle_control_id"],
            "role": "counterfactual_sequence_control_not_known_nonbinder",
            "h3_sequence": shuffled_h3(native_h3, int(config["construct"]["shuffle_seed"])),
            "heavy_sequence": replace_h3(
                native_heavy,
                shuffled_h3(native_h3, int(config["construct"]["shuffle_seed"])),
            ),
            "light_sequence": first["light_sequence"],
            "computational_rank": None,
        },
    ])
    return entities


def build_constructs(config, entities, boundary):
    from scripts.pipeline.score_shortlist_developability import score_sequence

    native = next(row for row in entities if row["role"] == "positive_control")
    native_heavy_cut = boundary.find_variable_region(native["heavy_sequence"], "heavy")
    native_light_cut = boundary.find_variable_region(native["light_sequence"], "light")
    native_scfv = (
        native_heavy_cut["variable_seq"] + config["construct"]["vh_vl_linker"]
        + native_light_cut["variable_seq"] + config["construct"]["tag_linker"]
        + config["construct"]["purification_tag"]
    )
    native_nglyco = len(re.findall(r"N[^P][ST]", native_scfv))
    light_sequences = {row["light_sequence"] for row in entities}
    constructs = []
    for entity in entities:
        heavy_cut = boundary.find_variable_region(entity["heavy_sequence"], "heavy")
        light_cut = boundary.find_variable_region(entity["light_sequence"], "light")
        vh, vl = heavy_cut["variable_seq"], light_cut["variable_seq"]
        mature = (
            vh + config["construct"]["vh_vl_linker"] + vl
            + config["construct"]["tag_linker"] + config["construct"]["purification_tag"]
        )
        precursor = config["construct"]["signal_peptide"] + mature
        score = score_sequence(mature)
        variable_cys_even = vh.count("C") % 2 == 0 and vl.count("C") % 2 == 0
        nglyco = len(re.findall(r"N[^P][ST]", mature))
        gates = {
            "heavy_j_boundary": (
                heavy_cut["boundary_status"] == "j_motif_found"
                and heavy_cut["boundary_motif"] == config["sequence_gates"]["expected_vh_terminal_motif"]
            ),
            "light_j_boundary": (
                light_cut["boundary_status"] == "j_motif_found"
                and light_cut["boundary_motif"] == config["sequence_gates"]["expected_vl_terminal_motif"]
            ),
            "identical_vl": not config["sequence_gates"]["require_identical_vl"] or len(light_sequences) == 1,
            "developability": score["sequence_risk"] <= config["sequence_gates"][
                "maximum_scfv_developability_risk"],
            "n_glycosylation": (
                not config["sequence_gates"]["forbid_new_n_glycosylation_relative_to_native"]
                or nglyco <= native_nglyco
            ),
            "variable_cysteines": (
                not config["sequence_gates"]["require_even_variable_region_cysteines"]
                or variable_cys_even
            ),
        }
        cds = reverse_translate_reference(precursor)
        insert = config["dna_reference"]["kozak"] + cds + config["dna_reference"]["stop_codon"]
        audit = dna_audit(insert)
        gc_min, gc_max = config["dna_reference"]["acceptable_gc_range_for_order"]
        audit["gc_order_range_pass"] = gc_min * 100 <= audit["gc_percent"] <= gc_max * 100
        constructs.append({
            **entity,
            "vh_sequence": vh,
            "vl_sequence": vl,
            "vh_boundary": heavy_cut,
            "vl_boundary": light_cut,
            "mature_scfv": mature,
            "precursor_protein": precursor,
            "mature_length": len(mature),
            "precursor_length": len(precursor),
            "developability": score,
            "n_glycosylation_motifs": nglyco,
            "reference_cds": cds,
            "reference_expression_insert": insert,
            "reference_insert_sha256": sequence_sha256(insert),
            "reference_dna_audit": audit,
            "translation_verified": str(Seq(cds).translate()) == precursor,
            "sequence_gates": gates,
            "all_sequence_gates_pass": all(gates.values()),
        })
    return constructs


def write_fasta(path, records):
    path.write_text("\n".join(records) + "\n", encoding="ascii")


def assign_blind_ids(config, constructs):
    rng = random.Random(int(config["bli"]["blind_seed"]))
    ids = [f"BL-{index:03d}" for index in range(1, len(constructs) + 1)]
    rng.shuffle(ids)
    return {row["construct_id"]: blind_id for row, blind_id in zip(constructs, ids, strict=True)}


def plate_wells():
    return [f"{row}{column}" for row in "ABCDEFGH" for column in range(1, 13)]


def build_bli_layout(config, constructs, blind_ids):
    samples = [(row["construct_id"], blind_ids[row["construct_id"]]) for row in constructs]
    samples.extend([
        (config["construct"]["external_negative_control"], "BL-EXT-NEG"),
        (config["construct"]["process_blank"], "BL-MOCK"),
    ])
    conditions = []
    for construct_id, blind_id in samples:
        for analyte in config["bli"]["analytes"]:
            for replicate in range(1, int(config["bli"]["technical_replicates"]) + 1):
                conditions.append({
                    "blind_sample_id": blind_id,
                    "construct_id_key_only": construct_id,
                    "analyte": analyte["id"],
                    "analyte_role": analyte["role"],
                    "technical_replicate": replicate,
                    "required_readout": "raw_response;blank_subtracted_response;fit_quality",
                })
    rng = random.Random(int(config["bli"]["blind_seed"]) + 1)
    rng.shuffle(conditions)
    wells = plate_wells()
    public_rows = []
    for index, row in enumerate(conditions):
        public_rows.append({
            "plate": index // len(wells) + 1,
            "well": wells[index % len(wells)],
            **{key: value for key, value in row.items() if key != "construct_id_key_only"},
        })
    return public_rows, conditions


def write_package(config, config_path, analysis_path, constructs, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    blind_ids = assign_blind_ids(config, constructs)
    public_layout, keyed_layout = build_bli_layout(config, constructs, blind_ids)
    write_fasta(output_dir / "scfv_mature_proteins.fasta", [
        value for row in constructs
        for value in (f">{row['construct_id']}|mature_scfv", row["mature_scfv"])
    ])
    write_fasta(output_dir / "scfv_precursor_proteins.fasta", [
        value for row in constructs
        for value in (f">{row['construct_id']}|signal_scfv_his6", row["precursor_protein"])
    ])
    write_fasta(output_dir / "reference_cds_not_order_ready.fasta", [
        value for row in constructs
        for value in (f">{row['construct_id']}|nonoptimized_reference_CDS_plus_stop",
                      row["reference_cds"] + config["dna_reference"]["stop_codon"])
    ])
    manifest_fields = [
        "construct_id", "role", "computational_rank", "h3_sequence", "mature_length",
        "precursor_length", "sequence_risk", "all_sequence_gates_pass",
        "translation_verified", "reference_insert_sha256", "order_status",
    ]
    with (output_dir / "construct_manifest.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        for row in constructs:
            writer.writerow({
                "construct_id": row["construct_id"], "role": row["role"],
                "computational_rank": row["computational_rank"], "h3_sequence": row["h3_sequence"],
                "mature_length": row["mature_length"], "precursor_length": row["precursor_length"],
                "sequence_risk": row["developability"]["sequence_risk"],
                "all_sequence_gates_pass": row["all_sequence_gates_pass"],
                "translation_verified": row["translation_verified"],
                "reference_insert_sha256": row["reference_insert_sha256"],
                "order_status": "blocked_pending_vendor_codon_optimization_and_manual_vector_review",
            })
    with (output_dir / "bli_blinded_layout.csv").open("w", newline="", encoding="ascii") as handle:
        fields = ["plate", "well", "blind_sample_id", "analyte", "analyte_role",
                  "technical_replicate", "required_readout"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(public_layout)
    with (output_dir / "bli_blinding_key.csv").open("w", newline="", encoding="ascii") as handle:
        fields = ["blind_sample_id", "construct_id", "role"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        role_by_id = {row["construct_id"]: row["role"] for row in constructs}
        for construct_id, blind_id in sorted(blind_ids.items(), key=lambda item: item[1]):
            writer.writerow({"blind_sample_id": blind_id, "construct_id": construct_id,
                             "role": role_by_id[construct_id]})
        writer.writerow({"blind_sample_id": "BL-EXT-NEG",
                         "construct_id": config["construct"]["external_negative_control"],
                         "role": "external_negative_control"})
        writer.writerow({"blind_sample_id": "BL-MOCK",
                         "construct_id": config["construct"]["process_blank"],
                         "role": "process_blank"})
    with (output_dir / "expression_batch_tracking.csv").open(
        "w", newline="", encoding="ascii"
    ) as handle:
        fields = [
            "construct_id", "blind_sample_id", "expression_batch", "expression_date",
            "operator", "yield_mg_per_l", "sec_monomer_fraction", "dls_pdi",
            "endotoxin_eu_per_mg", "qc_pass", "raw_data_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in constructs:
            for batch in range(1, int(config["bli"]["replication"][
                "independent_expression_batches"]) + 1):
                writer.writerow({
                    "construct_id": row["construct_id"],
                    "blind_sample_id": blind_ids[row["construct_id"]],
                    "expression_batch": batch,
                })
    with (output_dir / "procurement_bom_template.csv").open(
        "w", newline="", encoding="ascii"
    ) as handle:
        fields = [
            "item", "role", "exact_sequence_or_identity", "vendor", "catalog",
            "lot", "purity", "labeling", "qc_document", "status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item, role in (
            ("A-beta 1-11 peptide", "target analyte"),
            ("scrambled A-beta 1-11 peptide", "sequence control analyte"),
            ("BSA", "matrix control analyte"),
            (config["construct"]["external_negative_control"], "true negative protein control"),
            ("BLI sensors", "assay consumable"),
            ("expression vector", "construct backbone"),
        ):
            writer.writerow({"item": item, "role": role, "status": "required_before_order_or_assay"})
    with (output_dir / "bli_results_template.csv").open(
        "w", newline="", encoding="ascii"
    ) as handle:
        fields = [
            "plate", "well", "blind_sample_id", "analyte", "technical_replicate",
            "raw_response", "blank_subtracted_response", "replicate_cv", "fit_quality",
            "concentration", "concentration_unit", "raw_data_path", "operator_exclusion_reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in public_layout:
            writer.writerow({key: row.get(key, "") for key in fields})
    acceptance = {
        "status": "preregistered_before_measurements",
        "assay_validity": {
            "native_control": "3D6-native must show reproducible target signal",
            "external_negative": "matched irrelevant scFv defines nonspecific protein baseline",
            "mock": "mock-transfection blank must remain below assay detection threshold",
            "maximum_replicate_cv": config["bli"]["decision"]["maximum_replicate_cv"],
        },
        "candidate_screening_pass": {
            "minimum_signal_ratio_vs_external_negative": config["bli"]["decision"][
                "minimum_blank_subtracted_signal_ratio_vs_external_negative"],
            "maximum_scrambled_and_bsa_fraction_of_target": config["bli"]["decision"][
                "maximum_scrambled_and_bsa_fraction_of_target"],
            "requires_both_independent_expression_batches": True,
        },
        "followup": (
            "Screening pass advances to a separately randomized multi-concentration BLI/SPR run; "
            "single-concentration screening cannot establish KD."
        ),
        "counterfactual_control_boundary": (
            "3D6-H3-shuffle-counterfactual is not an experimentally established non-binder and "
            "cannot replace the external irrelevant-scFv negative control."
        ),
    }
    (output_dir / "experimental_acceptance_criteria.json").write_text(
        json.dumps(acceptance, indent=2) + "\n", encoding="ascii")
    report = {
        "schema_version": 1,
        "status": (
            "sequence_review_pass_order_blocked_pending_vendor_optimization"
            if all(row["all_sequence_gates_pass"] and row["translation_verified"] for row in constructs)
            else "no_go_construct_sequence_gate_failed"
        ),
        "config": str(config_path.relative_to(ROOT).as_posix()),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__)),
        "analysis": str(analysis_path.relative_to(ROOT).as_posix()),
        "analysis_sha256": sha256(analysis_path),
        "construct_contract": config["construct"],
        "dna_classification": config["dna_reference"]["classification"],
        "vendor_requirements": config["dna_reference"]["vendor_requirements"],
        "summary": {
            "candidate_constructs": sum(row["role"] == "candidate" for row in constructs),
            "internal_sequence_controls": sum(row["role"] != "candidate" for row in constructs),
            "all_sequence_gates_pass": sum(row["all_sequence_gates_pass"] for row in constructs),
            "translation_verified": sum(row["translation_verified"] for row in constructs),
            "reference_cds_gc_order_range_pass": sum(
                row["reference_dna_audit"]["gc_order_range_pass"] for row in constructs
            ),
            "bli_conditions": len(public_layout),
            "bli_plates": max(row["plate"] for row in public_layout),
        },
        "bli_decision_contract": config["bli"]["decision"],
        "constructs": constructs,
        "required_manual_decisions": [
            "Confirm expression vector, promoter, cloning method, and signal-peptide compatibility.",
            "Vendor returns HEK293-optimized insert and exact translated protein for approval.",
            "Procure irrelevant human scFv with matched His6 format and document lot identity.",
            "Define A-beta and scrambled-peptide vendor, lot, purity, biotinylation, and state QC.",
            "Keep bli_blinding_key.csv inaccessible to the assay operator and primary analyst.",
            "Do not treat the H3 shuffle counterfactual as a known non-binding negative control.",
        ],
        "claim_boundary": config["claim_boundary"],
    }
    (output_dir / "expression_package.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="ascii")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmarks/abeta_4hix_expression_prep_v1.yml")
    parser.add_argument("--out")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    analysis_path = ROOT / config["input"]["analysis"]
    boundary_path = ROOT / config["input"]["boundary_script"]
    if sha256(analysis_path) != config["input"]["analysis_sha256"]:
        raise ValueError("Frozen AF2 analysis hash mismatch")
    if sha256(boundary_path) != config["input"]["boundary_script_sha256"]:
        raise ValueError("Variable-boundary implementation hash mismatch")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    entities = build_entities(config, analysis)
    constructs = build_constructs(config, entities, load_boundary_module(boundary_path))
    output_dir = ROOT / (args.out or config["output"])
    report = write_package(config, config_path, analysis_path, constructs, output_dir)
    print(json.dumps({"status": report["status"], **report["summary"]}, indent=2))
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
