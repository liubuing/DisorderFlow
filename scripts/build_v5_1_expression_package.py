#!/usr/bin/env python3
"""Build a sequence-finalization package for the complete 5CSZ Fab panel."""

import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from Bio.Seq import Seq
from dnachisel import (
    AvoidPattern,
    CodonOptimize,
    DnaOptimizationProblem,
    EnforceGCContent,
    EnforceTranslation,
    reverse_translate,
)

from prepare_v5_1_biological_constructs import parse_seqres

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/v5_1_candidates/abeta42_biological_constructs"
HEAVY_SIGNAL = "MDWTWRILFLVAAATGAHS"
LIGHT_SIGNAL = "MDMRVPAQLLGLLLLWFPGARC"
FORBIDDEN = ["GGTCTC", "GAGACC", "CGTCTC", "GAGACG", "GCTCTTC", "GAAGAGC", "AATAAA"]
VECTOR = "pcDNA3.4-TOPO TA"
VECTOR_CATALOG = "Thermo Fisher A14697"
CLONING_METHOD = "TOPO TA cloning of Taq-amplified insert"
NEGATIVE_CONTROL = "commercial non-binding human IgG1/kappa Fab isotype control"


def optimize_cds(protein):
    initial = reverse_translate(protein)
    constraints = [
        EnforceTranslation(translation=protein),
        EnforceGCContent(mini=0.40, maxi=0.68, window=60),
        *[AvoidPattern(pattern) for pattern in FORBIDDEN],
        *[AvoidPattern(base * 7) for base in "ACGT"],
    ]
    problem = DnaOptimizationProblem(
        sequence=initial,
        constraints=constraints,
        objectives=[CodonOptimize(species="h_sapiens", method="match_codon_usage")],
        logger=None,
    )
    problem.resolve_constraints()
    problem.optimize()
    sequence = str(problem.sequence).upper()
    translated = str(Seq(sequence).translate())
    if translated != protein:
        raise ValueError("Optimized CDS translation mismatch")
    violations = [pattern for pattern in FORBIDDEN if pattern in sequence]
    homopolymers = [base * 7 for base in "ACGT" if base * 7 in sequence]
    if violations or homopolymers:
        raise ValueError(f"Unresolved synthesis motifs: {violations + homopolymers}")
    gc = 100.0 * (sequence.count("G") + sequence.count("C")) / len(sequence)
    return sequence, gc


def add_chain(output, chain_id, role, mature, signal):
    precursor = signal + mature
    cds, gc = optimize_cds(precursor)
    output[chain_id] = {
        "chain_id": chain_id,
        "role": role,
        "signal_peptide": signal,
        "mature_protein": mature,
        "precursor_protein": precursor,
        "mature_length": len(mature),
        "precursor_length": len(precursor),
        "cds": cds,
        "cds_with_stop": cds + "TGA",
        "expression_insert": "GCCACC" + cds + "TGA",
        "cds_length": len(cds),
        "gc_percent": gc,
        "translation_verified": True,
        "forbidden_sites_absent": True,
    }


def sequence_sha256(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest().upper()


def validate_order_package(report):
    if report["status"] != "order_ready_for_pcDNA3_4_topo_ta":
        raise ValueError("Expression package is not marked order-ready")
    if report["required_manual_decisions"]:
        raise ValueError("Order-ready package still has unresolved manual decisions")
    for chain in report["chains"].values():
        insert = chain["expression_insert"]
        if not insert.startswith("GCCACCATG") or not insert.endswith("TGA"):
            raise ValueError(f"Invalid Kozak/start/stop contract: {chain['chain_id']}")
        translated = str(Seq(chain["cds"]).translate())
        if translated != chain["precursor_protein"]:
            raise ValueError(f"Order insert translation mismatch: {chain['chain_id']}")
        if chain["insert_sha256"] != sequence_sha256(insert):
            raise ValueError(f"Order insert checksum mismatch: {chain['chain_id']}")


def main():
    random.seed(42)
    np.random.seed(42)
    construct_report = json.loads((BASE / "biological_construct_report.json").read_text(encoding="utf-8"))
    screen = json.loads((BASE / "biological_screen.json").read_text(encoding="utf-8"))
    relaxation = json.loads((BASE / "relaxed/relaxation_report.json").read_text(encoding="utf-8"))
    screen_by_id = {record["construct_id"]: record for record in screen["records"]}
    relaxation_by_id = {record["construct_id"]: record for record in relaxation["records"]}
    constructs = []
    for record in construct_report["records"]:
        structure = screen_by_id[record["construct_id"]]
        relax = relaxation_by_id[record["construct_id"]]
        if not structure["structure_pass"] or not relax["relax_pass"]:
            raise ValueError(f"Construct is not ready for expression packaging: {record['construct_id']}")
        constructs.append({
            **record,
            "antibody_plddt": structure["antibody_plddt"],
            "iptm": structure["iptm"],
            "interface_pae": structure["interface_pae"],
            "relaxed_pdb": relax["relaxed_pdb"],
            "relax_ca_rmsd_angstrom": relax["ca_rmsd_angstrom"],
        })
    priority = {"5CSZ_cg_0190": 1, "5CSZ_cg_0032_N52H": 2, "5CSZ_cg_0069_N52H": 3}
    constructs.sort(key=lambda record: priority[record["construct_id"]])

    seqres = parse_seqres(ROOT / "data/anti_abeta_refs/5CSZ.pdb")
    parent_heavy, common_light = seqres["A"], seqres["B"]
    if any(record["light_sequence"] != common_light for record in constructs):
        raise ValueError("Candidate light chains are not identical")

    chains = {}
    add_chain(chains, "5CSZ_parent_heavy", "positive_control_heavy", parent_heavy, HEAVY_SIGNAL)
    add_chain(chains, "5CSZ_common_light", "shared_kappa_light", common_light, LIGHT_SIGNAL)
    for record in constructs:
        add_chain(
            chains, f"{record['construct_id']}_heavy", "candidate_heavy",
            record["heavy_sequence"], HEAVY_SIGNAL)

    for chain in chains.values():
        chain["insert_sha256"] = sequence_sha256(chain["expression_insert"])

    output_dir = BASE / "expression_package"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "mature_proteins.fasta").open("w", encoding="ascii") as handle:
        for chain in chains.values():
            handle.write(f">{chain['chain_id']}|mature\n{chain['mature_protein']}\n")
    with (output_dir / "precursor_proteins.fasta").open("w", encoding="ascii") as handle:
        for chain in chains.values():
            handle.write(f">{chain['chain_id']}|signal_plus_mature\n{chain['precursor_protein']}\n")
    with (output_dir / "optimized_cds.fasta").open("w", encoding="ascii") as handle:
        for chain in chains.values():
            handle.write(f">{chain['chain_id']}|CDS_plus_stop\n{chain['cds_with_stop']}\n")
    with (output_dir / "expression_inserts.fasta").open("w", encoding="ascii") as handle:
        for chain in chains.values():
            handle.write(f">{chain['chain_id']}|Kozak_CDS_stop\n{chain['expression_insert']}\n")

    manifest_fields = [
        "order_id", "role", "vector", "vector_catalog", "cloning_method",
        "insert_length_bp", "insert_sha256", "signal_peptide_policy",
        "delivery", "required_clone_qc",
    ]
    manifest_rows = []
    for chain in chains.values():
        manifest_rows.append({
            "order_id": chain["chain_id"],
            "role": chain["role"],
            "vector": VECTOR,
            "vector_catalog": VECTOR_CATALOG,
            "cloning_method": CLONING_METHOD,
            "insert_length_bp": len(chain["expression_insert"]),
            "insert_sha256": chain["insert_sha256"],
            "signal_peptide_policy": "retain encoded chain-specific signal peptide",
            "delivery": "sequence-verified plasmid; transfection-grade preparation",
            "required_clone_qc": "bidirectional Sanger sequencing across the complete insert",
        })
    with (output_dir / "order_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    report = {
        "status": "order_ready_for_pcDNA3_4_topo_ta",
        "expression_contract": {
            "host": "HEK293F transient mammalian expression",
            "format": "secreted human Fab, separate heavy and light plasmids",
            "transfection_ratio": "1:1 heavy:light molar ratio",
            "heavy_signal_peptide": HEAVY_SIGNAL,
            "light_signal_peptide": LIGHT_SIGNAL,
            "purification": "kappa-select or Fab-affinity; no engineered tag",
            "vector": VECTOR,
            "vector_catalog": VECTOR_CATALOG,
            "promoter": "full-length CMV immediate-early promoter/enhancer",
            "cloning_method": CLONING_METHOD,
            "cloning_flanks": "none encoded; TA overhang is generated during Taq amplification",
            "signal_peptide_policy": "retain the encoded native chain-specific signal peptides",
            "clone_qc": "verify orientation and the complete insert by bidirectional Sanger sequencing",
        },
        "codon_optimization": {
            "tool": "DNA Chisel 3.2.16",
            "species": "h_sapiens",
            "gc_window": 60,
            "gc_range": [0.40, 0.68],
            "forbidden_patterns": FORBIDDEN,
            "maximum_homopolymer": 6,
        },
        "plasmids": len(chains),
        "shared_light_chain": "5CSZ_common_light",
        "positive_control": "5CSZ_parent_heavy + 5CSZ_common_light",
        "negative_control": NEGATIVE_CONTROL,
        "negative_control_policy": (
            "assay reagent only; do not synthesize or co-transfect with the candidate light chain; "
            "accept only a vendor lot with Fab format and human IgG1/kappa identity documented"
        ),
        "constructs": constructs,
        "chains": chains,
        "order_artifacts": {
            "insert_fasta": "expression_inserts.fasta",
            "manifest": "order_manifest.csv",
        },
        "required_manual_decisions": [],
        "procurement_checks": [
            "vendor confirms pcDNA3.4-TOPO TA orientation and complete-insert Sanger coverage",
            "vendor confirms transfection-grade plasmid preparation and provides concentration/QC data",
            "isotype-control supplier documents human IgG1/kappa Fab format and non-binding specificity",
        ],
    }
    validate_order_package(report)
    (output_dir / "expression_package.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "candidate_fabs": len(constructs),
        "unique_plasmids": len(chains),
        "gc_percent_range": [
            min(chain["gc_percent"] for chain in chains.values()),
            max(chain["gc_percent"] for chain in chains.values()),
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
