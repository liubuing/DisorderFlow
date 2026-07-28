#!/usr/bin/env python3
"""Rebuild complete 5CSZ Fab candidates from PDB SEQRES records."""

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDB = ROOT / "data/anti_abeta_refs/5CSZ.pdb"
AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def parse_seqres(path):
    chains = {}
    declared = {}
    with path.open(encoding="ascii") as handle:
        for line in handle:
            if not line.startswith("SEQRES"):
                continue
            chain = line[11:12].strip()
            declared[chain] = int(line[13:17])
            chains.setdefault(chain, []).extend(
                AA3[residue] for residue in line[19:70].split())
    sequences = {chain: "".join(residues) for chain, residues in chains.items()}
    for chain, sequence in sequences.items():
        if len(sequence) != declared[chain]:
            raise ValueError(f"SEQRES length mismatch for chain {chain}")
    return sequences


def apply_mutations(sequence, mutation_text, chain):
    output = list(sequence)
    applied = []
    for token in mutation_text.split(";"):
        mutation_chain, position, native, replacement = re.fullmatch(
            r"([A-Za-z0-9]):(\d+):([A-Z])>([A-Z])", token).groups()
        if mutation_chain != chain:
            continue
        index = int(position) - 1
        if output[index] != native:
            raise ValueError(
                f"Native mismatch {token}: complete SEQRES has {output[index]}")
        output[index] = replacement
        applied.append(token)
    return "".join(output), applied


def main():
    seqres = parse_seqres(PDB)
    heavy_parent = seqres["A"]
    light_parent = seqres["B"]
    epitope = seqres["E"]
    if (len(heavy_parent), len(light_parent), epitope) != (228, 215, "DAEFRHDSGYE"):
        raise ValueError("Unexpected 5CSZ biological sequence contract")
    developability_path = (
        ROOT / "outputs/synthesis_candidate_developability_nglyco_rescued_v2/shortlist_developability.csv")
    with developability_path.open(newline="", encoding="utf-8") as handle:
        source = {row["construct_id"]: row for row in csv.DictReader(handle)}
    decision = json.loads((
        ROOT / "results/v5_1_candidates/abeta42_final/candidate_decision.json"
    ).read_text(encoding="utf-8"))
    selected_ids = [row["construct_id"] for row in decision["synthesis_panel"]]
    output_dir = ROOT / "results/v5_1_candidates/abeta42_biological_constructs"
    fasta_dir = output_dir / "candidate_fastas"
    fasta_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for construct_id in selected_ids:
        row = source[construct_id]
        mutation_text = row["applied_mutations"]
        heavy, heavy_applied = apply_mutations(heavy_parent, mutation_text, "A")
        light, light_applied = apply_mutations(light_parent, mutation_text, "B")
        expected = len([token for token in mutation_text.split(";") if token])
        if len(heavy_applied) + len(light_applied) != expected:
            raise ValueError(f"Not all mutations applied for {construct_id}")
        fasta = fasta_dir / f"{construct_id}.fasta"
        fasta.write_text(
            f">{construct_id}\n{heavy}:{light}:{epitope}\n", encoding="ascii")
        records.append({
            "construct_id": construct_id,
            "heavy_sequence": heavy,
            "light_sequence": light,
            "epitope_sequence": epitope,
            "heavy_length": len(heavy),
            "light_length": len(light),
            "mutations": mutation_text,
            "fasta": str(fasta.relative_to(ROOT)),
            "heavy_terminal": heavy[-10:],
            "light_terminal": light[-10:],
            "light_terminal_cysteine_present": light.endswith("C"),
        })
    parent_fasta = output_dir / "5CSZ_complete_parent.fasta"
    parent_fasta.write_text(
        f">5CSZ_complete_parent\n{heavy_parent}:{light_parent}:{epitope}\n",
        encoding="ascii")
    parent_a3m = output_dir / "parent_msa64_r1/5CSZ_complete_parent.a3m"
    if parent_a3m.exists():
        a3m_dir = output_dir / "candidate_a3ms"
        a3m_dir.mkdir(exist_ok=True)
        parent_lines = parent_a3m.read_text(encoding="utf-8").splitlines()
        if parent_lines[0] != "#228,215,11\t1,1,1":
            raise ValueError("Unexpected complete parent A3M contract")
        for record in records:
            lines = list(parent_lines)
            lines[2] = (
                record["heavy_sequence"] + record["light_sequence"]
                + record["epitope_sequence"])
            candidate_a3m = a3m_dir / f"{record['construct_id']}.a3m"
            candidate_a3m.write_text("\n".join(lines) + "\n", encoding="utf-8")
            record["candidate_a3m"] = str(candidate_a3m.relative_to(ROOT))
    report = {
        "source": "5CSZ PDB SEQRES",
        "parent_heavy_length": len(heavy_parent),
        "parent_light_length": len(light_parent),
        "epitope_sequence": epitope,
        "missing_from_atom_derived_contract": {
            "heavy_residues": 16,
            "light_residues": 1,
            "critical_light_terminal": "Cys215",
        },
        "parent_fasta": str(parent_fasta.relative_to(ROOT)),
        "parent_a3m": str(parent_a3m.relative_to(ROOT)) if parent_a3m.exists() else None,
        "records": records,
    }
    (output_dir / "biological_construct_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    with (output_dir / "complete_protein_chains.fasta").open("w", encoding="ascii") as handle:
        for record in records:
            handle.write(f">{record['construct_id']}|heavy\n{record['heavy_sequence']}\n")
            handle.write(f">{record['construct_id']}|light\n{record['light_sequence']}\n")
    print(json.dumps({
        "reconstructed": len(records),
        "heavy_length": len(heavy_parent),
        "light_length": len(light_parent),
        "epitope": epitope,
    }, indent=2))


if __name__ == "__main__":
    main()
