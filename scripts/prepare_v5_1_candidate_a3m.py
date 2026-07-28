#!/usr/bin/env python3
"""Apply audited local mutations to native PDB chains and reuse parent MSAs."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "modules"))

from state_contact_scorer import AA3_TO_1, extract_contact_map  # noqa: E402


REFERENCES = {
    "4HIX": {
        "pdb": ROOT / "data/anti_abeta_refs/4HIX.pdb",
        "chains": ["H", "L"],
        "peptide_chain": "A",
        "a3m": ROOT / "results/v5_1_candidates/abeta42_colabfold_calibration/full_msa_no_templates/4HIX_native_DAEFRH.a3m",
    },
    "5CSZ": {
        "pdb": ROOT / "data/anti_abeta_refs/5CSZ.pdb",
        "chains": ["A", "B"],
        "peptide_chain": "E",
        "a3m": ROOT / "results/v5_1_candidates/abeta42_colabfold_calibration/5csz_msa_only/5CSZ_native_DAEFRHDSGY.a3m",
    },
}


def chain_residues(path, chain_id):
    residues = []
    seen = set()
    with path.open(encoding="ascii") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")) or line[21:22].strip() != chain_id:
                continue
            key = int(line[22:26])
            if key in seen:
                continue
            aa = AA3_TO_1.get(line[17:20].strip())
            if aa:
                seen.add(key)
                residues.append([key, aa])
    return residues


def apply_mutations(reference_name, mutation_text):
    reference = REFERENCES[reference_name]
    contact_map = extract_contact_map(
        str(reference["pdb"]), peptide_chain=reference["peptide_chain"])
    mutations = {}
    for token in mutation_text.split(";"):
        match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", token)
        if not match:
            raise ValueError(f"Invalid mutation {token}")
        native, index, replacement = match.groups()
        paratope = contact_map["paratope_residues"][int(index) - 1]
        if paratope["aa"] != native:
            raise ValueError(f"Native mismatch for {reference_name} {token}")
        mutations[(paratope["chain"], int(paratope["resid"]))] = replacement

    sequences = []
    applied = set()
    for chain in reference["chains"]:
        residues = chain_residues(reference["pdb"], chain)
        for residue in residues:
            key = (chain, residue[0])
            if key in mutations:
                residue[1] = mutations[key]
                applied.add(key)
        sequences.append("".join(residue[1] for residue in residues))
    if applied != set(mutations):
        raise ValueError(f"Unapplied mutations: {sorted(set(mutations) - applied)}")
    peptide = "".join(
        residue[1] for residue in chain_residues(reference["pdb"], reference["peptide_chain"]))
    return sequences, peptide


def main():
    decision = json.loads((
        ROOT / "results/v5_1_candidates/abeta42_final/candidate_decision.json"
    ).read_text(encoding="utf-8"))
    output_dir = ROOT / "results/v5_1_candidates/abeta42_colabfold_candidates/inputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for candidate in decision["hypothesis_queue"]:
        reference_name = candidate["reference_pdb"]
        sequences, peptide = apply_mutations(reference_name, candidate["mutations"])
        parent_path = REFERENCES[reference_name]["a3m"]
        lines = parent_path.read_text(encoding="utf-8").splitlines()
        replacement_query = "".join(sequences) + peptide
        lengths = [len(sequence) for sequence in sequences] + [len(peptide)]
        expected_lengths = [int(value) for value in lines[0][1:].split("\t")[0].split(",")]
        if lengths != expected_lengths or len(lines[2]) != len(replacement_query):
            raise ValueError(
                f"A3M contract mismatch for {candidate['construct_id']}: "
                f"{lengths} != {expected_lengths}")
        lines[2] = replacement_query
        output = output_dir / f"{candidate['construct_id']}.a3m"
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest.append({
            "construct_id": candidate["construct_id"],
            "reference_pdb": reference_name,
            "mutations": candidate["mutations"],
            "chain_lengths": lengths,
            "epitope_sequence": peptide,
            "parent_a3m": str(parent_path.relative_to(ROOT)),
            "candidate_a3m": str(output.relative_to(ROOT)),
        })
    (output_dir.parent / "input_manifest.json").write_text(
        json.dumps({"candidates": manifest}, indent=2), encoding="utf-8")
    print(json.dumps({"prepared": len(manifest), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
