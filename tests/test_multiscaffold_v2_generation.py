import json
from pathlib import Path

from Bio.PDB import PDBParser

from scripts.generate_multiscaffold_confirmatory_v2 import (
    attempt_rows,
    canonicalize_structure,
    full_heavy,
)

ROOT = Path(__file__).resolve().parents[1]


def first_component():
    holdout = json.loads((
        ROOT / "data/multiscaffold_confirmatory_v2/holdout_manifest.json"
    ).read_text())
    return holdout["components"][0]


def test_canonical_structure_has_contiguous_role_chains(tmp_path):
    component = first_component()
    representative = component["representative"]
    output = tmp_path / "canonical.pdb"
    canonicalize_structure(representative, output)
    model = PDBParser(QUIET=True).get_structure("canonical", output)[0]
    expected_lengths = {
        "H": len(representative["heavy_sequence"]),
        "L": len(representative["light_sequence"]),
        "P": len(representative["antigen_sequence"]),
    }
    assert {chain.id for chain in model} == {"H", "L", "P"}
    for chain_id, expected_length in expected_lengths.items():
        residues = list(model[chain_id].get_residues())
        assert len(residues) == expected_length
        assert [residue.id[1] for residue in residues] == list(
            range(1, expected_length + 1))


def test_attempt_rows_preserve_every_failed_slot():
    component = first_component()
    rows = attempt_rows(
        component, "proteinmpnn", 2609, 8, [], "generation failed",
        {"input_cif_sha256": "a", "canonical_pdb_sha256": "b",
         "checkpoint_or_model_sha256": "c"}, 1.25)
    assert len(rows) == 8
    assert len({row["attempt_id"] for row in rows}) == 8
    assert all(row["status"] == "failed" for row in rows)
    assert all(row["error"] == "generation failed" for row in rows)


def test_full_heavy_changes_only_frozen_h3_positions():
    representative = first_component()["representative"]
    indices = representative["h3_heavy_indices_zero_based"]
    replacement = "A" * len(indices)
    designed = full_heavy(representative["heavy_sequence"], indices, replacement)
    assert "".join(designed[index] for index in indices) == replacement
    assert all(
        designed[index] == amino_acid
        for index, amino_acid in enumerate(representative["heavy_sequence"])
        if index not in set(indices))
