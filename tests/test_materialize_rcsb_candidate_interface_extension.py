import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "build" / "materialize_rcsb_candidate_interface_extension.py"
SPEC = importlib.util.spec_from_file_location("materialize_rcsb_extension", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_entity(entity_id, description, length, chains, polymer_type="Protein"):
    return {
        "rcsb_id": entity_id,
        "rcsb_polymer_entity": {"pdbx_description": description},
        "entity_poly": {
            "rcsb_entity_polymer_type": polymer_type,
            "rcsb_sample_sequence_length": length,
            "pdbx_seq_one_letter_code_can": "A" * length,
        },
        "rcsb_polymer_entity_container_identifiers": {
            "entity_id": "1",
            "asym_ids": chains,
            "auth_asym_ids": chains,
        },
        "rcsb_cluster_membership": [],
    }


def test_role_chains_splits_auth_chain_ids():
    entry = {
        "struct": {"title": "Fab in complex with peptide"},
        "polymer_entities": [
            make_entity("X_1", "antibody heavy chain", 120, ["E"]),
            make_entity("X_2", "antibody light chain", 110, ["D"]),
            make_entity("X_3", "MAGE-A4 peptide", 10, ["C"]),
            make_entity("X_4", "MHC class I alpha", 280, ["A"]),
            make_entity("X_5", "beta-2 microglobulin", 100, ["B"]),
        ],
    }
    heavy, light, peptide = MODULE.role_chains(entry, "Fab in complex with peptide")
    assert heavy == ["E"]
    assert light == ["D"]
    assert peptide == ["C"]


def test_role_chains_ignores_nonprotein_peptides():
    entry = {
        "struct": {"title": ""},
        "polymer_entities": [
            make_entity("X_1", "antibody heavy chain", 120, ["H"]),
            make_entity("X_2", "antibody light chain", 110, ["L"]),
            make_entity("X_3", "RNA aptamer", 24, ["P"], polymer_type="RNA"),
        ],
    }
    heavy, light, peptide = MODULE.role_chains(entry, "")
    assert heavy == ["H"]
    assert light == ["L"]
    assert peptide == []


def test_min_distance_basic_and_empty():
    left = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    right = np.array([[3.0, 0.0, 0.0]])
    assert MODULE.min_distance(left, right) == pytest.approx(3.0)
    assert MODULE.min_distance(np.zeros((0, 3)), right) == float("inf")


def test_min_distance_blocks_large_arrays():
    left = np.tile([[0.0, 0.0, 0.0]], (1000, 1))
    right = np.tile([[1.0, 0.0, 0.0]], (1000, 1))
    assert MODULE.min_distance(left, right, block=64) == pytest.approx(1.0)


def test_combo_rank_prefers_more_h3_contacts():
    weaker = {
        "n_contacting_h3_positions": 1,
        "n_h3_antigen_residue_contacts": 2,
        "minimum_h3_antigen_distance": 3.0,
        "minimum_antibody_peptide_distance": 3.0,
        "heavy_chain_id": "A", "light_chain_id": "B", "peptide_chain_id": "C",
    }
    stronger = dict(weaker, n_contacting_h3_positions=3, heavy_chain_id="Z")
    assert MODULE.combo_rank(stronger) < MODULE.combo_rank(weaker)


def test_combo_rank_breaks_ties_deterministically():
    base = {
        "n_contacting_h3_positions": 2,
        "n_h3_antigen_residue_contacts": 4,
        "minimum_h3_antigen_distance": 3.5,
        "minimum_antibody_peptide_distance": 3.0,
        "heavy_chain_id": "A", "light_chain_id": "B", "peptide_chain_id": "C",
    }
    other = dict(base, heavy_chain_id="B")
    assert MODULE.combo_rank(base) < MODULE.combo_rank(other)
    none_distance = dict(base, minimum_h3_antigen_distance=None)
    assert MODULE.combo_rank(none_distance) > MODULE.combo_rank(base)


def test_materialize_refuses_existing_manifest(tmp_path):
    discovery = tmp_path / "discovery.json"
    discovery.write_text(json.dumps({"classification": "metadata_only"}))
    output = tmp_path / "structures"
    (output / "pdb").mkdir(parents=True)
    (output / "structural_manifest.json").write_text("sentinel")
    with pytest.raises(FileExistsError):
        MODULE.materialize(discovery, output)


def test_materialize_allows_resume_without_manifest(tmp_path):
    discovery = tmp_path / "discovery.json"
    discovery.write_text(json.dumps({
        "classification": "metadata_only",
        "entries": [],
        "snapshot_dir": "data/does-not-exist",
    }))
    output = tmp_path / "structures"
    (output / "pdb").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        MODULE.materialize(discovery, output)
