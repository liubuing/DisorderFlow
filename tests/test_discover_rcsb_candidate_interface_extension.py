import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "build" / "discover_rcsb_candidate_interface_extension.py"
SPEC = importlib.util.spec_from_file_location("discover_rcsb_extension", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_entry(pdb_id, entities, title="", method="X-RAY DIFFRACTION", resolution=2.0):
    return {
        "rcsb_id": pdb_id,
        "struct": {"title": title},
        "rcsb_accession_info": {"initial_release_date": "2026-01-01T00:00:00Z"},
        "exptl": [{"method": method}],
        "rcsb_entry_info": {"resolution_combined": [resolution]},
        "polymer_entities": entities,
    }


def make_entity(entity_id, description, length, polymer_type="Protein"):
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
            "asym_ids": ["A"],
            "auth_asym_ids": ["A"],
        },
        "rcsb_cluster_membership": [],
    }


def write_snapshot(tmp_path, entries):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    entries_payload = json.dumps({"entries": entries}, indent=2).encode("ascii")
    (snapshot / "entries.json").write_bytes(entries_payload)
    import hashlib
    manifest = {
        "schema_version": "candidate_interface_extension_rcsb_snapshot_v1",
        "classification": "metadata_only",
        "files": {"entries.json": {
            "byte_count": len(entries_payload),
            "sha256": hashlib.sha256(entries_payload).hexdigest(),
        }},
    }
    (snapshot / "acquisition.json").write_text(json.dumps(manifest), encoding="ascii")
    return snapshot


NOW = lambda: datetime(2026, 9, 1, tzinfo=timezone.utc)


def test_classify_entity_roles():
    assert MODULE.classify_entity(make_entity("X_1", "Fab heavy chain", 120), "") == "heavy"
    assert MODULE.classify_entity(make_entity("X_2", "Fab light chain", 110), "") == "light"
    assert MODULE.classify_entity(make_entity("X_3", "VHH nanobody", 118), "") == "heavy"
    assert MODULE.classify_entity(make_entity("X_4", "some kinase", 300), "") == "unclassified"


def test_peptide_candidate_discovered(tmp_path):
    entities = [
        make_entity("X_1", "antibody heavy chain", 120),
        make_entity("X_2", "antibody light chain", 110),
        make_entity("X_3", "designed peptide", 18),
    ]
    snapshot = write_snapshot(tmp_path, [make_entry("1ABC", entities, "Fab in complex with peptide")])
    artifact = MODULE.discover(snapshot, tmp_path / "discovery.json", now=NOW)
    entry = artifact["entries"][0]
    assert entry["status"] == "candidate"
    assert entry["peptide_antigens"][0]["length"] == 18
    assert artifact["counts"]["antibody_peptide_candidates"] == 1


def test_viral_antigen_deferred_not_excluded(tmp_path):
    entities = [
        make_entity("X_1", "antibody heavy chain", 120),
        make_entity("X_2", "antibody light chain", 110),
        make_entity("X_3", "influenza hemagglutinin peptide", 15),
    ]
    snapshot = write_snapshot(tmp_path, [make_entry("2DEF", entities, "Fab bound to viral peptide")])
    artifact = MODULE.discover(snapshot, tmp_path / "discovery.json", now=NOW)
    entry = artifact["entries"][0]
    assert entry["status"] == "deferred_viral_antigen"
    assert artifact["counts"]["deferred_viral_antigen"] == 1
    assert artifact["counts"]["antibody_peptide_candidates"] == 0


def test_nonviral_protein_antigen_excluded(tmp_path):
    entities = [
        make_entity("X_1", "antibody heavy chain", 120),
        make_entity("X_2", "antibody light chain", 110),
        make_entity("X_3", "human serum albumin", 585),
    ]
    snapshot = write_snapshot(tmp_path, [make_entry("3GHI", entities, "Fab bound to albumin")])
    artifact = MODULE.discover(snapshot, tmp_path / "discovery.json", now=NOW)
    entry = artifact["entries"][0]
    assert entry["status"] == "excluded"
    assert "no_peptide_antigen_5_50" in entry["reasons"]


def test_discovery_refuses_overwrite(tmp_path):
    snapshot = write_snapshot(tmp_path, [])
    output = tmp_path / "discovery.json"
    output.write_text("sentinel")
    with pytest.raises(FileExistsError):
        MODULE.discover(snapshot, output, now=NOW)
    assert output.read_text() == "sentinel"
