import csv

import pytest
from statecontrast_experimental_data import (
    SCHEMA_VERSION,
    ExperimentalDataValidationError,
    load_binding_state_matrix_csv,
    load_hdx_peptide_uptake_csv,
)
from statecontrast_real_ensembles import (
    StructureValidationError,
    parse_pdb_ensemble,
    real_ensemble_mode_status,
)


def _atom(serial, atom, residue, chain, residue_number, x, model_offset=0):
    return (
        f"ATOM  {serial:5d} {atom:^4s} {residue:>3s} {chain:1s}{residue_number:4d}    "
        f"{x + model_offset:8.3f}{2.0:8.3f}{3.0:8.3f}{1.0:6.2f}{20.0:6.2f}"
        f"          {atom[0]:>2s}  \n"
    )


def _multimodel_pdb():
    lines = ["SEQRES   1 A    2  ALA GLY\n", "MODEL        1\n"]
    lines.extend([
        _atom(1, "N", "ALA", "A", 1, 1.0),
        _atom(2, "CA", "ALA", "A", 1, 1.5),
        _atom(3, "CA", "GLY", "A", 2, 2.5),
        _atom(4, "CA", "TYR", "H", 10, 4.0),
        "ENDMDL\n",
        "MODEL        2\n",
        _atom(5, "N", "ALA", "A", 1, 1.0, 10),
        _atom(6, "CA", "ALA", "A", 1, 1.5, 10),
        _atom(7, "CA", "GLY", "A", 2, 2.5, 10),
        _atom(8, "CA", "TYR", "H", 10, 4.0, 10),
        "ENDMDL\n",
    ])
    return "".join(lines)


def test_pdb_adapter_parses_all_models_and_audits_selected_chain(tmp_path):
    path = tmp_path / "ensemble.pdb"
    path.write_text(_multimodel_pdb(), encoding="utf-8")

    ensemble = parse_pdb_ensemble(path, chain="A", expected_sequence="AG")

    assert ensemble["selected_models"] == [1, 2]
    assert ensemble["selected_chains"] == ["A"]
    assert ensemble["conformer_count"] == 2
    assert ensemble["contacts"] is None
    audit = ensemble["conformers"][0]["chains"][0]["sequence_audit"]
    assert audit["observed_sequence"] == "AG"
    assert audit["seqres_coverage"] == 1.0
    assert audit["expected_sequence_exact"] is True


def test_pdb_adapter_selects_model_and_fails_for_invalid_structure(tmp_path):
    path = tmp_path / "ensemble.pdb"
    path.write_text(_multimodel_pdb(), encoding="utf-8")
    assert parse_pdb_ensemble(path, model=2, chain="A")["selected_models"] == [2]

    invalid = tmp_path / "invalid.pdb"
    invalid.write_text("HEADER    NO COORDINATES\n", encoding="utf-8")
    with pytest.raises(StructureValidationError, match="no recognized protein coordinates"):
        parse_pdb_ensemble(invalid)


def test_geometry_readiness_blocks_contacts_without_explicit_antibody_pose(tmp_path):
    path = tmp_path / "ensemble.pdb"
    path.write_text(_multimodel_pdb(), encoding="utf-8")
    config = {"negative_states": {"mode": "structure_ensemble", "ensembles": [{
        "name": "state", "pdb": path.name, "chain": "A", "state_type": "fibril"
    }]}}

    status = real_ensemble_mode_status(config, tmp_path)

    assert status["geometry_status"] == "geometry_ready"
    assert status["contact_scoring_status"] == "blocked_without_pose"
    assert status["status"] == "geometry_ready_but_contact_scoring_blocked_without_pose"
    assert status["entries"][0]["geometry"]["contacts"] is None


def _write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_binding_state_matrix_preserves_states_replicates_units_and_censoring(tmp_path):
    path = tmp_path / "binding.csv"
    fields = [
        "schema_version", "experiment_id", "assay_type", "binder_id", "state_id",
        "replicate_id", "metric", "value", "unit", "censoring", "quality_flag",
        "evidence_tier",
    ]
    base = {
        "schema_version": SCHEMA_VERSION, "experiment_id": "exp-1", "assay_type": "SPR",
        "binder_id": "ab-1", "replicate_id": "r1", "metric": "kd", "unit": "nM",
        "quality_flag": "pass", "evidence_tier": "tier_1_primary",
    }
    _write_csv(path, fields, [
        {**base, "state_id": "monomer", "value": "2.5", "censoring": "none"},
        {**base, "state_id": "fibril", "value": "100", "censoring": "right"},
    ])

    rows = load_binding_state_matrix_csv(path)

    assert [row["state_id"] for row in rows] == ["monomer", "fibril"]
    assert rows[0]["value"] == 2.5
    assert rows[1]["censoring"] == "right"
    assert rows[1]["unit"] == "nM"


def test_hdx_loader_validates_peptide_uptake_and_rejects_made_up_units(tmp_path):
    path = tmp_path / "hdx.csv"
    fields = [
        "schema_version", "experiment_id", "protein_id", "state_id", "peptide_id",
        "peptide_start", "peptide_end", "timepoint", "timepoint_unit", "uptake_value",
        "uptake_unit", "replicate_id", "censoring", "quality_flag", "evidence_tier",
    ]
    row = {
        "schema_version": SCHEMA_VERSION, "experiment_id": "hdx-1", "protein_id": "p1",
        "state_id": "bound", "peptide_id": "pep-1", "peptide_start": "3",
        "peptide_end": "8", "timepoint": "30", "timepoint_unit": "s",
        "uptake_value": "1.4", "uptake_unit": "Da", "replicate_id": "r2",
        "censoring": "none", "quality_flag": "warning",
        "evidence_tier": "tier_2_processed",
    }
    _write_csv(path, fields, [row])
    assert load_hdx_peptide_uptake_csv(path)[0]["peptide_start"] == 3

    row["uptake_unit"] = "invented"
    _write_csv(path, fields, [row])
    with pytest.raises(ExperimentalDataValidationError, match="invalid uptake_unit"):
        load_hdx_peptide_uptake_csv(path)
