import json
from pathlib import Path

import pytest
import torch
import yaml
from Bio.PDB import MMCIFParser

from disorderflow.utils.data import PaddingCollate
from scripts.build.materialize_successor_v3_confirmatory import (
    contact_metrics,
    resolved_residues,
    write_canonical_pdb,
)
from scripts.evaluate_successor_v3_confirmatory import record_batch


ROOT = Path(__file__).resolve().parents[1]


def test_protocol_freezes_new_seeds_thresholds_and_one_shot_policy():
    protocol = yaml.safe_load((
        ROOT / "configs/benchmarks/successor_v3_confirmatory.yml"
    ).read_text(encoding="utf-8"))
    assert protocol["status"] == "frozen_before_live_snapshot_acquisition"
    assert protocol["primary_endpoint"]["seeds"] == [4103, 4111, 4127]
    assert protocol["isolation"]["minimum_identity"] == {
        "vh": 0.9, "vl": 0.9, "paired_cdr": 0.7, "h3": 0.5,
        "antigen": 0.3,
    }
    assert protocol["execution_policy"]["confirmatory_evaluation_attempts"] == 1
    assert protocol["execution_policy"]["insufficient_components_action"].startswith(
        "freeze_feasibility_failure")


def test_contact_metric_and_canonical_batch_path_on_frozen_exposed_structure(tmp_path):
    holdout = json.loads((
        ROOT / "data/multiscaffold_confirmatory_v2/holdout_manifest.json"
    ).read_text(encoding="utf-8"))
    record = dict(holdout["components"][0]["representative"])
    parser = MMCIFParser(QUIET=True, auth_chains=True, auth_residues=True)
    model = next(parser.get_structure(
        record["instance"], str(ROOT / record["cif_path"])).get_models())
    metrics = contact_metrics(
        resolved_residues(model[record["heavy_chain"]]),
        resolved_residues(model[record["antigen_chain"]]),
        record["h3_heavy_indices_zero_based"],
    )
    assert metrics["n_contacting_h3_positions"] >= 1

    pdb_path = tmp_path / "canonical.pdb"
    write_canonical_pdb(pdb_path, [
        ("H", model[record["heavy_chain"]]),
        ("L", model[record["light_chain"]]),
        ("P", model[record["antigen_chain"]]),
    ])
    record.update({
        "pdb_path": str(pdb_path),
        "heavy_chain": "H",
        "light_chain": "L",
        "antigen_chain": "P",
    })
    raw = record_batch(record)
    batch = PaddingCollate()([raw, raw])
    assert batch["aa"].shape[0] == 2
    assert batch["generate_flag"].any(dim=1).tolist() == [True, True]
    antigen_fragment = 2
    assert torch.all(batch["fragment_type"][batch["chain_id"] == "P"] == antigen_fragment)


def test_confirmatory_evaluator_refuses_existing_attempt(tmp_path, monkeypatch):
    from scripts import evaluate_successor_v3_confirmatory as evaluator

    output = tmp_path / "analysis.json"
    output.with_suffix(".attempt.json").write_text("{}", encoding="ascii")
    monkeypatch.setattr(
        "sys.argv", ["evaluate", "--panel", "unused.json", "--output", str(output)])
    with pytest.raises(FileExistsError, match="one-shot"):
        evaluator.main()


def test_current_snapshot_stops_before_model_access_when_pdb_upper_bound_is_too_small():
    decision = json.loads((
        ROOT / "publication/successor_v3_confirmatory_feasibility_decision.json"
    ).read_text(encoding="utf-8"))
    assert decision["status"] == "metadata_feasibility_gate_failed"
    assert decision["discovery"]["peptide_candidate_pdbs"] == 9
    assert decision["frozen_gate"] == {
        "minimum_reference_independent_components": 12,
        "maximum_possible_exact_pdb_independent_components": 9,
        "passed": False,
    }
    assert "without structure download" in decision["decision"]
    assert "neither supports nor rejects" in decision["claim_boundary"]
