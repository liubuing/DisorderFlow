import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_pre_af2_selection_preserves_frozen_slots_and_sources():
    selection_path = (
        ROOT / "results/multiscaffold_confirmatory_v2/pre_af2_selection.json")
    selection = json.loads(selection_path.read_text())
    config = yaml.safe_load((
        ROOT / "configs/benchmarks/multiscaffold_confirmatory_v2_selection.yml"
    ).read_text())
    raw_path = ROOT / config["raw_generation"]
    raw = json.loads(raw_path.read_text())
    source_by_id = {row["attempt_id"]: row for row in raw["attempts"]}

    assert selection["status"] == "pre_af2_selection_complete"
    assert selection["summary"] == {
        "expected_slots": 300,
        "recorded_slots": 300,
        "selected_candidates": 280,
        "failed_slots": 20,
        "selected_source_attempts_unique": 280,
    }
    assert selection["raw_generation_sha256"] == hashlib.sha256(
        raw_path.read_bytes()).hexdigest()
    assert len({row["selection_id"] for row in selection["selections"]}) == 300

    selected = [
        row for row in selection["selections"] if row["status"] == "selected"]
    failed = [
        row for row in selection["selections"] if row["status"] == "failed"]
    assert all(row["arm"] == "esm_if" for row in failed)
    assert all(row["reason"] == "insufficient_unique_candidates" for row in failed)
    for row in selected:
        source = source_by_id[row["source_attempt_id"]]
        assert source["status"] == "success"
        assert row["sequence"] == source["sequence"]
        assert row["full_heavy_sequence"] == source["full_heavy_sequence"]
        assert "generator_score" not in row


def test_selected_h3_sequences_are_unique_within_each_group():
    selection = json.loads((
        ROOT / "results/multiscaffold_confirmatory_v2/pre_af2_selection.json"
    ).read_text())
    groups = {}
    for row in selection["selections"]:
        if row["status"] == "selected":
            groups.setdefault((row["component_id"], row["arm"]), []).append(row)
    for rows in groups.values():
        assert len({row["sequence"] for row in rows}) == len(rows)
        assert len({row["source_attempt_id"] for row in rows}) == len(rows)
