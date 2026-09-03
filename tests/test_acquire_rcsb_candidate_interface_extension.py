import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "build" / "acquire_rcsb_candidate_interface_extension.py"
SPEC = importlib.util.spec_from_file_location("acquire_rcsb_extension", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_result_ids_accepts_compact_response():
    response = {"total_count": 2, "result_set": ["1ABC", "2DEF"]}
    assert MODULE.result_ids(response) == ["1ABC", "2DEF"]


def test_result_ids_accepts_verbose_response():
    response = {
        "total_count": 2,
        "result_set": [{"identifier": "1abc"}, {"identifier": "2def"}],
    }
    assert MODULE.result_ids(response) == ["1ABC", "2DEF"]


def test_result_ids_rejects_truncated_response():
    with pytest.raises(ValueError, match="truncated"):
        MODULE.result_ids({"total_count": 3, "result_set": ["1ABC"]})


def test_acquire_rejects_changed_frozen_query(tmp_path):
    query = tmp_path / "query.json"
    protocol = tmp_path / "protocol.json"
    query.write_text("{}\n", encoding="ascii")
    protocol.write_text(json.dumps({
        "discovery_contract": {"query_config_sha256": "not-the-query-hash"},
    }), encoding="ascii")

    with pytest.raises(ValueError, match="Frozen RCSB query hash"):
        MODULE.acquire(query, protocol, tmp_path / "snapshot")


def test_acquire_rejects_missing_protocol_section(tmp_path):
    import hashlib
    query = tmp_path / "query.json"
    protocol = tmp_path / "protocol.json"
    query.write_text("{}\n", encoding="ascii")
    protocol.write_text(json.dumps({
        "discovery_round_v2": {
            "query_config_sha256": hashlib.sha256(b"{}\n").hexdigest(),
        },
    }), encoding="ascii")

    with pytest.raises(ValueError, match="Protocol lacks section"):
        MODULE.acquire(query, protocol, tmp_path / "snapshot")
