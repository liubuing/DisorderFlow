import hashlib
import json

from scripts.validate_release_lineage import (
    SOURCE_ONLY_DOCUMENTS,
    collect_references,
    run_validation,
)


def test_collect_references_supports_structured_and_suffix_pairs():
    payload = {
        "structured": {"path": "a.json", "sha256": "a" * 64},
        "source": "b.json",
        "source_sha256": "b" * 64,
    }
    references = collect_references(payload)
    assert ("$.structured", "a.json", "a" * 64, "raw") in references
    assert ("$.source", "b.json", "b" * 64, "raw") in references


def test_canonical_lf_hash_survives_checkout_line_endings(tmp_path):
    from scripts.validate_release_lineage import sha256

    path = tmp_path / "document.md"
    path.write_bytes(b"first\r\nsecond\r\n")
    windows_hash = sha256(path, "canonical_lf")
    path.write_bytes(b"first\nsecond\n")
    assert sha256(path, "canonical_lf") == windows_hash


def test_recursive_validator_detects_stale_hash(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="ascii")
    document = tmp_path / "document.json"
    document.write_text(json.dumps({
        "input": {"path": "source.json", "sha256": "0" * 64}
    }), encoding="ascii")
    result = run_validation(tmp_path, ["document.json"])
    assert result["status"] == "invalid"
    assert "hash mismatch" in result["errors"][0]


def test_current_release_lineage_is_valid():
    result = run_validation(__import__("pathlib").Path(__file__).resolve().parents[1])
    assert result["errors"] == []


def test_source_only_release_lineage_is_valid():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    result = run_validation(root, SOURCE_ONLY_DOCUMENTS)
    assert result["errors"] == []
