import gzip
import json
import pickle

import lmdb
import pytest

from disorderflow.data_factory import build_dataset, generate_counterfactuals, verify_dataset


def sample_raw(pdb_id="1abc"):
    return {
        "pdb_id": pdb_id,
        "vh_sequence": "QVQLVESCARDRGFDYWGGG",
        "vl_sequence": "DIQMTQSPSS",
        "cdr_h3_sequence": "CARDRGFDYW",
        "antigen_sequence": "DAEFRHDSGYE",
    }


def write_phase3(path, records):
    env = lmdb.open(str(path), map_size=1024 * 1024, subdir=False)
    identifiers = []
    with env.begin(write=True) as txn:
        for record in records:
            identifier = record["pdb_id"]
            identifiers.append(identifier)
            txn.put(identifier.encode(), pickle.dumps(record))
    env.close()
    with open(str(path) + "-ids", "wb") as handle:
        pickle.dump(identifiers, handle)


def source(path, split="train", name="phase3_test"):
    return {
        "name": name,
        "type": "phase3_lmdb",
        "path": path.name,
        "split": split,
        "release": "test-v1",
        "license": "test-only",
        "evidence_tier": "experimental_structure",
        "usage": split,
    }


def config(sources):
    return {
        "dataset_name": "test_factory",
        "seed": 17,
        "shard_size": 5,
        "counterfactuals": {"operations": [
            "cdr_single_mutation", "cdr_composition_shuffle",
            "antigen_composition_shuffle",
        ]},
        "sources": sources,
    }


def test_counterfactuals_are_matched_but_not_false_nonbinder_labels():
    from disorderflow.data_factory import make_observed_record

    observed = make_observed_record(sample_raw(), source(type("P", (), {"name": "x"})()), "train")
    variants = generate_counterfactuals(observed, [
        "cdr_single_mutation", "cdr_composition_shuffle",
        "antigen_composition_shuffle",
    ], seed=3)

    assert len(variants) == 3
    assert all(item["group_id"] == observed["group_id"] for item in variants)
    assert all(item["targets"]["binding_label"] is None for item in variants)
    assert all(item["evidence"]["tier"] == "synthetic_weak" for item in variants)
    assert variants[0]["antibody"]["cdr_h3"] != observed["antibody"]["cdr_h3"]


def test_factory_build_is_deterministic_and_verifiable(tmp_path):
    database = tmp_path / "train.lmdb"
    write_phase3(database, [sample_raw("1abc"), sample_raw("2def")])
    settings = config([source(database)])
    first = tmp_path / "first"
    second = tmp_path / "second"

    manifest_a = build_dataset(settings, tmp_path, first)
    manifest_b = build_dataset(settings, tmp_path, second)
    audit = verify_dataset(first)

    assert manifest_a["records"] == 8
    assert manifest_a["groups"] == 2
    assert manifest_a["canonical_fingerprint"] == manifest_b["canonical_fingerprint"]
    assert [item["sha256"] for item in manifest_a["shards"]] == [
        item["sha256"] for item in manifest_b["shards"]]
    assert audit["status"] == "pass"
    assert audit["lineage_errors"] == 0
    with gzip.open(first / manifest_a["shards"][0]["path"], "rt", encoding="ascii") as handle:
        record = json.loads(next(handle))
    assert record["lineage"]["operation"] == "observed_bound"


def test_factory_deduplicates_source_records_before_counterfactual_expansion(tmp_path):
    database = tmp_path / "train.lmdb"
    records = [sample_raw("same"), sample_raw("same")]
    write_phase3(database, records)
    manifest = build_dataset(config([source(database)]), tmp_path, tmp_path / "output")

    assert manifest["groups"] == 1
    assert manifest["records"] == 4
    assert manifest["duplicate_source_records_skipped"] == 1


def test_factory_rejects_external_evaluation_sources(tmp_path):
    database = tmp_path / "train.lmdb"
    write_phase3(database, [sample_raw()])
    forbidden = source(database, name="caid2_binding")
    with pytest.raises(ValueError, match="External evaluation source"):
        build_dataset(config([forbidden]), tmp_path, tmp_path / "output")


def test_all_six_imgt_cdrs_generate_matched_chain_consistent_counterfactuals():
    from disorderflow.data_factory import make_observed_record

    cdrs = {
        "H1": "GFTFSSYT", "H2": "ISSGGAYT", "H3": "CARDRGFDYW",
        "L1": "QSVSSSY", "L2": "GAS", "L3": "LQIYNMPIT",
    }
    raw = {
        "pdb_id": "six-cdr",
        "vh_sequence": "QVQLGFTFSSYTWVRQISSGGAYTKGRFCARDRGFDYWGQGT",
        "vl_sequence": "DIQMQSVSSSYWYQQGASRFSGLQIYNMPITFGQG",
        "cdr_h3_sequence": cdrs["H3"],
        "cdrs": cdrs,
        "antigen_sequence": "DAEFRHDSGYE",
        "cdr_annotation": {"scheme": "IMGT", "source": "official_precomputed"},
    }
    metadata = source(type("P", (), {"name": "x"})())
    metadata["require_all_six_cdrs"] = True
    observed = make_observed_record(raw, metadata, "train", "statecontrast_factory_v2")
    variants = generate_counterfactuals(observed, [
        "cdr_single_mutation_each", "cdr_composition_shuffle_each",
        "antigen_composition_shuffle",
    ], seed=9)

    mutation_variants = [
        item for item in variants if item["lineage"]["operation"].startswith("cdr_single")]
    assert observed["antibody"]["cdr_annotation"]["scheme"] == "IMGT"
    assert set(observed["antibody"]["cdrs"]) == set(cdrs)
    assert len(mutation_variants) == 6
    for variant in mutation_variants:
        cdr_name = variant["lineage"]["operation_detail"]["cdr"]
        chain = "vh" if cdr_name.startswith("H") else "vl"
        assert variant["antibody"]["cdrs"][cdr_name] in variant["antibody"][chain]
        assert variant["antibody"][chain] != observed["antibody"][chain]
