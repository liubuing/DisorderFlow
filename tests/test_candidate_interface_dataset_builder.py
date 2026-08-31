import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts/build_candidate_interface_confidence_v1.py"
    spec = importlib.util.spec_from_file_location(
        "build_candidate_interface_confidence_v1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_variable_positions_are_shared_scaffold_masks():
    mod = load_script()
    constructs, manifest = mod.load_constructs()
    contracts = mod.family_contracts(constructs, manifest)
    assert contracts["8B9V"]["antigen_chain"] == "C"
    assert contracts["5MP3"]["antigen_chain"] == "C"
    assert contracts["3STB"]["candidate_positions"]["A"] == (
        list(range(26, 34)) + list(range(51, 59)) + list(range(97, 114)))
    assert all(contracts["8B9V"]["candidate_positions"].values())
    assert all(contracts["5MP3"]["candidate_positions"].values())


def test_frozen_bundle_discovery_is_scaffold_disjoint():
    mod = load_script()
    constructs, _ = mod.load_constructs()
    train = mod.discover_bundles(mod.TRAIN_ROOTS, constructs)
    validation = mod.discover_bundles(mod.VAL_ROOTS, constructs)
    assert {row["family"] for row in train} == {"8B9V", "3STB"}
    assert {row["family"] for row in validation} == {"5MP3"}
    assert {row["source_identity"] for row in train}.isdisjoint(
        {row["source_identity"] for row in validation})
