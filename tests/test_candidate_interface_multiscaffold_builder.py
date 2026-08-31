import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts/build_candidate_interface_multiscaffold_v1.py"
    spec = importlib.util.spec_from_file_location(
        "build_candidate_interface_multiscaffold_v1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_split_is_disjoint_and_complete():
    runtime_path = ROOT / "data/candidate_interface_multiscaffold_v1/split_manifest.json"
    versioned_path = ROOT / "publication/candidate_interface_multiscaffold_v1_split_manifest.json"
    assert runtime_path.read_bytes() == versioned_path.read_bytes()
    split = json.loads(runtime_path.read_text(encoding="utf-8"))
    train = set(split["train"])
    calibration = set(split["calibration"])
    test = set(split["test"])
    assert len(train) == 15
    assert len(calibration) == 2
    assert len(test) == 3
    assert not train & calibration
    assert not train & test
    assert not calibration & test
    assert len(train | calibration | test) == 20


def test_smoke_pae_and_pdb_plddt_are_valid():
    mod = load_script()
    path = ROOT / "results/candidate_interface_multiscaffold_v1/smoke/results.json"
    if not path.exists():
        return
    results = json.loads(path.read_text(encoding="utf-8"))
    row = results["results"][0]
    plddt, sequences = mod.residue_plddt_and_sequences(ROOT / row["pdb"])
    pae = np.load(ROOT / row["pae_npz"], allow_pickle=False)["pae"]
    assert list(sequences) == ["A", "B", "C"]
    assert pae.shape == (len(plddt), len(plddt))
    assert np.isfinite(pae).all()
    assert np.ptp(pae) > 0
    assert abs(float(plddt.mean() / 100.0) - row["plddt"]) <= 5e-4
