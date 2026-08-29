import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "run_ibec_tau_candidates.py"
    spec = importlib.util.spec_from_file_location("run_ibec_tau_candidates", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_conservative_library_is_reproducible_and_cys_free():
    mod = load_script()
    first = mod.conservative_library("ARDYYGTSFAMDY", 50, 7301, 1, 4)
    second = mod.conservative_library("ARDYYGTSFAMDY", 50, 7301, 1, 4)
    assert first == second
    assert len(set(first)) == 50
    assert all("C" not in row for row in first)
    assert all(1 <= mod.hamming(row, "ARDYYGTSFAMDY") <= 4 for row in first)


def test_composition_shuffles_preserve_exact_composition():
    mod = load_script()
    rows = mod.composition_shuffles("ARDYYGTSFAMDY", 30, 7303)
    assert len(set(rows)) == 30
    assert all(sorted(row) == sorted("ARDYYGTSFAMDY") for row in rows)


def test_reference_h3_and_resid_window_are_frozen():
    mod = load_script()
    path = ROOT / "data/non_abeta_idp/ensembles_v1/tau/5MP3_tau_core_complex.pdb"
    heavy, resids = mod.chain_sequence_and_resids(path, "A")
    assert heavy[95:108] == "ARDYYGTSFAMDY"
    assert resids[95:108] == list(range(97, 110))
    assert heavy[94] == "C" and heavy[108] == "W"


def test_numpy_gate_comparisons_are_normalized_for_json():
    mod = load_script()
    checks = mod.normalize_gate_checks({
        "bfn_mean_noninferior": np.float64(0.1) >= np.float64(0.0),
    })
    assert json.loads(json.dumps(checks)) == {"bfn_mean_noninferior": True}
