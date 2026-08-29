import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "run_abeta_4hix_prospective.py"
    spec = importlib.util.spec_from_file_location("run_abeta_4hix_prospective", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_conservative_library_is_reproducible_and_bounded():
    mod = load_script()
    first = mod.conservative_library("VRYDHYSGSSDY", 40, 17)
    second = mod.conservative_library("VRYDHYSGSSDY", 40, 17)
    assert first == second
    assert len(set(first)) == 40
    assert all(1 <= mod.hamming(row, "VRYDHYSGSSDY") <= 4 for row in first)
    assert all("C" not in row for row in first)


def test_composition_controls_preserve_exact_composition():
    mod = load_script()
    rows = mod.composition_shuffles("VRYDHYSGSSDY", 20, 23)
    assert len(set(rows)) == 20
    assert all(sorted(row) == sorted("VRYDHYSGSSDY") for row in rows)


def test_candidate_batch_expansion_is_exact_and_independent():
    import torch

    from modules.bfn_prospective import inject_candidate_sequences

    batch = {
        "aa": torch.tensor([[9, 9, 9, 9]]),
        "generate_flag": torch.tensor([[False, True, False, True]]),
        "mask": torch.ones(1, 4, dtype=torch.bool),
        "chain_id": [("H", "H", "H", "H")],
    }
    expanded = inject_candidate_sequences(batch, ["AC", "WY"])
    assert torch.equal(expanded["aa"], torch.tensor([[9, 0, 9, 1], [9, 18, 9, 19]]))
    assert expanded["mask"].shape == (2, 4)
    assert len(expanded["chain_id"]) == 2


def test_diverse_shortlist_enforces_pairwise_distance():
    mod = load_script()
    config = {"selection": {"minimum_pairwise_hamming": 2, "shortlist_maximum": 3}}
    rows = [
        {"sequence": "AAAA", "all_gates_pass": True,
         "scoring": {"mean_complex_minus_stripped": 3.0, "std_complex_minus_stripped": 0.1}},
        {"sequence": "AAAT", "all_gates_pass": True,
         "scoring": {"mean_complex_minus_stripped": 2.0, "std_complex_minus_stripped": 0.1}},
        {"sequence": "AATT", "all_gates_pass": True,
         "scoring": {"mean_complex_minus_stripped": 1.0, "std_complex_minus_stripped": 0.1}},
    ]
    selected = mod.diverse_shortlist(config, rows)
    assert [row["sequence"] for row in selected] == ["AAAA", "AATT"]
