import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts/summarize_candidate_interface_multiseed.py"
    spec = importlib.util.spec_from_file_location(
        "summarize_candidate_interface_multiseed", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dispersion_ratios_distinguish_std_and_variance():
    module = load_script()
    std_ratio, variance_ratio = module.dispersion_ratios(1.0, 2.0)
    assert std_ratio == pytest.approx(0.5)
    assert variance_ratio == pytest.approx(0.25)


def test_pair_bootstrap_resamples_scaffolds_and_entities():
    module = load_script()
    records = []
    for scaffold, predictions in (("A", [0.1, 0.2, 0.3]),
                                  ("B", [0.3, 0.2, 0.1])):
        for index, prediction in enumerate(predictions):
            records.append({
                "scaffold_family": scaffold,
                "target_plddt": float(index),
                "pred_plddt": prediction,
                "noise_plddt": 0.0,
            })
    result = module.pair_statistics(records, "plddt", bootstrap_seed=17)
    assert result["pairs"] == 6
    assert result["contributing_scaffolds"] == 2
    assert result["max_scaffold_pair_fraction"] == pytest.approx(0.5)
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["bootstrap_unit"] == "scaffold_then_entity"
    assert 0.0 <= result["bootstrap_95_lower"] <= 1.0


def test_selection_key_treats_undefined_correlation_as_failure():
    module = load_script()
    evaluation = {"metrics": {}}
    for name in module.OUTPUTS:
        evaluation["metrics"][name] = {
            "entity_aggregated": {
                "within_scaffold": {"mean_group_spearman": None},
                "mae": 0.0,
            },
        }
    key = module.selection_key(evaluation)
    assert key[0] == float("-inf")


def test_pair_evidence_requires_multiple_scaffolds_without_concentration():
    module = load_script()
    assert module.MIN_RELIABLE_PAIRS == 30
    assert module.MIN_PAIR_SCAFFOLDS == 3
    assert module.MAX_SCAFFOLD_PAIR_FRACTION == pytest.approx(0.5)

    records = []
    for scaffold, size in (("A", 9), ("B", 7), ("C", 6)):
        for index in range(size):
            records.append({
                "scaffold_family": scaffold,
                "target_iptm": float(index),
                "pred_iptm": float(index),
                "noise_iptm": 0.0,
            })
    result = module.pair_statistics(records, "iptm", bootstrap_seed=23)
    assert result["pairs"] >= module.MIN_RELIABLE_PAIRS
    assert result["contributing_scaffolds"] == 3
    assert result["max_scaffold_pair_fraction"] <= 0.5
