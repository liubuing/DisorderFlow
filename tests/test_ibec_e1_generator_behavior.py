import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "analyze_ibec_e1_generator_behavior.py"
    spec = importlib.util.spec_from_file_location("analyze_ibec_e1_generator_behavior", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_position_entropy_detects_diversity():
    mod = load_script()
    assert mod.position_entropy(["AAAA", "AAAA"]) == 0.0
    assert mod.position_entropy(["AAAA", "CCCC"]) > 0.0


def test_pairwise_hamming_is_normalized():
    mod = load_script()
    assert mod.hamming_fraction("AAAA", "AACC") == 0.5
    assert mod.mean_pairwise_hamming(["AAAA", "CCCC"]) == 1.0


def test_interpretation_reports_quality_tradeoff():
    mod = load_script()
    config = {"interpretation": {
        "diversity_advantage_requires_metrics": 2,
        "developability_noninferiority_margin": -0.05,
    }}
    differences = {
        "unique_fraction": {"mean_difference": 0.1, "ci95": [0.01, 0.2]},
        "mean_position_entropy_nats": {"mean_difference": 0.1, "ci95": [0.01, 0.2]},
        "mean_pairwise_hamming_fraction": {"mean_difference": 0.1, "ci95": [-0.01, 0.2]},
        "developability_pass_fraction_at_0_55": {
            "mean_difference": -0.2, "ci95": [-0.3, -0.1]
        },
    }
    result = mod.interpretation(config, differences)
    assert result["decision"] == "expanded_exploration_with_developability_tradeoff"
