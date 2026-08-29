import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "audit_ibec_e2_idp_evidence.py"
    spec = importlib.util.spec_from_file_location("audit_ibec_e2_idp_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config():
    return {
        "classification": {
            "fully_disordered_minimum_fraction": 0.7,
            "partially_disordered_minimum_fraction": 0.3,
            "idr_minimum_length": 20,
            "source_label_semantics": {
                "ebi_mobidb_lite": "prediction_only_disorder",
                "af2_plddt_fallback": "conformational_proxy_only",
                "folded": "no_disorder_evidence",
            },
        }
    }


def test_interval_metrics_merge_overlapping_regions():
    mod = load_script()
    metrics = mod.interval_metrics([(1, 10), (5, 20)], 100)
    assert metrics["disorder_residues"] == 20
    assert metrics["disorder_fraction"] == 0.2
    assert metrics["longest_disorder_region"] == 16


def test_exact_disprot_regions_receive_experimental_tiers():
    mod = load_script()
    source = {"sequence": "A" * 100, "disorder_source": "folded"}
    entry = {"disprot_id": "DP1", "acc": "P1", "regions": [{
        "term_namespace": "Structural state", "term_name": "disorder",
        "start": 1, "end": 75,
    }]}
    result = mod.classify(config(), source, [entry])
    assert result["evidence_tier"] == "experimentally_curated_fully_disordered"


def test_prediction_and_proxy_never_upgrade_to_experimental():
    mod = load_script()
    predicted = mod.classify(
        config(), {"sequence": "A" * 50, "disorder_source": "ebi_mobidb_lite"}, []
    )
    proxy = mod.classify(
        config(), {"sequence": "A" * 50, "disorder_source": "af2_plddt_fallback"}, []
    )
    assert predicted["evidence_tier"] == "prediction_only_disorder"
    assert proxy["evidence_tier"] == "conformational_proxy_only"
