import numpy as np

from scripts.audit_idp_ensemble_expanded_structural_v2 import contact_metrics


def test_contact_metrics_detect_h3_antigen_contact():
    heavy = [{"atoms": np.array([[0.0, 0.0, 0.0]])}]
    antigen = [{"atoms": np.array([[0.0, 0.0, 4.0]])}]
    result = contact_metrics(heavy, antigen, [0], 4.5)
    assert result["contacting_h3_positions"] == [0]
    assert result["contact_pair_count"] == 1
