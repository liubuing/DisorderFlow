import torch

from scripts.generate_statecontrast_v2_candidates import inject_h3, substitutions


def test_h3_injection_changes_only_design_region():
    batch = {
        "aa": torch.tensor([[9, 9, 9, 9]]),
        "design_region_flag": torch.tensor([[False, True, True, False]]),
    }
    output = inject_h3(batch, "AC")
    assert output["aa"].tolist() == [[9, 0, 1, 9]]
    assert batch["aa"].tolist() == [[9, 9, 9, 9]]


def test_substitution_count_is_exact():
    assert substitutions("AAAA", "ACCA") == 2
