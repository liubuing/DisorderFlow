import types

import pytest
import torch

from disorderflow.utils.protein.constants import Fragment
from modules.bfn_loader import (
    _score_exact_candidate_confidence,
    _validate_candidate_interface_checkpoint,
    build_region_batch,
    parse_region_spec,
)


def test_generic_complex_assigns_explicit_antigen_chain():
    batch = build_region_batch(
        "data/anti_abeta_refs/4HIX.pdb",
        "H:96-107",
        context_chains=["L", "A"],
        antigen_chains=["A"],
        device="cpu",
    )
    antigen = batch["fragment_type"] == int(Fragment.Antigen)
    generated = batch["generate_flag"].bool()
    assert int(antigen.sum()) == 6
    assert int(generated.sum()) == 12
    assert not (antigen & generated).any()


def test_frozen_4hix_h3_mask_is_twelve_one_based_positions():
    assert parse_region_spec("H:96-107") == {"H": list(range(95, 107))}


def test_candidate_interface_checkpoint_requires_all_new_weights():
    model = types.SimpleNamespace(
        bfn=types.SimpleNamespace(
            receiver=types.SimpleNamespace(
                confidence_head_kind="candidate_interface_v1")),
        state_dict=lambda: {
            "bfn.receiver.candidate_interface_pair.weight": torch.zeros(1),
        },
    )
    with pytest.raises(RuntimeError, match="missing confidence weights"):
        _validate_candidate_interface_checkpoint(model, {})
    _validate_candidate_interface_checkpoint(
        model, {"bfn.receiver.candidate_interface_pair.weight": torch.zeros(1)})


def test_generated_candidate_confidence_post_scores_exact_sequence():
    class Core:
        @staticmethod
        def _mask_antigen(batch, candidate_mask):
            return torch.tensor([[False, False, True]])

    class Model:
        bfn = Core()

        @staticmethod
        def score(batch, fixed_t):
            assert fixed_t == 0.5
            assert torch.equal(batch["aa"], torch.tensor([[0, 1, 2]]))
            return {
                "plddt": torch.tensor([[0.1, 0.8, 0.3]]),
                "iptm": torch.tensor([0.7]),
                "pae": torch.tensor([[[0.0, 0.0, 0.4],
                                       [0.0, 0.0, 0.2],
                                       [0.0, 0.0, 0.0]]]),
            }

    batch = {
        "aa": torch.tensor([[9, 9, 2]]),
        "mask": torch.ones(1, 3, dtype=torch.bool),
        "generate_flag": torch.tensor([[True, True, False]]),
    }
    result = _score_exact_candidate_confidence(Model(), batch, "AC")
    assert result == {
        "plddt": pytest.approx(0.45),
        "plddt_std": pytest.approx(0.35),
        "iptm": pytest.approx(0.7),
        "pae": pytest.approx(0.3),
        "pae_scope": "candidate_to_antigen",
    }
    assert torch.equal(batch["aa"], torch.tensor([[9, 9, 2]]))
