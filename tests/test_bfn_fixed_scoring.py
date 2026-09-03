import types

import pytest
import torch
import torch.nn as nn


def test_region_spec_converts_one_based_positions_exactly():
    from modules.bfn_loader import parse_region_spec

    assert parse_region_spec('B:1-3, 7,3;A:2') == {
        'B': [0, 1, 2, 6],
        'A': [1],
    }
    with pytest.raises(ValueError, match='1-based'):
        parse_region_spec('B:0-3')


def test_candidate_injection_is_exact_and_validates_length():
    from modules.bfn_loader import inject_candidate_sequence

    batch = {
        'aa': torch.tensor([[9, 9, 9, 9]]),
        'generate_flag': torch.tensor([[False, True, False, True]]),
    }
    inject_candidate_sequence(batch, 'AC')
    assert torch.equal(batch['aa'], torch.tensor([[9, 0, 9, 1]]))

    with pytest.raises(ValueError, match='requires 2'):
        inject_candidate_sequence(batch, 'A')


class _FixedReceiver(nn.Module):
    def forward(self, seq, pos, ori, ang, t, pair_feat, mask_res, **kwargs):
        assert torch.equal(seq.argmax(dim=-1), torch.tensor([[0, 1, 2]]))
        assert kwargs['orientation_is_rotation'] is True
        n, length = mask_res.shape
        zeros = torch.zeros(n, length)
        return (
            zeros.unsqueeze(-1), zeros.unsqueeze(-1), zeros.unsqueeze(-1),
            zeros.unsqueeze(-1),
            torch.tensor([[1.0, 2.0, 3.0]]),
            torch.tensor([4.0]),
            torch.arange(9, dtype=torch.float32).reshape(1, 3, 3),
            torch.tensor([[5.0, 6.0, 7.0]]),
            torch.tensor([[8.0, 9.0, 10.0]]),
            torch.tensor([11.0]),
        )


def test_score_fixed_propagates_outputs_and_masks_padding():
    from disorderflow.modules.bfn.core import AntibodyBFN_Core

    core = object.__new__(AntibodyBFN_Core)
    nn.Module.__init__(core)
    core.num_classes = 22
    core.receiver = _FixedReceiver()
    core.register_buffer('position_mean', torch.zeros(1, 1, 3))
    core.register_buffer('position_scale', torch.ones(1, 1, 1))
    batch = {
        'aa': torch.tensor([[0, 1, 2]]),
        'mask': torch.tensor([[True, True, False]]),
        'generate_flag': torch.tensor([[False, True, False]]),
        'fragment_type': torch.zeros(1, 3, dtype=torch.long),
        'pos_heavyatom': torch.tensor([[[
            [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[2.0, 1.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 1.0, 0.0]],
            [[4.0, 1.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 1.0, 0.0]],
        ]]),
        'torsion': torch.zeros(1, 3, 4),
        'pair_feat': torch.zeros(1, 3, 3, 2),
    }

    result = core.score_fixed(batch)

    assert torch.equal(result['plddt'], torch.tensor([[1.0, 2.0, 0.0]]))
    assert torch.equal(result['pae'], torch.tensor([[[0.0, 1.0, 0.0],
                                                    [3.0, 4.0, 0.0],
                                                    [0.0, 0.0, 0.0]]]))
    assert torch.equal(result['disorder'], torch.tensor([[5.0, 6.0, 0.0]]))
    assert torch.equal(result['contact'], torch.tensor([[8.0, 9.0, 0.0]]))
    assert result['iptm'].item() == 4.0
    assert result['state_compatibility'].item() == 11.0


def test_model_score_uses_unmasked_embeddings_and_propagates_result():
    from disorderflow.models.bfn_model import AntibodyBFN

    class Core(nn.Module):
        def score_fixed(self, batch, fixed_t):
            assert batch['pair_feat'].item() == 2.0
            return {'state_compatibility': torch.tensor([fixed_t])}

    model = object.__new__(AntibodyBFN)
    nn.Module.__init__(model)
    model.bfn = Core()

    def encode(self, batch, remove_structure, remove_sequence):
        assert remove_structure is False
        assert remove_sequence is False
        return torch.tensor(1.0), torch.tensor(2.0)

    model.encode = types.MethodType(encode, model)
    result = model.score({}, fixed_t=0.25)
    assert result['state_compatibility'].item() == 0.25


def test_pair_contact_noisy_or_aggregation_and_empty_antigen():
    from disorderflow.modules.bfn.receiver import aggregate_pair_contact_logits

    pair_logits = torch.tensor([[[0.0, 0.0], [4.0, -4.0]]])
    antigen = torch.tensor([[True, True]])
    result = aggregate_pair_contact_logits(pair_logits, antigen)
    assert torch.allclose(torch.sigmoid(result[0, 0]), torch.tensor(0.75))
    assert torch.sigmoid(result[0, 1]) > 0.98
    empty = aggregate_pair_contact_logits(pair_logits, torch.zeros_like(antigen))
    assert torch.equal(empty, torch.full_like(empty, -20.0))

    saturated = aggregate_pair_contact_logits(
        torch.full((1, 2, 200), 10.0, dtype=torch.bfloat16),
        torch.ones((1, 200), dtype=torch.bool))
    assert torch.isfinite(saturated).all()


def test_patch_protein_guarantees_required_context_fragment():
    from disorderflow.utils.transforms.patch_protein import PatchProtein

    length = 20
    data = {
        'aa': torch.zeros(length, dtype=torch.long),
        'fragment_type': torch.tensor([0] * 5 + [1] * 10 + [2] * 5),
        'anchor_flag': torch.tensor([True] + [False] * (length - 1)),
        'generate_flag': torch.tensor([True] + [False] * (length - 1)),
        'pos_heavyatom': torch.zeros(length, 15, 3),
        'mask_heavyatom': torch.ones(length, 15, dtype=torch.bool),
    }
    data['pos_heavyatom'][:, :, 1] = torch.arange(length).view(-1, 1)
    transform = PatchProtein(
        initial_patch_size=2, context_size=2,
        required_fragment_types=[2], required_context_cap=3)
    patched = transform(data)
    assert (patched['fragment_type'] == 2).sum().item() == 3
