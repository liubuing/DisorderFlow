"""Tests for BFN confidence head forward shapes (CPU, no checkpoint required).

Validates the V12 pure-sequence confidence heads (plddt/iptm/pae) produce
correct shapes on a tiny synthetic input. This catches architecture regressions
without needing the full model or GPU.
"""
import pytest

torch = pytest.importorskip("torch")

from disorderflow.modules.bfn.receiver import (
    AntibodyBFN_Receiver,
    candidate_antigen_geometry,
)

N, L, D, P = 2, 16, 64, 32  # batch, length, res_feat_dim, pair_feat_dim
NUM_CLASSES = 20


def _make_receiver(
        confidence_version="v12", seq_only=True,
        confidence_head_kind="legacy_v12"):
    return AntibodyBFN_Receiver(
        res_feat_dim=D, pair_feat_dim=P, num_layers=2,
        num_classes=NUM_CLASSES, seq_only=seq_only,
        head_dropout=0.0, disorder_head=False,
        confidence_version=confidence_version,
        confidence_head_kind=confidence_head_kind,
    )


def _inputs():
    torch.manual_seed(0)
    theta_seq = torch.randn(N, L, NUM_CLASSES)
    theta_pos = torch.randn(N, L, 3)
    theta_ori = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(N, L, 3, 3).clone()
    theta_ang = torch.zeros(N, L, 4)
    t = torch.rand(N)
    pair_feat = torch.randn(N, L, L, P)
    mask_res = torch.ones(N, L, dtype=torch.bool)
    return theta_seq, theta_pos, theta_ori, theta_ang, t, pair_feat, mask_res


def test_v12_confidence_head_shapes_seqonly():
    """V12 heads output per-residue plddt, scalar iptm, per-pair pae."""
    recv = _make_receiver("v12", seq_only=True)
    theta_seq, theta_pos, theta_ori, theta_ang, t, pair_feat, mask_res = _inputs()
    out = recv(theta_seq, theta_pos, theta_ori, theta_ang, t, pair_feat, mask_res)
    pred_seq, pred_pos, pred_ori_6d, pred_ang_sc, pred_plddt, pred_iptm, pred_pae, pred_disorder, _, _ = out
    assert pred_plddt.shape == (N, L)
    assert pred_iptm.shape == (N,)
    assert pred_pae.shape == (N, L, L)
    assert pred_disorder is None  # disorder_head=False


def test_v12_confidence_outputs_in_unit_interval():
    """pLDDT / ipTM / PAE are sigmoids → must lie in [0, 1]."""
    recv = _make_receiver("v12", seq_only=True)
    theta_seq, theta_pos, theta_ori, theta_ang, t, pair_feat, mask_res = _inputs()
    _, _, _, _, pred_plddt, pred_iptm, pred_pae, _, _, _ = recv(
        theta_seq, theta_pos, theta_ori, theta_ang, t, pair_feat, mask_res)
    assert torch.all(pred_plddt >= 0) and torch.all(pred_plddt <= 1)
    assert torch.all(pred_iptm >= 0) and torch.all(pred_iptm <= 1)
    assert torch.all(pred_pae >= 0) and torch.all(pred_pae <= 1)


def test_default_confidence_head_is_legacy_equivalent():
    default = _make_receiver()
    explicit = _make_receiver(confidence_head_kind="legacy_v12")
    explicit.load_state_dict(default.state_dict())
    default.eval()
    explicit.eval()
    inputs = _inputs()
    default_out = default(*inputs)
    explicit_out = explicit(*inputs)
    for index in (4, 5, 6):
        assert torch.equal(default_out[index], explicit_out[index])


def test_v11_backbone_plus_bypass_shapes():
    """V11 head (backbone + seq bypass) produces the same shapes."""
    recv = _make_receiver("v11", seq_only=True)
    theta_seq, theta_pos, theta_ori, theta_ang, t, pair_feat, mask_res = _inputs()
    _, _, _, _, pred_plddt, pred_iptm, pred_pae, _, _, _ = recv(
        theta_seq, theta_pos, theta_ori, theta_ang, t, pair_feat, mask_res)
    assert pred_plddt.shape == (N, L)
    assert pred_iptm.shape == (N,)
    assert pred_pae.shape == (N, L, L)


def test_recycle_gates_declared_and_low_init():
    """V14 Complex-mode fix: scalar confidence feedback channels are gated,
    initialised low (0.3) so design-specific seq_emb feedback dominates."""
    recv = _make_receiver("v12", seq_only=True)
    for name in ("recycle_gate_conf", "recycle_gate_iptm", "recycle_gate_pae"):
        p = dict(recv.named_parameters())[name]
        assert p.shape == ()                      # scalar gate
        assert abs(p.item() - 0.3) < 1e-3         # low init


def test_recycle_feedback_forward_runs():
    """Passing prev_conf/iptm/pae/seq_emb feedback through the gates works
    (recycle path exercised) — critical for Complex-mode recycling."""
    recv = _make_receiver("v12", seq_only=True)
    theta_seq, theta_pos, theta_ori, theta_ang, t, pair_feat, mask_res = _inputs()
    prev_conf = torch.rand(N, L)
    prev_iptm = torch.rand(N)
    prev_pae = torch.rand(N, L, L)
    prev_seq_emb = torch.rand(N, L, 64)
    out = recv(theta_seq, theta_pos, theta_ori, theta_ang, t, pair_feat, mask_res,
               prev_conf=prev_conf, prev_iptm=prev_iptm, prev_pae=prev_pae,
               prev_seq_emb=prev_seq_emb)
    pred_plddt = out[4]
    assert pred_plddt.shape == (N, L)


def _interface_masks():
    candidate = torch.zeros(N, L, dtype=torch.bool)
    antigen = torch.zeros(N, L, dtype=torch.bool)
    candidate[:, 2:6] = True
    antigen[:, 10:14] = True
    return candidate, antigen


def test_candidate_interface_head_is_sequence_sensitive():
    recv = _make_receiver(confidence_head_kind="candidate_interface_v1")
    recv.eval()
    inputs = list(_inputs())
    candidate, antigen = _interface_masks()
    baseline = recv(*inputs, mask_gen=candidate, mask_antigen=antigen)
    inputs[0] = inputs[0].clone()
    inputs[0][:, 2:6] = -20.0
    inputs[0][:, 2:6, 7] = 20.0
    variant = recv(*inputs, mask_gen=candidate, mask_antigen=antigen)
    assert not torch.allclose(baseline[4][:, 2:6], variant[4][:, 2:6])
    assert not torch.allclose(baseline[5], variant[5])
    assert not torch.allclose(baseline[6], variant[6])


def test_candidate_interface_v2_does_not_soften_probabilities_twice():
    recv = _make_receiver(confidence_head_kind="candidate_interface_v2")
    recv.eval()
    inputs = list(_inputs())
    inputs[0] = torch.nn.functional.one_hot(
        torch.arange(L).remainder(NUM_CLASSES), NUM_CLASSES).float()
    inputs[0] = inputs[0].unsqueeze(0).expand(N, -1, -1).clone()
    candidate, antigen = _interface_masks()
    observed = []
    handle = recv.seq_embed.register_forward_pre_hook(
        lambda _module, args: observed.append(args[0].detach().clone()))
    try:
        recv(*inputs, mask_gen=candidate, mask_antigen=antigen)
    finally:
        handle.remove()
    assert torch.equal(observed[0], inputs[0])


def test_candidate_interface_v1_retains_historical_second_softmax():
    recv = _make_receiver(confidence_head_kind="candidate_interface_v1")
    recv.eval()
    inputs = list(_inputs())
    inputs[0] = torch.nn.functional.one_hot(
        torch.arange(L).remainder(NUM_CLASSES), NUM_CLASSES).float()
    inputs[0] = inputs[0].unsqueeze(0).expand(N, -1, -1).clone()
    candidate, antigen = _interface_masks()
    observed = []
    handle = recv.seq_embed.register_forward_pre_hook(
        lambda _module, args: observed.append(args[0].detach().clone()))
    try:
        recv(*inputs, mask_gen=candidate, mask_antigen=antigen)
    finally:
        handle.remove()
    assert torch.allclose(observed[0], torch.softmax(inputs[0], dim=-1))


def test_candidate_interface_v2_exposes_trainable_entity_summaries():
    recv = _make_receiver(confidence_head_kind="candidate_interface_v2")
    candidate, antigen = _interface_masks()
    inputs = list(_inputs())
    inputs[0] = torch.softmax(inputs[0], dim=-1).requires_grad_()
    recv(*inputs, mask_gen=candidate, mask_antigen=antigen)
    summaries = recv.last_candidate_interface_summaries
    assert set(summaries) == {"plddt", "iptm", "pae"}
    assert all(value.shape == (N,) for value in summaries.values())
    sum(value.mean() for value in summaries.values()).backward()
    assert inputs[0].grad[candidate].abs().sum() > 0
    assert recv.candidate_interface_plddt_summary[-1].weight.grad is not None
    assert recv.candidate_interface_pae_summary[-1].weight.grad is not None


def test_candidate_interface_geometry_is_frame_invariant():
    inputs = _inputs()
    candidate, antigen = _interface_masks()
    baseline = candidate_antigen_geometry(inputs[1], candidate, antigen)
    shifted = candidate_antigen_geometry(
        inputs[1] + torch.tensor([10.0, -4.0, 7.0]), candidate, antigen)
    assert torch.allclose(baseline, shifted, atol=1e-6)


def test_candidate_interface_v3_summaries_use_explicit_geometry():
    recv = _make_receiver(confidence_head_kind="candidate_interface_v3")
    recv.eval()
    candidate, antigen = _interface_masks()
    inputs = list(_inputs())
    inputs[0] = torch.softmax(inputs[0], dim=-1)
    recv(*inputs, mask_gen=candidate, mask_antigen=antigen)
    baseline = {
        key: value.detach().clone()
        for key, value in recv.last_candidate_interface_summaries.items()}
    inputs[1] = inputs[1].clone()
    inputs[1][antigen] += torch.tensor([20.0, 0.0, 0.0])
    recv(*inputs, mask_gen=candidate, mask_antigen=antigen)
    changed = recv.last_candidate_interface_summaries
    assert set(changed) == {"plddt", "iptm", "pae"}
    assert all(not torch.allclose(baseline[key], changed[key]) for key in baseline)


def test_candidate_interface_head_is_interface_sensitive():
    recv = _make_receiver(confidence_head_kind="candidate_interface_v1")
    recv.eval()
    inputs = list(_inputs())
    candidate, antigen = _interface_masks()
    baseline = recv(*inputs, mask_gen=candidate, mask_antigen=antigen)
    inputs[5] = inputs[5].clone()
    inputs[5][:, 2:6, 10:14] += 5.0
    variant = recv(*inputs, mask_gen=candidate, mask_antigen=antigen)
    assert not torch.allclose(baseline[4][:, 2:6], variant[4][:, 2:6])
    assert not torch.allclose(baseline[5], variant[5])
    assert not torch.allclose(baseline[6], variant[6])


def test_candidate_interface_head_requires_both_masks():
    recv = _make_receiver(confidence_head_kind="candidate_interface_v1")
    inputs = _inputs()
    candidate, antigen = _interface_masks()
    with pytest.raises(ValueError, match="candidate and antigen masks"):
        recv(*inputs)
    with pytest.raises(ValueError, match="antigen residues"):
        recv(*inputs, mask_gen=candidate, mask_antigen=torch.zeros_like(antigen))


def test_candidate_interface_outputs_backpropagate_to_sequence_and_pairs():
    recv = _make_receiver(confidence_head_kind="candidate_interface_v1")
    candidate, antigen = _interface_masks()
    logits = torch.randn(N, L, NUM_CLASSES, requires_grad=True)
    pairs = torch.randn(N, L, L, P, requires_grad=True)
    plddt, iptm, pae, _ = recv._candidate_interface_confidence(
        torch.softmax(logits, dim=-1), pairs, _inputs()[1],
        torch.ones(N, L, dtype=torch.bool), candidate, antigen,
    )
    interface_pae = pae[candidate.unsqueeze(-1) & antigen.unsqueeze(-2)]
    (plddt[candidate].mean() + iptm.mean() + interface_pae.mean()).backward()
    assert logits.grad[candidate].abs().sum() > 0
    interface_mask = candidate.unsqueeze(-1) & antigen.unsqueeze(-2)
    assert pairs.grad[interface_mask].abs().sum() > 0
