"""Tests for BFN confidence head forward shapes (CPU, no checkpoint required).

Validates the V12 pure-sequence confidence heads (plddt/iptm/pae) produce
correct shapes on a tiny synthetic input. This catches architecture regressions
without needing the full model or GPU.
"""
import pytest

torch = pytest.importorskip("torch")

from disorderflow.modules.bfn.receiver import AntibodyBFN_Receiver

N, L, D, P = 2, 16, 64, 32  # batch, length, res_feat_dim, pair_feat_dim
NUM_CLASSES = 20


def _make_receiver(confidence_version="v12", seq_only=True):
    return AntibodyBFN_Receiver(
        res_feat_dim=D, pair_feat_dim=P, num_layers=2,
        num_classes=NUM_CLASSES, seq_only=seq_only,
        head_dropout=0.0, disorder_head=False,
        confidence_version=confidence_version,
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
