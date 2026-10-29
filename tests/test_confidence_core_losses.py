"""置信度核心损失（grouped ranking / pair difference）的契约测试：守护候选-抗原配对掩码的方向性、
低于重复测量噪声的配对不计入损失、按不确定性加权，以及归一化 pair difference 的尺度不变性
与"无证据时梯度恒为零"的静默行为。"""
import torch

from disorderflow.modules.bfn.core_losses import (
    candidate_antigen_pair_mask,
    grouped_margin_ranking,
    grouped_pair_difference,
)


# 掩码默认单向：只允许 候选→抗原 方向的配对，防止 v2+ 损失口径意外包含反向对。
def test_candidate_antigen_pair_mask_is_forward_only_by_default():
    candidate = torch.tensor([[True, False, False]])
    antigen = torch.tensor([[False, False, True]])
    mask = candidate_antigen_pair_mask(candidate, antigen)
    assert mask[0, 0, 2]
    assert not mask[0, 2, 0]
    assert mask.sum().item() == 1


# 向后兼容：bidirectional=True 时掩码包含正反两个方向，用于复现 v1 双向损失口径，
# 防止兼容开关静默失效导致新旧结果不可比。
def test_candidate_antigen_pair_mask_can_reproduce_v1_bidirectional_loss():
    candidate = torch.tensor([[True, False, False]])
    antigen = torch.tensor([[False, False, True]])
    mask = candidate_antigen_pair_mask(candidate, antigen, bidirectional=True)
    assert mask[0, 0, 2]
    assert mask[0, 2, 0]
    assert mask.sum().item() == 2


# 噪声容忍：目标差异（0.01）低于重复噪声（0.02）的配对不产生损失，防止模型拟合测量噪声。
def test_grouped_margin_ignores_differences_below_replicate_noise():
    prediction = torch.tensor([0.9, 0.1], requires_grad=True)
    target = torch.tensor([0.51, 0.50])
    group = torch.tensor([0, 0])
    noise = torch.tensor([0.02, 0.02])
    loss = grouped_margin_ranking(
        prediction, target, group, target_noise=noise)
    assert loss.item() == 0.0


# 可靠性加权：超出噪声的排序违反要产生损失且按样本权重缩放，梯度可回传——
# 保证强证据配对主导优化方向。
def test_grouped_margin_weights_reliable_pairs():
    prediction = torch.tensor([0.1, 0.9], requires_grad=True)
    target = torch.tensor([0.8, 0.2])
    group = torch.tensor([0, 0])
    noise = torch.tensor([0.02, 0.02])
    weights = torch.tensor([1.0, 0.5])
    loss = grouped_margin_ranking(
        prediction, target, group, margin=0.05,
        target_noise=noise, sample_weight=weights)
    assert loss.item() > 0.0
    loss.backward()
    assert prediction.grad is not None


# 配对不确定性按平方和（quadrature）合成：两组 SEM 各 0.025 时联合噪声大于目标差 0.03，
# 该配对应被忽略——防止不确定性合成口径被误改为线性求和。
def test_grouped_margin_uses_quadrature_pair_uncertainty():
    prediction = torch.tensor([0.9, 0.1], requires_grad=True)
    target = torch.tensor([0.53, 0.50])
    group = torch.tensor([0, 0])
    sem = torch.tensor([0.025, 0.025])
    loss = grouped_margin_ranking(
        prediction, target, group, target_noise=sem)
    assert loss.item() == 0.0


# 坍缩破解：预测完全坍缩为零时 pair difference 必须给出非零损失与非零梯度，
# 防止排序损失在坍缩解上梯度消失。
def test_grouped_pair_difference_breaks_exact_prediction_collapse():
    prediction = torch.zeros(3, requires_grad=True)
    target = torch.tensor([0.2, 0.5, 0.8])
    group = torch.tensor([0, 0, 0])
    sem = torch.tensor([0.01, 0.01, 0.01])
    loss = grouped_pair_difference(
        prediction, target, group, target_noise=sem)
    loss.backward()
    assert loss.item() > 0
    assert prediction.grad.abs().sum() > 0


# 联合 SEM 内配对忽略：差异 0.01 小于联合噪声时损失为零，防止对噪声内差异过拟合。
def test_grouped_pair_difference_ignores_pairs_within_joint_sem():
    prediction = torch.tensor([0.0, 1.0], requires_grad=True)
    target = torch.tensor([0.50, 0.51])
    group = torch.tensor([0, 0])
    sem = torch.tensor([0.02, 0.02])
    loss = grouped_pair_difference(
        prediction, target, group, target_noise=sem)
    assert loss.item() == 0.0


# 归一化模式同样必须破解坍缩：目标量级很小时（RMS 归一化放大差异）损失仍非零、有限且可回传。
def test_normalized_grouped_pair_difference_breaks_prediction_collapse():
    prediction = torch.zeros(3, requires_grad=True)
    target = torch.tensor([0.20, 0.21, 0.23])
    group = torch.tensor([0, 0, 0])
    sem = torch.tensor([0.001, 0.001, 0.001])
    loss = grouped_pair_difference(
        prediction, target, group, target_noise=sem,
        normalize_by_target_rms=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert prediction.grad.abs().sum() > 0


# 尺度不变性：预测/目标/噪声整体放大 10 倍后损失逐位一致——RMS 归一化保证损失与目标量纲无关。
def test_normalized_grouped_pair_difference_is_scale_invariant():
    prediction = torch.tensor([0.10, 0.15, 0.22])
    target = torch.tensor([0.20, 0.24, 0.31])
    group = torch.tensor([0, 0, 0])
    sem = torch.tensor([0.002, 0.003, 0.002])
    loss = grouped_pair_difference(
        prediction, target, group, target_noise=sem,
        normalize_by_target_rms=True)
    scaled_loss = grouped_pair_difference(
        prediction * 10, target * 10, group, target_noise=sem * 10,
        normalize_by_target_rms=True)
    assert torch.allclose(loss, scaled_loss)


# 零点校准：预测差异与目标差异完全一致时归一化损失必须为零，防止归一化引入系统偏移。
def test_normalized_grouped_pair_difference_is_zero_at_exact_differences():
    target = torch.tensor([0.20, 0.24, 0.31])
    prediction = target.clone().requires_grad_(True)
    group = torch.tensor([0, 0, 0])
    loss = grouped_pair_difference(
        prediction, target, group, normalize_by_target_rms=True)
    assert loss.item() == 0.0


# 数值稳定：目标 RMS 趋近零（1e-7 量级）时归一化不得产生 NaN/Inf，防止小数值靶点炸掉训练。
def test_normalized_grouped_pair_difference_handles_tiny_eligible_rms():
    prediction = torch.zeros(2, requires_grad=True)
    target = torch.tensor([0.0, 1e-7])
    group = torch.tensor([0, 0])
    sem = torch.zeros(2)
    loss = grouped_pair_difference(
        prediction, target, group, target_noise=sem,
        normalize_by_target_rms=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()


# 静默不隐形训练：无合格配对（全在联合 SEM 内）时损失为零且梯度严格为零——
# 防止被忽略的配对通过未 mask 的路径偷偷贡献梯度。
def test_normalized_grouped_pair_difference_stays_connected_without_evidence():
    prediction = torch.tensor([0.0, 1.0], requires_grad=True)
    target = torch.tensor([0.50, 0.51])
    group = torch.tensor([0, 0])
    sem = torch.tensor([0.02, 0.02])
    loss = grouped_pair_difference(
        prediction, target, group, target_noise=sem,
        normalize_by_target_rms=True)
    loss.backward()
    assert loss.item() == 0.0
    assert prediction.grad is not None
    assert prediction.grad.abs().sum().item() == 0.0
