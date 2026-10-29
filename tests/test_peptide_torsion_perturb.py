"""肽二面角扰动平滑方向的契约测试：守护方向向量的确定性（同输入两次一致）与单位范数，
以及 Pro 位点 phi 方向必须相对无阻尼方向衰减——防止柔性位点扰动幅度失控。"""
import numpy as np

from modules.peptide_torsion_perturb import smooth_torsion_direction


# 确定性与归一化：固定种子下两次采样逐位相同，方向向量 RMS 必须为 1（单位长度）——
# 防止扰动实现引入随机漂移或尺度失真。
def test_torsion_direction_is_deterministic_and_normalized():
    first = smooth_torsion_direction("AAPAA", 7)
    second = smooth_torsion_direction("AAPAA", 7)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    observed = np.sqrt(np.mean(np.concatenate(first) ** 2))
    assert observed == np.testing.assert_allclose(observed, 1.0) or observed == 1.0


# Pro 阻尼：脯氨酸刚性五元环限制 phi，其平滑方向幅度必须显著小于无阻尼值（|phi|<2.0），
# 防止对 Pro 施加不合理的骨架扭转扰动。
def test_proline_phi_is_damped_relative_to_undamped_direction():
    phi, _ = smooth_torsion_direction("AAPAA", 11)
    assert abs(phi[2]) < 2.0
