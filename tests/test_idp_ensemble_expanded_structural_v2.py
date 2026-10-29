"""IdP 系综扩展结构 v2 审计的契约测试：守护 H3-抗原接触检测按距离阈值（4.5 Å）正确计数，
防止接触判定口径漂移导致界面证据虚增或丢失。"""
import numpy as np

from scripts.audit_idp_ensemble_expanded_structural_v2 import contact_metrics


# 接触几何：恰好 4.0 Å（<4.5 阈值）的 H3 残基 0 必须被判为接触且配对数为 1——
# 防止阈值比较方向写反（<=/>）后接触集合全错。
def test_contact_metrics_detect_h3_antigen_contact():
    heavy = [{"atoms": np.array([[0.0, 0.0, 0.0]])}]
    antigen = [{"atoms": np.array([[0.0, 0.0, 4.0]])}]
    result = contact_metrics(heavy, antigen, [0], 4.5)
    assert result["contacting_h3_positions"] == [0]
    assert result["contact_pair_count"] == 1
