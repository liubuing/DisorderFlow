"""IDP 集成 v3 来源敏感性面板测试：验证实验/采样 pose 的面板切分。

守护 scripts.score_idp_ensemble_v3_source_sensitivity.panel_rows：
pose 必须按来源拆为 experimental_only / sampled_only / mixed 三面板且互不混淆——
保证来源敏感性消融（实验构象 vs 采样构象）的组别纯净。
"""
import numpy as np

from scripts.score_idp_ensemble_v3_source_sensitivity import panel_rows


# 面板切分：own_entry 进实验面板、受限采样进采样面板、mixed 含全部两条——
# 防止来源字段误判导致消融组间污染。
def test_source_panels_split_experimental_and_sampled():
    rows = [
        {"pose_id": "experimental", "source": "own_entry"},
        {
            "pose_id": "sampled",
            "source": "restrained_local_interface_sampling",
        },
    ]
    panels = panel_rows(rows)
    assert [row["pose_id"] for row in panels["experimental_only"]] == [
        "experimental"
    ]
    assert [row["pose_id"] for row in panels["sampled_only"]] == ["sampled"]
    assert len(panels["mixed"]) == 2
