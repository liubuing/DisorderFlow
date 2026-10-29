"""IDP 集成扩展重采样计划 v2 测试：验证仅双就绪组件进入重采样请求。

守护 scripts.prepare_idp_ensemble_expanded_resampling_v2.prepare：
只有同时满足 multi_pose_ready 与 experimental_pose_ready 的组件才发起采样请求，
且组件输出必须翻转就绪标志（请求后视为已消费）——防止重复采样或绕过就绪闸门。
"""
import json

from scripts.prepare_idp_ensemble_expanded_resampling_v2 import prepare


# 双闸门语义：仅 C1（两处均就绪）被请求采样；输出中 C1 的 multi_pose_ready
# 翻转为 False（已消费）、C2 翻转为 True（等待中）——防止就绪状态被重复消费。
def test_resampling_requests_experimental_ready_components(tmp_path):
    readiness = tmp_path / "readiness.json"
    donor = tmp_path / "donor.json"
    readiness.write_text(json.dumps({
        "components": [
            {"component_id": "C1", "multi_pose_ready": True},
            {"component_id": "C2", "multi_pose_ready": False},
        ]
    }), encoding="ascii")
    donor.write_text(json.dumps({
        "components": [
            {"component_id": "C1", "experimental_pose_ready": True},
            {"component_id": "C2", "experimental_pose_ready": False},
        ]
    }), encoding="ascii")
    result = prepare(readiness, donor, tmp_path / "output.json")
    assert result["sampling_requested_count"] == 1
    assert result["components"][0]["multi_pose_ready"] is False
    assert result["components"][1]["multi_pose_ready"] is True
