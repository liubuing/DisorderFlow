"""IDP 集成 v3 测量完备性检查测试：验证缺测量时 fail-closed 阻断。

守护 scripts.check_idp_ensemble_v3_measurements.check：
测量 CSV 缺失或无匹配组件时状态必须为 measurements_blocked 且计数为 0——
防止在零测量数据上继续推进确认流程。
"""
from scripts.check_idp_ensemble_v3_measurements import check


# fail-closed：测量文件不存在时必须阻断且 measured_components=0——防止半空数据冒充完成。
def test_missing_measurements_are_blocked(tmp_path):
    panel = tmp_path / "panel.json"
    panel.write_text(
        '{"components": [{"component_id": "C1"}]}', encoding="ascii"
    )
    result = check(tmp_path / "missing.csv", panel, tmp_path / "status.json")
    assert result["status"] == "measurements_blocked"
    assert result["measured_components"] == 0
