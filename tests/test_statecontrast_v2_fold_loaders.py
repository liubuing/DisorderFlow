"""状态对比 v2 折加载器审计的契约测试：守护当前交叉验证契约的加载器检查全部通过
且留出组件数为冻结值（23）——防止折数据加载路径偏离契约。"""
from scripts.audit_statecontrast_v2_fold_loaders import audit


# 冻结核对：全部加载器检查为真且留出组件数恰为 23——防止折组件集被静默增删。
def test_current_fold_loaders_are_component_disjoint(tmp_path):
    result = audit(
        __import__("pathlib").Path(
            "reviewer_outputs/statecontrast_v2_cross_validation_v2/contract.json"),
        tmp_path / "audit.json",
    )
    assert all(result["checks"].values())
    assert result["heldout_component_count"] == 23
