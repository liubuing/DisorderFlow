"""StateContrast v2 交叉验证隔离审计测试：验证候选组跨折泄漏的检测。

守护 scripts.audit_statecontrast_v2_cv_isolation.audit：
同一 group 的记录分散在多个 fold 时必须判 cv_isolation_failed 并点名泄漏组——
防止交叉验证因组级泄漏而虚高泛化估计。
"""
import json

from scripts.audit_statecontrast_v2_cv_isolation import audit


# 泄漏检测：同组记录分属 fold 0 与 fold 1 时审计必须失败且精确报出泄漏组 g——
# 防止折划分只按记录切而忽略组边界（伪独立样本）。
def test_cv_audit_detects_candidate_group_leak(tmp_path):
    manifest = tmp_path / "manifest.json"
    base = {
        "antibody_lineage_cluster": "l", "global_sequence_cluster": "s",
        "teacher_group_id": "t", "group_id": "g", "component_id": "c",
    }
    manifest.write_text(json.dumps({
        "cross_validation_folds": 2,
        "records": [{**base, "fold_id": 0}, {**base, "fold_id": 1}],
    }))
    result = audit(manifest, tmp_path / "out.json")
    assert result["status"] == "statecontrast_v2_cv_isolation_failed"
    assert result["leaks"]["candidate_groups"] == ["g"]
