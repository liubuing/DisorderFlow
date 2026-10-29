"""IDP 集成 BFN 状态兼容性诊断测试：验证端点对候选敏感且决策受控。

守护 scripts.analyze_idp_ensemble_bfn_state_compatibility：
端点必须是 bfn_fixed_candidate_state_compatibility 且候选分数量化区间显著大于 0
（证明打分器对候选真的有区分度），决策只允许进入冻结或拒绝提升两个合法值。
"""
from pathlib import Path

from scripts.analyze_idp_ensemble_bfn_state_compatibility import analyze

ROOT = Path(__file__).resolve().parents[1]


# 候选敏感性与决策收敛：median_candidate_score_range > 0.01 证明端点非常数，
# decision 必须落在预注册的合法集合内——防止用无区分度的指标冻结下游面板。
def test_state_compatibility_diagnostic_uses_candidate_sensitive_scores(tmp_path):
    result = analyze(
        ROOT / "configs/benchmarks/idp_ensemble_bfn_state_compatibility_dev_v1.yml",
        tmp_path / "analysis.json",
    )
    assert result["endpoint"]["name"] == "bfn_fixed_candidate_state_compatibility"
    assert result["median_candidate_score_range"] > 0.01
    assert result["decision"] in {
        "eligible_to_freeze_for_new_untouched_panel",
        "do_not_promote_state_compatibility_endpoint",
    }
