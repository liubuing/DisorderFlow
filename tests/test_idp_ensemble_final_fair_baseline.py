"""IdP 系综终点公平基线的契约测试：守护从全量均值与保留子集均值反推被省略姿态分数的代数正确性，
防止基线汇总因缺姿态而被系统性抬高或压低。"""
from scripts.analyze_idp_ensemble_final_fair_baseline_development import (
    recover_omitted_pose_score,
)


# 反推一致性：全集均值与保留均值之差必须精确还原遗漏姿态分数 7.0——
# 保证"补齐遗漏"的基线修正只依赖已发布统计量且无自由度。
def test_recover_omitted_pose_score_from_full_and_retained_means():
    scores = [1.0, 2.0, 7.0]
    full = sum(scores) / 3
    retained = sum(scores[:2]) / 2
    assert recover_omitted_pose_score(full, retained, 3) == 7.0
