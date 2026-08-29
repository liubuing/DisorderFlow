from scripts.analyze_idp_ensemble_final_fair_baseline_development import (
    recover_omitted_pose_score,
)


def test_recover_omitted_pose_score_from_full_and_retained_means():
    scores = [1.0, 2.0, 7.0]
    full = sum(scores) / 3
    retained = sum(scores[:2]) / 2
    assert recover_omitted_pose_score(full, retained, 3) == 7.0
