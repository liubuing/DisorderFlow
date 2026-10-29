"""AAYL 低阶基准（v2）的契约测试：守护 32 组按家族划分的拆分预算、训练集仅含 ≤2 突变候选、
测试集拟合不受测试特征污染，以及排除候选审计的账目守恒。"""
import json
import numpy as np

from scripts import benchmark_aayl_low_order as v2


# 拆分契约：train/test 无泄漏、按突变阶数分组、家族内取样、固定测试规模与训练预算，
# 且 covered 字段必须精确等于"训练中已覆盖全部突变位点的测试候选"——保证低阶外推评估口径可复现。
def test_split_budget_coverage_and_no_test_candidate_leakage():
    _, rows, changes = v2.base.load()
    splits = json.loads((v2.OUT / "splits.json").read_text())["splits"]
    assert len(splits) == 32
    for s in splits:
        assert not set(s["train"]) & set(s["test"])
        assert all(len(changes[i]) <= 2 for i in s["train"])
        assert all(len(changes[i]) == 3 for i in s["test"])
        assert all(rows[i]["family"] == s["family"] for i in s["train"] + s["test"])
        assert len(s["test"]) == (102 if s["family"] == "AAYL49" else 95)
        assert len(s["train"]) == (s["budget"] if isinstance(s["budget"],int) else (179 if s["family"] == "AAYL49" else 132))
        seen = frozenset().union(*(changes[i] for i in s["train"]))
        assert s["covered"] == [i for i in s["test"] if changes[i] <= seen]


# 防泄漏回归：把测试特征平移 10000 后，选出的超参、验证 MSE 与 scaler 均值必须逐位不变——
# 确保拟合与标准化只使用训练数据，测试集不参与任何统计量估计。
def test_test_features_do_not_change_fit_or_scaler():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(30, 4))
    train, test = np.arange(20), np.arange(20, 30)
    y = x[train, 0] + .3*x[train, 1]
    _, params, mse, fit = v2.fit_predict(x, y, train, test, "esm2_ridge", {"alpha":[.1, 1.]}, 7)
    changed = x.copy()
    changed[test] += 10000
    _, params2, mse2, fit2 = v2.fit_predict(changed, y, train, test, "esm2_ridge", {"alpha":[.1, 1.]}, 7)
    assert params == params2 and mse == mse2
    scaler = fit.best_estimator_.regressor_.named_steps["standardscaler"]
    np.testing.assert_allclose(scaler.mean_, x[train].mean(0))
    np.testing.assert_allclose(fit.predict(x[train]), fit2.predict(x[train]))


# 审计账目守恒：included 与 excluded 之和必须等于全部候选数，排除原因与有限读数分类各自求和也与 excluded 一致，
# 防止候选在审计与基准之间被静默丢弃或重复计数。
def test_missing_audit_accounts_for_all_candidates():
    d, rows, changes = v2.base.load()
    audit = v2.label_audit(d,rows,changes)
    assert sum(a["included"] for a in audit) == len(rows)
    assert sum(a["excluded"] for a in audit) == len(d["excluded"])
    for a in audit:
        assert sum(a["excluded_reasons"].values()) == a["excluded"]
        assert sum(a["finite_readings_in_excluded"].values()) == a["excluded"]
