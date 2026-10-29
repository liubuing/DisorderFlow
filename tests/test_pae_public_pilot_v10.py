"""PAE 公开试点 v10 的契约测试：守护哈希面板选择对输入顺序不敏感、筛选指标的预算/召回/
随机期望口径，以及恒定预测不得报告 Spearman 相关性——防止试点汇总的统计口径漂移。"""
from scripts.run_pae_public_pilot_v10 import selected_sequences, screening_metrics


# 选择确定性：同一池子（任意顺序）按哈希选出完全相同的 20 条——防止顺序抖动改变入选集合。
def test_hash_panel_selection_ignores_input_order():
    pool=[str(i) for i in range(30)]
    assert selected_sequences('PUB001',pool)==selected_sequences('PUB001',pool[::-1])
    assert len(selected_sequences('PUB001',pool))==20


# 指标口径：预算 3 下完美排序召回为 1、随机期望为 3/11、完全逆序召回为 0——
# 防止召回比较缺失随机基线或方向颠倒。
def test_screening_rounding_and_random_expectation():
    ids=[str(i) for i in range(11)]
    result=screening_metrics(ids,list(range(11)),list(range(11)))
    assert result['budget']==3
    assert result['recall']==1
    assert result['random_expected_recall']==3/11
    result=screening_metrics(ids,list(range(11)),list(reversed(range(11))))
    assert result['recall']==0


# 零方差守护：恒定预测（无离散度）时 Spearman 必须为 None 而非 0 或 NaN——防止伪相关进入报告。
def test_constant_predictions_are_not_reported_as_correlation():
    assert screening_metrics(['a','b','c','d','e'],[1]*5,list(range(5)))['spearman'] is None
