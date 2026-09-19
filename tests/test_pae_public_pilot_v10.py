from scripts.run_pae_public_pilot_v10 import selected_sequences, screening_metrics


def test_hash_panel_selection_ignores_input_order():
    pool=[str(i) for i in range(30)]
    assert selected_sequences('PUB001',pool)==selected_sequences('PUB001',pool[::-1])
    assert len(selected_sequences('PUB001',pool))==20


def test_screening_rounding_and_random_expectation():
    ids=[str(i) for i in range(11)]
    result=screening_metrics(ids,list(range(11)),list(range(11)))
    assert result['budget']==3
    assert result['recall']==1
    assert result['random_expected_recall']==3/11
    result=screening_metrics(ids,list(range(11)),list(reversed(range(11))))
    assert result['recall']==0


def test_constant_predictions_are_not_reported_as_correlation():
    assert screening_metrics(['a','b','c','d','e'],[1]*5,list(range(5)))['spearman'] is None
