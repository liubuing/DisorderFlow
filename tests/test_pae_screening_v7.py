from scripts.pae_screening_next_v7 import rank_ids


def test_screen_excludes_controls_and_handles_ties_and_components():
    rows = [
        {'entity_id': 'b', 'component_id': 'A', 'entity_type': 'candidate'},
        {'entity_id': 'a', 'component_id': 'A', 'entity_type': 'candidate'},
        {'entity_id': 'native', 'component_id': 'A', 'entity_type': 'native'},
        {'entity_id': 'c', 'component_id': 'B', 'entity_type': 'candidate'},
    ]
    assert rank_ids(rows, [1, 1, -100, 5]) == ['a', 'c']


def test_selection_budget_is_ceil_and_order_independent():
    rows = [{'entity_id': str(i), 'component_id': 'A', 'entity_type': 'candidate'} for i in range(11)]
    scores = list(range(11))
    assert rank_ids(rows, scores) == ['0', '1', '2']
    assert rank_ids(rows[::-1], scores[::-1]) == ['0', '1', '2']
