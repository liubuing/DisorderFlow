"""PAE 筛选 v7 排名选择的契约测试：守护对照实体排除、平局按组件代表性处理与预算取整、
以及选择对输入顺序不敏感——防止筛选名单随行序抖动或混入对照。"""
from scripts.pae_screening_next_v7 import rank_ids


# 候选过滤与平局：native 对照被排除；同组件平局（a、b 同分）保留排序在前的 a——
# 防止对照进入筛选或平局处理随机化。
def test_screen_excludes_controls_and_handles_ties_and_components():
    rows = [
        {'entity_id': 'b', 'component_id': 'A', 'entity_type': 'candidate'},
        {'entity_id': 'a', 'component_id': 'A', 'entity_type': 'candidate'},
        {'entity_id': 'native', 'component_id': 'A', 'entity_type': 'native'},
        {'entity_id': 'c', 'component_id': 'B', 'entity_type': 'candidate'},
    ]
    assert rank_ids(rows, [1, 1, -100, 5]) == ['a', 'c']


# 预算口径：11 条候选 30% 预算向上取整为 3（ceil），且整体翻转行序/分数后选择不变——
# 防止预算截断方向（floor/ceil）或顺序敏感性引入偏差。
def test_selection_budget_is_ceil_and_order_independent():
    rows = [{'entity_id': str(i), 'component_id': 'A', 'entity_type': 'candidate'} for i in range(11)]
    scores = list(range(11))
    assert rank_ids(rows, scores) == ['0', '1', '2']
    assert rank_ids(rows[::-1], scores[::-1]) == ['0', '1', '2']
