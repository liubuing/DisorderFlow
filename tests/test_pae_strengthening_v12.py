import numpy as np
from scripts.strengthen_pae_v12 import recall


def test_fixed_target_variable_budget():
    ids = list(map(str,range(20)))
    a = recall(ids,list(range(20)),list(range(20)),.1)
    assert a['target_count']==4 and a['budget']==2 and a['recall']==.5


def test_constant_scores_have_random_expected_recall():
    for budget in (.1,.2,.3,.5):
        a=recall(list(map(str,range(20))),[0]*20,list(range(20)),budget)
        assert np.isclose(a['tie_averaged_recall'],budget)
        assert a['spearman'] is None


def test_partial_cutoff_tie():
    a=recall(['a','b','c','d','e'],[0,1,1,1,2],[4,0,1,2,3],.4)
    assert np.isclose(a['tie_averaged_recall'],1/3)
