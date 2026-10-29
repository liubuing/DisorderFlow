"""PAE 强化 v12 召回指标的契约测试：守护固定靶点比例下预算与召回的换算、恒定分数的平局平均
召回必须等于随机期望（预算本身）且无 Spearman，以及部分越线平局的平均处理——防止平局被当作命中。"""
import numpy as np
from scripts.strengthen_pae_v12 import recall


# 预算换算：20 条、10% 靶点比例得 4 个目标、20% 预算得 2 个名额、召回 0.5——
# 防止预算取整与召回分母口径错位。
def test_fixed_target_variable_budget():
    ids = list(map(str,range(20)))
    a = recall(ids,list(range(20)),list(range(20)),.1)
    assert a['target_count']==4 and a['budget']==2 and a['recall']==.5


# 平局即随机：全平局时平局平均召回必须等于预算比例（随机期望），且恒定分数不得报告 Spearman——
# 防止无信息打分冒充筛选能力。
def test_constant_scores_have_random_expected_recall():
    for budget in (.1,.2,.3,.5):
        a=recall(list(map(str,range(20))),[0]*20,list(range(20)),budget)
        assert np.isclose(a['tie_averaged_recall'],budget)
        assert a['spearman'] is None


# 跨线平局：预算切割线上的平局成员按平均贡献计（1/3），防止整组平局被全额计入命中。
def test_partial_cutoff_tie():
    a=recall(['a','b','c','d','e'],[0,1,1,1,2],[4,0,1,2,3],.4)
    assert np.isclose(a['tie_averaged_recall'],1/3)
