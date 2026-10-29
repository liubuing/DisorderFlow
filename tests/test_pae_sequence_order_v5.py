"""PAE 序列顺序编码 v5 测试：验证顺序特征打破组成不变性、平局方向与选择隔离。

守护 scripts.pae_sequence_order_v5 的打分与选择管线：
order_features/bigrams 必须区分组成相同但顺序不同的序列、
平局指标等于随机期望且方向为低 PAE 更优（PAE 越低越好）、
模型选择不得访问选择集之外的迁移标签（隔离性）、
generator-only 基线在生成器内部无排序能力（median_rho=None、pair accuracy=0.5）。
"""
import numpy as np
from scripts import pae_sequence_order_v5 as v5


# 组成不变性破解：ACDA 与 ADCA 组成相同但特征必须不同，
# 且 one-hot 编码可逆解码回原序列——防止模型退化为组成计数器。
def test_order_encoding_breaks_composition_invariance():
    a,b='ACDA','ADCA'
    assert sorted(a)==sorted(b)
    assert not np.array_equal(v5.order_features(a),v5.order_features(b))
    assert not np.array_equal(v5.bigrams(a),v5.bigrams(b))
    x=v5.order_features(a).reshape(2,32,20)
    assert x.sum()==2*len(a)
    assert ''.join(v5.AA[j] for j in x[0,:len(a)].argmax(axis=1))==a


# 平局与方向：全平局时 spearman 为 None、top20_recall 等于随机期望（0.25）、
# 可信配对准确率恰为 0.5；完美排序（低 PAE 在前）时 top20_recall=1——
# 防止方向写反或平局灌水。
def test_ties_are_chance_and_direction_is_low_pae():
    rows=[{'target':float(i),'sem':0} for i in range(12)]
    m=v5.metrics(rows,np.ones(12))
    assert m['spearman'] is None and m['top20_recall']==.25
    assert m['reliable_pair_accuracy']==.5
    assert v5.metrics(rows,np.arange(12))['top20_recall']==1


# 选择隔离：篡改校准集之外的测试行标签不得改变所选特征与模型预测——
# 防止模型选择阶段读取迁移/评估标签（信息泄漏）。
def test_selection_has_no_transfer_label_access():
    rng=np.random.default_rng(17)
    x=rng.normal(size=(40,8))
    rows=[{'target':float(x[i,0]),'sem':0.,'entity_type':'candidate','scaffold':str(i//10)} for i in range(40)]
    tr,cal=np.arange(20),np.arange(20,30)
    model,selected=v5.choose(x,rows,tr,cal)
    altered=[dict(r) for r in rows]
    for i in range(30,40): altered[i]['target']=1000-i
    model2,selected2=v5.choose(x,altered,tr,cal)
    assert selected==selected2
    np.testing.assert_array_equal(model.predict(x),model2.predict(x))


# 生成器内部排序审计：generator_mean_train_only 在生成器组内必须无排序能力
# （median_rho=None、配对准确率=0.5）且独立验证标记为 False——
# 防止把生成器自身偏好误读为泛化排序能力。
def test_generator_only_cannot_rank_within_generator():
    report=v5.read(v5.OUT/'report.json')
    m=report['models']['generator_mean_train_only']['within_generator']
    assert m['median_rho'] is None
    assert m['macro_reliable_pair_accuracy']==.5
    assert m['groups']==24
    assert sum(c['reliable_pairs'] for c in m['cells'].values())==25
    assert report['new_independent_validation'] is False
