import numpy as np
from scripts import pae_sequence_order_v5 as v5


def test_order_encoding_breaks_composition_invariance():
    a,b='ACDA','ADCA'
    assert sorted(a)==sorted(b)
    assert not np.array_equal(v5.order_features(a),v5.order_features(b))
    assert not np.array_equal(v5.bigrams(a),v5.bigrams(b))
    x=v5.order_features(a).reshape(2,32,20)
    assert x.sum()==2*len(a)
    assert ''.join(v5.AA[j] for j in x[0,:len(a)].argmax(axis=1))==a


def test_ties_are_chance_and_direction_is_low_pae():
    rows=[{'target':float(i),'sem':0} for i in range(12)]
    m=v5.metrics(rows,np.ones(12))
    assert m['spearman'] is None and m['top20_recall']==.25
    assert m['reliable_pair_accuracy']==.5
    assert v5.metrics(rows,np.arange(12))['top20_recall']==1


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


def test_generator_only_cannot_rank_within_generator():
    report=v5.read(v5.OUT/'report.json')
    m=report['models']['generator_mean_train_only']['within_generator']
    assert m['median_rho'] is None
    assert m['macro_reliable_pair_accuracy']==.5
    assert m['groups']==24
    assert sum(c['reliable_pairs'] for c in m['cells'].values())==25
    assert report['new_independent_validation'] is False
