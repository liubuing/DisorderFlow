import numpy as np
import pytest
import torch

from scripts.harden_pae_surrogate_v4 import (
    directional_target, screening, evaluate, ScalarHead, OUT, read,
)


def test_directional_target_excludes_framework_and_reverse_direction():
    pae = np.full((9,9),31.0)
    # Heavy ACD, light EF, antigen GHIK. H3 C is heavy index 1.
    pae[1,5:] = 6.2
    pae[5:,1] = 24.8
    assert directional_target(pae,'ACD','EF','C','GHIK') == pytest.approx(0.2)
    with pytest.raises(ValueError):
        directional_target(pae,'ACA','EF','A','GHIK')


def test_tied_screening_is_chance_and_order_invariant():
    y = np.arange(14,dtype=float)
    out = screening(np.ones(14),y,0.2)
    assert out['selected']==3
    assert out['top20_recall']==pytest.approx(3/14)
    assert screening(np.ones(14),y[::-1],0.2)==out
    assert screening(y,y,0.2)['top20_recall']==1
    assert screening(-y,y,0.2)['top20_recall']==0


def test_positive_correlation_is_correct_direction_for_nll_and_pae():
    rows = [{'scaffold':'a','target':float(i),'sem':0} for i in range(4)]
    assert evaluate(rows,[0,1,2,3])['median_scaffold_spearman']==1
    reverse = evaluate(rows,[3,2,1,0])
    assert reverse['median_scaffold_spearman']==-1
    assert reverse['scaffolds']['a']['pair_sensitivity']['1']['accuracy']==0


def test_no_antigen_ablation_cannot_access_context():
    torch.manual_seed(10)
    head = ScalarHead(no_antigen=True).eval()
    x = torch.randn(4,256)
    changed = x.clone()
    changed[:,64:] += 100
    assert torch.equal(head(x),head(changed))


def test_revision_artifact_split_and_feature_provenance():
    path = OUT/'data_manifest.json'
    if not path.exists():
        pytest.skip('Experiment not materialized')
    data = read(path)
    rows = data['rows']
    assert len(rows)==356
    assert not {'V2C001','V2C002','V2C003'} & {r['scaffold'] for r in rows}
    splitsets = [{r['scaffold'] for r in rows if r['split']==s} for s in ('train','calibration','transfer')]
    assert [len(s) for s in splitsets]==[15,2,6]
    assert not (splitsets[0]&splitsets[1] or splitsets[0]&splitsets[2] or splitsets[1]&splitsets[2])
    assert all('/generation_work/structures/' in p['pdb'] for p in data['provenance'])
    assert all(np.isfinite(r['target']) and r['sem']>=0 for r in rows)


def test_pooled_branch_native_shuffle_scores_are_order_blind():
    path = OUT/'predictions.json'
    if not path.exists():
        pytest.skip('Experiment not materialized')
    data = read(path)
    for seed in (2041,2053,2069):
        scores = dict(zip(data['entities'],data['models'][f'full_s{seed}']))
        for i in range(1,7):
            native = scores[f'EXT{i:03d}|control|native']
            shuffle = scores[f'EXT{i:03d}|control|composition_shuffle']
            assert native == pytest.approx(shuffle,abs=2e-6)
