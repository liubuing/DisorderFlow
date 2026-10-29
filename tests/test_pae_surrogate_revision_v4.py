"""PAE 代理模型修订 v4 测试：验证方向性目标、平局筛选、消融隔离与产物溯源。

守护 scripts.harden_pae_surrogate_v4 的代理强化管线：
directional_target 只取 H3→抗原方向且 H3 必须真实存在于重链、
全平局筛选等于随机抽样且顺序不变、spearman 方向与配对敏感度符号正确、
no_antigen 消融头不得访问上下文特征、
产物清单的拆分/特征溯源与 shuffle 对照的顺序盲性。
"""
import numpy as np
import pytest
import torch

from scripts.harden_pae_surrogate_v4 import (
    directional_target, screening, evaluate, ScalarHead, OUT, read,
)


# 方向性目标：只取 H3 残基到抗原块的正向 PAE（0.2 而非反向的 24.8），
# H3 不在重链中必须报错——防止方向取反或伪 H3 残基混入。
def test_directional_target_excludes_framework_and_reverse_direction():
    pae = np.full((9,9),31.0)
    # Heavy ACD, light EF, antigen GHIK. H3 C is heavy index 1.
    pae[1,5:] = 6.2
    pae[5:,1] = 24.8
    assert directional_target(pae,'ACD','EF','C','GHIK') == pytest.approx(0.2)
    with pytest.raises(ValueError):
        directional_target(pae,'ACA','EF','A','GHIK')


# 平局筛选：全平局时选中数=随机期望、top20_recall=3/14、顺序不变；
# 正向完美=1、反向=0——防止平局处理有偏或方向写反。
def test_tied_screening_is_chance_and_order_invariant():
    y = np.arange(14,dtype=float)
    out = screening(np.ones(14),y,0.2)
    assert out['selected']==3
    assert out['top20_recall']==pytest.approx(3/14)
    assert screening(np.ones(14),y[::-1],0.2)==out
    assert screening(y,y,0.2)['top20_recall']==1
    assert screening(-y,y,0.2)['top20_recall']==0


# 相关性方向：升序预测 spearman=1、降序=-1 且配对敏感度精度为 0——
# 防止 NLL/PAE 的"越低越好"方向被写反。
def test_positive_correlation_is_correct_direction_for_nll_and_pae():
    rows = [{'scaffold':'a','target':float(i),'sem':0} for i in range(4)]
    assert evaluate(rows,[0,1,2,3])['median_scaffold_spearman']==1
    reverse = evaluate(rows,[3,2,1,0])
    assert reverse['median_scaffold_spearman']==-1
    assert reverse['scaffolds']['a']['pair_sensitivity']['1']['accuracy']==0


# 消融隔离：no_antigen 头对上下文段的大扰动必须输出完全相同——
# 防止消融模型偷偷访问抗原上下文特征。
def test_no_antigen_ablation_cannot_access_context():
    torch.manual_seed(10)
    head = ScalarHead(no_antigen=True).eval()
    x = torch.randn(4,256)
    changed = x.clone()
    changed[:,64:] += 100
    assert torch.equal(head(x),head(changed))


# 产物溯源审计：356 行数据、旧 V2 脚手架不得回流、
# train/calibration/transfer 支架数 15/2/6 且互斥（拆分隔离）、
# PDB 路径必须出自 generation_work、target 有限且 sem 非负——防止脏数据入表。
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


# 顺序盲性守护：pooled 分支对天然对照与组成打乱对照的打分必须几乎相同——
# 防止该分支隐式使用序列顺序特征（否则声明为顺序盲就不成立）。
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
