"""AAYL 失败诊断 v3 的科学契约测试：守护 H3 突变编码的精确性、内外拆分的隔离性、
加性正对照的可学习性，以及修复面板对三突变候选的子集覆盖完整性。"""
import json
import numpy as np
from scripts import diagnose_aayl_failure_v3 as v3


# 编码必须无损：one-hot 块中非零元素数恰为 2*突变数（位置+残基各一），且每行求和为零（差异编码），
# 并能从编码精确还原突变集合，防止特征编码引入信息丢失或错位。
def test_h3_encoding_preserves_exact_mutations():
    d,rows,changes=v3.base.load()
    for family in d['structures']:
        x=v3.features(d,rows,family)
        s=d['structures'][family]
        for i,r in enumerate(rows):
            if r['family']!=family: continue
            blocks=x[i].reshape(-1,20)
            assert np.count_nonzero(blocks)==2*len(changes[i])
            assert np.all(blocks.sum(axis=1)==0)
            recovered={(s['h3_indices_zero_based_observed_heavy'][j],v3.base.AA[a]) for j,a in zip(*np.where(blocks==1))}
            assert recovered==changes[i]


# 拆分溯源：外层测试集必须与所有内层训练/验证折互斥，且难度偏移折（shift）保证内层验证难度严格更高，
# 防止评估阶段的候选泄漏导致指标虚高。
def test_outer_test_excluded_from_all_inner_validation():
    _,rows,changes=v3.base.load()
    splits=json.loads((v3.OUT/'splits.json').read_text())['splits']
    for s in splits:
        tr=np.array(s['train']); te=set(s['test'])
        assert not set(tr)&te
        for shift in [False,True]:
            deg=np.array([len(changes[i]) for i in tr])
            for a,b in v3.inner_splits(deg,shift):
                assert not set(tr[a])&set(tr[b])
                assert not (set(tr[a])|set(tr[b]))&te
                if shift: assert max(deg[a])<min(deg[b])


# 正对照：纯加性信号应被拟合到 rho>0.99，且对测试集整体加常数不得改变选中的 alpha——
# 守护"超参选择只依赖训练数据"这一管线不变量。
def test_additive_positive_control_learns_and_test_change_does_not_select_alpha():
    rng=np.random.default_rng(31)
    x=rng.normal(size=(120,8)); y=x@np.arange(1.,9.)
    deg=np.tile([1,2],50)
    p,a=v3.fit_model(x[:100],y[:100],deg,x[100:],'position','wide_rank')
    _,b=v3.fit_model(x[:100],y[:100],deg,x[100:]+1000,'position','wide_rank')
    assert a==b
    assert v3.rho(y[100:],p)>.99


# 修复面板完备性：对每个三突变候选，其全部 0/1/2 突变子集序列必须已被测量或在补测面板中，
# 且面板不与已有单/双突变测量重复——保证加性分解分析的数据基础完整无缺口。
def test_repair_panel_completes_exact_single_and_pair_controls():
    from itertools import combinations
    d,rows,changes=v3.base.load()
    panel=json.loads((v3.OUT/'measurement_repair_panel.json').read_text())['records']
    assert len({(p['family'],p['heavy_sequence']) for p in panel})==len(panel)
    for family,s in d['structures'].items():
        available={r['heavy_sequence'] for i,r in enumerate(rows) if r['family']==family and len(changes[i])<=2}
        supplied={p['heavy_sequence'] for p in panel if p['family']==family}
        assert not available&supplied
        for i,r in enumerate(rows):
            if r['family']!=family or len(changes[i])!=3: continue
            for size in [0,1,2]:
                for subset in combinations(sorted(changes[i]),size):
                    seq=list(s['sequences'][s['heavy_chain']])
                    for pos,aa in subset: seq[pos]=aa
                    assert ''.join(seq) in available|supplied
