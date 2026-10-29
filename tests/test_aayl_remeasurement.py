"""AAYL 复测准备审计：验证先导实验设计的完全析因配对与复测数据溯源。

守护 AAYL 复测（re-measurement）准备管线的两条不变量：
先导设计的 40 个构建体严格等于所选三元组的全因子组合（且模板全部待测）、
复测记录只允许混入同 assay 且父本组完整的匹配——防止跨 assay 混杂或重复父本污染回收统计。
"""
import json
from scripts import prepare_aayl_remeasurement as prep


# 先导设计必须是所选三元组突变位点的完全析因（factorial）展开，
# 且读数模板 120 条记录全部处于 pending——防止设计缺组合或预填结果。
def test_pilot_complete_factorial_and_pairing():
    d,rows,changes=prep.load()
    design=json.loads((prep.OUT/'pilot_design.json').read_text())
    by_id={c['construct_id']:c for c in design['constructs']}
    assert len(by_id)==design['unique_constructs']==40
    assert all(c['measurement_status']=='not_measured_in_this_pilot' for c in by_id.values())
    for choice in design['selected_triples']:
        i=next(i for i,r in enumerate(rows) if r['poi']==choice['triple_poi'])
        s=d['structures'][choice['family']]
        actual={by_id[k]['heavy_sequence'] for k in choice['factorial_construct_ids']}
        assert actual==prep.factorial(s['sequences'][s['heavy_chain']],changes[i])
        assert all(by_id[k]['light_sequence']==s['sequences'][s['light_chain']] for k in choice['factorial_construct_ids'])
    template=json.loads((prep.OUT/'blank_readout_template.json').read_text())['records']
    assert len(template)==120
    assert all(r['estimate'] is None and r['readout_status']=='pending' for r in template)


# 复测可回收性只能由"同一原始 assay 且父本三次重复完整"的唯一匹配决定，
# 且来源文件、assay 编号须严格固定——防止跨 assay 或重复父本混入回收记录。
def test_recovery_never_mixes_assays_or_duplicate_parent_groups():
    recovery=json.loads((prep.OUT/'source_recovery.json').read_text())
    assert len(recovery['records'])==748
    for r in recovery['records']:
        same=[g for g in r['matches'] if g['same_original_assay']]
        assert r['recoverable_same_assay']==(len(same)==1 and same[0]['complete_original_three_replicates'])
        for g in same:
            assert g['assay']==('1' if r['family']=='AAYL49' else '2')
            assert g['source']=='MITLL_AAlphaBio_Ab_Binding_dataset.csv.zip'
