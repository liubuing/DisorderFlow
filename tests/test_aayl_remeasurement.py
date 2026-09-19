import json
from scripts import prepare_aayl_remeasurement as prep


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


def test_recovery_never_mixes_assays_or_duplicate_parent_groups():
    recovery=json.loads((prep.OUT/'source_recovery.json').read_text())
    assert len(recovery['records'])==748
    for r in recovery['records']:
        same=[g for g in r['matches'] if g['same_original_assay']]
        assert r['recoverable_same_assay']==(len(same)==1 and same[0]['complete_original_three_replicates'])
        for g in same:
            assert g['assay']==('1' if r['family']=='AAYL49' else '2')
            assert g['source']=='MITLL_AAlphaBio_Ab_Binding_dataset.csv.zip'
