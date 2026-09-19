import numpy as np
from scripts.prepare_pae_dense_v6 import OUT,read
from scripts.run_pae_dense_v6 import key,target


def test_teacher_cache_key_separates_chain_context_protocol_and_seed():
    e={'component_id':'EXT001','heavy_sequence':'ACD','light_sequence':'EF','antigen_sequence':'GH'}
    assert key(e,1,7103)!=key(e,2,7103)
    assert key(e,1,7103)!=key(e,1,7111)
    assert key(e,1,7103)!=key(dict(e,light_sequence='EE'),1,7103)


def test_dense_contract_has_no_gp2_and_fixed_models():
    p=read(OUT/'protocol.json')
    assert p['target_unique_per_arm']==20 and len(p['arms'])==4
    assert p['seeds']==[20260917,20261917,20262917,20263917]
    assert 'no new fitting' in p['models']
    assert len(p['models_sha256'])==3
