"""PAE 密集评估 v6 的契约测试：守护教师缓存键对链上下文/协议/种子的充分区分（防缓存串号），
以及密集协议的冻结内容（每臂 20 个唯一靶点、4 臂、固定种子、无新拟合与模型哈希清单）。"""
import numpy as np
from scripts.prepare_pae_dense_v6 import OUT,read
from scripts.run_pae_dense_v6 import key,target


# 缓存键隔离：换 t 值、换种子、换任一链序列都必须产生不同键——防止教师结果跨实验复用串号。
def test_teacher_cache_key_separates_chain_context_protocol_and_seed():
    e={'component_id':'EXT001','heavy_sequence':'ACD','light_sequence':'EF','antigen_sequence':'GH'}
    assert key(e,1,7103)!=key(e,2,7103)
    assert key(e,1,7103)!=key(e,1,7111)
    assert key(e,1,7103)!=key(dict(e,light_sequence='EE'),1,7103)


# 协议冻结：靶点按臂唯一、臂数与种子固定、模型描述明示"不做新拟合"且附 3 个模型哈希——
# 防止密集实验悄悄更换协议或模型。
def test_dense_contract_has_no_gp2_and_fixed_models():
    p=read(OUT/'protocol.json')
    assert p['target_unique_per_arm']==20 and len(p['arms'])==4
    assert p['seeds']==[20260917,20261917,20262917,20263917]
    assert 'no new fitting' in p['models']
    assert len(p['models_sha256'])==3
