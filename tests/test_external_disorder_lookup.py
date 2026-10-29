"""外部无序度查找表测试：验证剖面值域、拆分封套 fail-closed 与裸查找拒绝。

守护外部无序度数据接入（disorder_lookup）管线：
sequence_profile 必须有限且落在 [0,1]、
加载时严格校验 split 名称与 ids 指纹（不匹配即报错，fail-closed）、
缺 split 封套的裸 pickle 查找表必须拒绝——防止跨拆分泄漏与不可溯源数据进入训练。
"""
import pickle

import numpy as np
import pytest

from disorderflow.datasets.disorder_augmented import load_disorder_lookup
from scripts.build.build_disorder_lookup_experimental import ids_digest, sequence_profile


# 剖面生成器：输出形状与输入一致且全部有限、有界于 [0,1]——防止 NaN/越界值污染监督信号。
def test_sequence_profile_is_finite_and_bounded():
    profile = sequence_profile(np.arange(20))
    assert profile.shape == (20,)
    assert np.isfinite(profile).all()
    assert ((profile >= 0) & (profile <= 1)).all()


# 拆分溯源 fail-closed：split 不匹配或 ids sha256 指纹不符必须报错，
# 匹配时缺失条目以单位置信度兜底——防止 train/val 数据混用或身份不明的查找表被加载。
def test_split_artifact_fails_closed(tmp_path):
    ids = ['a', 'b']
    artifact = {
        'schema_version': 1,
        'split': 'train',
        'ids_sha256': ids_digest(ids),
        'profiles': {key: np.zeros(2, dtype=np.float32) for key in ids},
    }
    path = tmp_path / 'lookup.pkl'
    path.write_bytes(pickle.dumps(artifact))
    loaded = load_disorder_lookup(path, 'train', ids, True)
    assert set(loaded) == set(ids)
    assert np.allclose(loaded['a']['confidence'], 1.0)
    with pytest.raises(ValueError, match='split mismatch'):
        load_disorder_lookup(path, 'val', ids, True)
    with pytest.raises(ValueError, match='fingerprint mismatch'):
        load_disorder_lookup(path, 'train', ['a'], True)


# 旧式裸查找表（无 split 封套）在 require_envelope 模式下必须拒绝——
# 防止绕过拆分隔离的历史产物静默回流。
def test_raw_lookup_rejected_when_envelope_required(tmp_path):
    path = tmp_path / 'raw.pkl'
    path.write_bytes(pickle.dumps({'a': np.zeros(2)}))
    with pytest.raises(ValueError, match='split-scoped'):
        load_disorder_lookup(path, require_envelope=True)
