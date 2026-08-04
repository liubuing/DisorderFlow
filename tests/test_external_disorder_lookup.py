import pickle

import numpy as np
import pytest

from disorderflow.datasets.disorder_augmented import load_disorder_lookup
from scripts.build.build_disorder_lookup_experimental import ids_digest, sequence_profile


def test_sequence_profile_is_finite_and_bounded():
    profile = sequence_profile(np.arange(20))
    assert profile.shape == (20,)
    assert np.isfinite(profile).all()
    assert ((profile >= 0) & (profile <= 1)).all()


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
    assert set(load_disorder_lookup(path, 'train', ids, True)) == set(ids)
    with pytest.raises(ValueError, match='split mismatch'):
        load_disorder_lookup(path, 'val', ids, True)
    with pytest.raises(ValueError, match='fingerprint mismatch'):
        load_disorder_lookup(path, 'train', ['a'], True)


def test_raw_lookup_rejected_when_envelope_required(tmp_path):
    path = tmp_path / 'raw.pkl'
    path.write_bytes(pickle.dumps({'a': np.zeros(2)}))
    with pytest.raises(ValueError, match='split-scoped'):
        load_disorder_lookup(path, require_envelope=True)
