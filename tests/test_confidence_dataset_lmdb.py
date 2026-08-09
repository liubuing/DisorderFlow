import pickle

import lmdb

from disorderflow.datasets.confidence_dataset import ConfidenceRegressionDataset


def test_confidence_dataset_infers_length_without_metadata(tmp_path):
    path = tmp_path / "confidence.lmdb"
    env = lmdb.open(str(path), map_size=1_000_000)
    with env.begin(write=True) as transaction:
        transaction.put(b"00000000", pickle.dumps({"sequence": "AA"}))
        transaction.put(b"00000001", pickle.dumps({"sequence": "AAA"}))
    env.close()
    dataset = ConfidenceRegressionDataset({"db_path": str(path)})
    assert len(dataset) == 2
    dataset.close()


def test_confidence_datasets_share_readonly_environment(tmp_path):
    path = tmp_path / "shared.lmdb"
    env = lmdb.open(str(path), map_size=1_000_000)
    with env.begin(write=True) as transaction:
        transaction.put(b"00000000", pickle.dumps({"sequence": "AA"}))
    env.close()
    first = ConfidenceRegressionDataset({"db_path": str(path)})
    second = ConfidenceRegressionDataset({"db_path": str(path)})
    assert first._env is second._env
    first.close()
    assert len(second) == 1
    second.close()
