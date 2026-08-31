import pickle

import lmdb
import pytest
import torch

from disorderflow.datasets.confidence_dataset import ConfidenceRegressionDataset
from disorderflow.utils.protein.constants import Fragment


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


def _candidate_interface_entry(schema="candidate_interface_confidence_v1"):
    return {
        "schema_version": schema,
        "sequence": "AAA",
        "batch": {
            "aa": torch.tensor([0, 1, 2]),
            "generate_flag": torch.tensor([True, False, False]),
            "fragment_type": torch.tensor([
                int(Fragment.Heavy), int(Fragment.Light), int(Fragment.Antigen)]),
        },
        "af2_plddt": torch.tensor([0.8, 0.7, 0.6]),
        "af2_iptm": torch.tensor(0.5),
        "af2_pae_matrix": torch.tensor([
            [0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]]),
        "af2_pae_normalized": False,
    }


def test_candidate_interface_dataset_preserves_full_complex_contract(tmp_path):
    path = tmp_path / "successor"
    path.mkdir()
    with (path / "00000000.pkl").open("wb") as handle:
        pickle.dump(_candidate_interface_entry(), handle)
    dataset = ConfidenceRegressionDataset({
        "db_path": str(path), "candidate_interface_v1": True})
    batch = dataset[0]
    assert torch.equal(
        batch["generate_flag"], torch.tensor([True, False, False]))
    assert batch["af2_pae_matrix"].shape == (3, 3)
    assert batch["af2_pae_normalized"].item() is False
    assert dataset.requires_complete_groups is True
    assert dataset.group_indices == [[0]]


def test_candidate_interface_dataset_attaches_stability_metadata(tmp_path):
    path = tmp_path / "weighted_successor"
    path.mkdir()
    entry = _candidate_interface_entry()
    entry.update({
        "confidence_sample_weight": 0.4,
        "af2_iptm_std": 0.06,
        "af2_candidate_plddt_std": 0.03,
        "af2_interface_pae_normalized_std": 0.08,
        "af2_iptm_sem": 0.024,
        "af2_candidate_plddt_sem": 0.012,
        "af2_interface_pae_normalized_sem": 0.032,
    })
    with (path / "00000000.pkl").open("wb") as handle:
        pickle.dump(entry, handle)
    dataset = ConfidenceRegressionDataset({
        "db_path": str(path), "candidate_interface_v1": True})
    batch = dataset[0]
    assert batch["confidence_sample_weight"].item() == pytest.approx(0.4)
    assert batch["af2_iptm_std"].item() == pytest.approx(0.06)
    assert batch["af2_candidate_plddt_std"].item() == pytest.approx(0.03)
    assert batch["af2_interface_pae_normalized_std"].item() == pytest.approx(0.08)
    assert batch["af2_iptm_sem"].item() == pytest.approx(0.024)
    assert batch["af2_candidate_plddt_sem"].item() == pytest.approx(0.012)
    assert batch["af2_interface_pae_normalized_sem"].item() == pytest.approx(0.032)


def test_candidate_interface_dataset_rejects_legacy_schema(tmp_path):
    path = tmp_path / "legacy"
    path.mkdir()
    with (path / "00000000.pkl").open("wb") as handle:
        pickle.dump(_candidate_interface_entry(schema=1), handle)
    dataset = ConfidenceRegressionDataset({
        "db_path": str(path), "candidate_interface_v1": True})
    with pytest.raises(ValueError, match="successor dataset schema"):
        dataset[0]
