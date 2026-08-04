import numpy as np
import torch

from scripts.evaluate_v5_1_phase3 import condition_batch


def test_mismatched_arm_preserves_factual_distribution():
    batch = {
        'epitope_disorder_profile': torch.tensor([[0.0, 0.1, 0.4, 0.9]]),
        'fragment_type': torch.tensor([[1, 3, 3, 3]]),
        'epitope_disorder': torch.tensor([[[0.0]]]),
    }
    factual = batch['epitope_disorder_profile'][0, 1:].clone()
    condition_batch(batch, 'mismatched', np.array([0.8, 0.1, 0.7, 0.2]))
    mismatch = batch['epitope_disorder_profile'][0, 1:]
    assert torch.allclose(factual.sort().values, mismatch.sort().values)
    assert not torch.allclose(factual, mismatch)
