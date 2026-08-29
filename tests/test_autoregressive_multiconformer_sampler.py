import numpy as np

from scripts.autoregressive_multiconformer_sampler import (
    aggregate_pose_log_probs,
    sample_autoregressive,
)


class FakePose:
    def __init__(self, preferred):
        self.preferred = preferred
        self.calls = []

    def next_log_probs(self, prefix, position):
        self.calls.append((prefix, position))
        values = np.full(3, -4.0)
        values[self.preferred[position]] = -0.1
        return values


def test_aggregate_pose_log_probs_is_equal_weight_by_default():
    result = aggregate_pose_log_probs([[0.0, -2.0], [-2.0, 0.0]])
    np.testing.assert_allclose(result, [-1.0, -1.0])


def test_sampler_recomputes_each_pose_for_each_prefix():
    models = [FakePose([0, 1]), FakePose([1, 0])]
    sequence = sample_autoregressive(
        "AA", models, np.random.default_rng(3), 2, "ABC"
    )
    assert len(sequence) == 2
    assert models[0].calls == [("", 0), (sequence[0], 1)]
    assert models[1].calls == [("", 0), (sequence[0], 1)]


def test_sampler_accepts_single_batched_pose_adapter():
    class Batched:
        pose_count = 3

        def next_log_probs(self, prefix, position):
            values = np.full(4, -3.0)
            values[0] = -0.2
            return values

    sequence = sample_autoregressive(
        "AA", [Batched()], np.random.default_rng(1), 1, "ABCD"
    )
    assert len(sequence) == 2


def test_sampler_rejects_empty_pose_models():
    import pytest

    with pytest.raises(ValueError, match="at least one pose model"):
        sample_autoregressive("AA", [], np.random.default_rng(1), 1, "ABCD")
