import numpy as np

from scripts.benchmark_h3_t2_recovery import state_advantage


def test_state_advantage_requires_real_runner(monkeypatch, tmp_path):
    logp = np.zeros((1, 21), dtype=float)
    logp[0, 0] = -1.0

    def fake_run(*args, **kwargs):
        return logp, {}, []

    monkeypatch.setattr("scripts.benchmark_h3_t2_recovery.run_conditional_probs", fake_run)
    observed = state_advantage(
        {}, [tmp_path / "a.pdb"], ["H", "L"], "A", ["C"], [0], tmp_path)
    assert observed["native_mean_ecls"] == 0.0
    assert observed["ecls_advantage"] == 0.0
