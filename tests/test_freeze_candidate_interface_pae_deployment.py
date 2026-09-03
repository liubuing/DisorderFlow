import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "freeze_candidate_interface_pae_deployment.py"
SPEC = importlib.util.spec_from_file_location("freeze_pae_deploy", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_fit_std_matching_reaches_unit_variance_ratio():
    pred = np.asarray([0.1, 0.2, 0.3, 0.4])
    target = pred * 2.5 + 0.3
    slope, intercept = MODULE.fit_std_matching(pred, target)
    corrected = slope * pred + intercept
    assert np.std(corrected) == pytest.approx(np.std(target), rel=1e-6)
    assert slope == pytest.approx(2.5)


def test_fit_std_matching_removes_mean_offset():
    pred = np.asarray([0.1, 0.2, 0.3])
    target = pred + 5.0
    slope, intercept = MODULE.fit_std_matching(pred, target)
    assert np.mean(slope * pred + intercept) == pytest.approx(np.mean(target), rel=1e-6)


def test_spearman_monotone():
    left = [1.0, 2.0, 3.0, 4.0]
    right = [10.0, 20.0, 30.0, 40.0]
    assert MODULE.spearman(left, right) == pytest.approx(1.0)
    assert MODULE.spearman(left, list(reversed(right))) == pytest.approx(-1.0)


def test_scaffold_spearman_groups_by_scaffold():
    pred = [0.1, 0.3, 0.5, 0.7, 0.9]
    target = [0.1, 0.2, 0.4, 0.6, 0.8]
    scaffolds = ["A", "A", "B", "B", "C"]
    result = MODULE.scaffold_spearman(pred, target, scaffolds)
    assert set(result) == {"A", "B"}
