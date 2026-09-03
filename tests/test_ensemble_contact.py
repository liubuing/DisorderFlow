import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).parents[1] / "modules" / "ensemble_contact.py"
SPEC = importlib.util.spec_from_file_location("ensemble_contact", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_robust_contacts_p25():
    summary = MODULE.robust_contacts([10, 20, 30, 40], "p25")
    assert summary["robust"] == pytest.approx(17.5)
    assert summary["min"] == 10.0
    assert summary["max"] == 40.0
    assert summary["n_missing"] == 0


def test_robust_contacts_min_and_median():
    assert MODULE.robust_contacts([10, 20, 30], "min")["robust"] == 10.0
    assert MODULE.robust_contacts([10, 20, 30], "median")["robust"] == 20.0


def test_robust_contacts_handles_missing():
    summary = MODULE.robust_contacts([5, None, 25], "p25")
    assert summary["robust"] == pytest.approx(10.0)
    assert summary["n_missing"] == 1
    assert summary["n_conformations"] == 3


def test_robust_contacts_all_missing():
    summary = MODULE.robust_contacts([None, None])
    assert summary["robust"] is None


def test_kabsch_align_translates_and_rotates():
    reference = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mobile = reference + np.array([5.0, 3.0, 2.0])
    aligned = MODULE._kabsch_align(mobile, reference)
    assert np.allclose(aligned, reference, atol=1e-6)


def test_kabsch_align_rotation_invariant():
    reference = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    theta = np.pi / 3
    rotation = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    mobile = (reference @ rotation.T) + np.array([1.0, -2.0, 0.5])
    aligned = MODULE._kabsch_align(mobile, reference)
    assert np.allclose(aligned, reference, atol=1e-6)
