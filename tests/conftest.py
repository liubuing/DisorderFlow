"""Pytest configuration for DisorderFlow.

Tests are split:
  - default: CPU-only, no checkpoints, no GPU — pure-function logic.
  - @pytest.mark.gpu / @pytest.mark.slow: deselected by default (see pyproject.ini_options).

The repo root is added to sys.path so `modules/*.py` and `disorderflow/` import
the way the scripts do (sys.path.insert(0,'.'); sys.path.insert(0,'modules')).
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# `modules/` is imported as a flat package by the design scripts.
MODULES = os.path.join(ROOT, "modules")
if os.path.isdir(MODULES) and MODULES not in sys.path:
    sys.path.insert(0, MODULES)


def pytest_collection_modifyitems(config, items):
    """Skip gpu/slow tests unless explicitly requested via -m."""
    # 守护默认测试面的轻量性：gpu/slow 用例默认打跳过标记，
    # 保证纯 CPU 逻辑测试可以随时快速全量运行，不被环境依赖阻塞。
    skip_gpu = pytest.mark.skip(reason="gpu test — run with -m gpu")
    skip_slow = pytest.mark.skip(reason="slow test — run with -m slow")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# --- optional-dependency collection guards ---------------------------------
# CI installs only .[dev]. These modules exercise local OpenMM/ANARCII
# workflows and import those packages at module scope; skip collection when
# the heavy optional dependency is absent instead of failing the whole suite.
import importlib.util as _importlib_util

collect_ignore = []
if _importlib_util.find_spec("openmm") is None:
    collect_ignore += [
        "test_h3_t1_ensemble.py",
        "test_h3_t2_recovery.py",
    ]
if _importlib_util.find_spec("anarcii") is None:
    collect_ignore += [
        "test_idp_ensemble_development_extension_audit.py",
        "test_idp_ensemble_expanded_scorer_inputs_v2.py",
        "test_idp_ensemble_expanded_structural_v2.py",
    ]
