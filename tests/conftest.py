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
    skip_gpu = pytest.mark.skip(reason="gpu test — run with -m gpu")
    skip_slow = pytest.mark.skip(reason="slow test — run with -m slow")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
