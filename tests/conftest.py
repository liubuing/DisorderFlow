"""Pytest configuration for DisorderFlow.

Tests are split:
  - default: CPU-only, no checkpoints, no GPU — pure-function logic.
  - @pytest.mark.gpu / @pytest.mark.slow: deselected by default (see pyproject.ini_options).

The repo root is added to sys.path so `modules/*.py` and `disorderflow/` import
the way the scripts do (sys.path.insert(0,'.'); sys.path.insert(0,'modules')).
"""
import os
import sys
from pathlib import Path

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

# --- local-data collection guards ------------------------------------------
# The scientific-contract tests read frozen local artifacts (data/, results/,
# outputs/) that are intentionally not tracked in Git. A clean CI checkout has
# none of them, so skip those modules instead of failing with FileNotFoundError.
if not (Path(ROOT) / "data" / "anti_abeta_refs" / "4HIX.pdb").exists():
    collect_ignore += [
        "test_aayl_failure_v3.py",
        "test_aayl_learnability.py",
        "test_aayl_low_order.py",
        "test_aayl_remeasurement.py",
        "test_abeta_4hix_af2_gate.py",
        "test_bfn_loader_antigen_roles.py",
        "test_candidate_interface_dataset_builder.py",
        "test_candidate_interface_multiscaffold_builder.py",
        "test_csn_c0_contracts.py",
        "test_ecls_submission.py",
        "test_esel_e0b_contracts.py",
        "test_flex_j0b_contracts.py",
        "test_followup_scientific_contracts.py",
        "test_h3_publication_package.py",
        "test_ibec_tau_candidates.py",
        "test_ibec_tau_sanity.py",
        "test_idp_ensemble_bfn_state_compatibility.py",
        "test_idp_ensemble_correction_v3.py",
        "test_idp_ensemble_expanded_rcsb_v2.py",
        "test_idp_ensemble_final_confirmation_v1.py",
        "test_idp_ensemble_matched_generation.py",
        "test_idp_ensemble_matched_scoring.py",
        "test_idp_level2_computational_validation.py",
        "test_idp_level2_stability_abstention.py",
        "test_multiscaffold_v2_generation.py",
        "test_pae_dense_v6.py",
        "test_pae_sequence_order_v5.py",
        "test_publication_alignment.py",
        "test_release_lineage.py",
        "test_runtime_environment.py",
        "test_statecontrast_v2_fold_loaders.py",
        "test_successor_v3.py",
        "test_successor_v3_confirmatory_contract.py",
        "test_successor_v3_exploratory.py",
        "test_successor_v3_future_snapshot_runner.py",
        "test_target_design_helpers.py",
        "test_v5_1_expression_package.py",
    ]

# --- Python 3.10 hashlib compatibility -------------------------------------
# hashlib.file_digest() was added in 3.11; the frozen validators call it.
# Patch it at test time instead of editing the hash-referenced validators.
import hashlib as _hashlib

if not hasattr(_hashlib, "file_digest"):
    def _file_digest(fileobj, digest_name):
        return _hashlib.new(digest_name, fileobj.read())
    _hashlib.file_digest = _file_digest
