#!/usr/bin/env python
"""Single command surface for the active StateContrast-v2 workflow."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMANDS = {
    "readiness": "scripts/audit_statecontrast_v2_expanded_readiness.py",
    "prepare-states": "scripts/prepare_statecontrast_v2_explicit_states.py",
    "build-manifest": "scripts/build_statecontrast_v2_manifest.py",
    "prepare-cv": "scripts/prepare_statecontrast_v2_cross_validation.py",
    "audit-cv-isolation": "scripts/audit_statecontrast_v2_cv_isolation.py",
    "audit-fold-loaders": "scripts/audit_statecontrast_v2_fold_loaders.py",
    "run-cv": "scripts/run_statecontrast_v2_cross_validation.py",
    "evaluate": "scripts/evaluate_statecontrast_v2_checkpoint.py",
    "aggregate-oof": "scripts/aggregate_statecontrast_v2_oof.py",
    "generate": "scripts/generate_statecontrast_v2_candidates.py",
    "coverage": "scripts/audit_idp_ensemble_coverage.py",
    "developability": "scripts/audit_developability_measurements.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS))
    args, remaining = parser.parse_known_args()
    script = ROOT / COMMANDS[args.command]
    sys.argv = [str(script), *remaining]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
