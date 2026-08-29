#!/usr/bin/env python
"""Run the standard trainer with StateContrast-v2 registrations enabled."""

import runpy
import sys
import json
from pathlib import Path

import yaml

# Importing executes the registration decorators without changing the frozen
# default model or dataset implementations.
import disorderflow.datasets.statecontrast_v2_structural  # noqa: F401
import disorderflow.datasets.statecontrast_v2_pose_manifest  # noqa: F401
import disorderflow.models.statecontrast_v2  # noqa: F401
import disorderflow.utils.data as data_utils
from disorderflow.utils.source_balanced_sampler import (
    SourceBalancedCompleteGroupBatchSampler,
)

# train.py imports this symbol from data_utils. The process-local replacement
# enables source balancing without modifying the historical default trainer.
data_utils.CompleteGroupBatchSampler = SourceBalancedCompleteGroupBatchSampler

# Refuse to train when the explicit-state/data-isolation audit is missing or
# blocked. The first positional argument is the standard trainer config path.
if len(sys.argv) < 2:
    raise SystemExit("Usage: train_statecontrast_v2.py <config> [train options]")
config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
readiness_path = Path(config["dataset"]["readiness_artifact"])
if not readiness_path.is_file():
    raise RuntimeError(f"Missing StateContrast-v2 readiness artifact: {readiness_path}")
readiness = json.loads(readiness_path.read_text(encoding="ascii"))
if readiness.get("status") != "statecontrast_v2_expanded_training_ready":
    raise RuntimeError(
        f"StateContrast-v2 training is blocked: {readiness.get('checks')}")
isolation_path = Path(config["dataset"]["isolation_artifact"])
if not isolation_path.is_file():
    raise RuntimeError(f"Missing StateContrast-v2 isolation audit: {isolation_path}")
isolation = json.loads(isolation_path.read_text(encoding="ascii"))
if isolation.get("status") != "statecontrast_v2_cv_isolation_passed":
    raise RuntimeError(f"StateContrast-v2 split isolation failed: {isolation.get('leaks')}")
runpy.run_path("train.py", run_name="__main__")
