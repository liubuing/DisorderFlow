"""Compatibility entry point for the frozen ESM experiment on PyTorch >=2.6."""
import argparse
import runpy
import sys
from pathlib import Path

import torch


if __name__ == "__main__":
    # ESM's checkpoint contains argparse.Namespace. Keep weights_only safety;
    # allow only this standard-library metadata class, not arbitrary pickle.
    torch.serialization.add_safe_globals([argparse.Namespace])
    path = Path(__file__).with_name("benchmark_aayl_learnability.py")
    sys.argv[0] = str(path)
    runpy.run_path(str(path), run_name="__main__")
