"""ProteinMPNN conditional-probability subprocess wrapper.

The flag set mirrors the canonical publication scorer
(scripts/benchmark_h3_epitope_delta.py::run_mpnn_state): conditional
log-probabilities on a fixed backbone, everything outside the given H3 indices
held fixed, seeded, one sequence per target. ``--score_only`` is deliberately
not used because the frozen results recomputed per-residue NLL in Python from
the saved ``log_p`` array; this tool keeps that arithmetic.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def score_backbone(
    mpnn_script: str,
    weights_dir: str,
    model_name: str,
    seed: int,
    pdb_path: Path,
    score_chain: str,
    h3_indices: list[int],
    work_dir: Path,
    timeout: int = 600,
):
    """Run ProteinMPNN on ``pdb_path``; return (log_p[0], command).

    ``score_chain`` is the heavy-chain id in the standardized backbone;
    ``h3_indices`` are 0-based design positions on that chain. Raises if
    ProteinMPNN fails or any H3 position is masked out.
    """
    pdb_path = Path(pdb_path).resolve()
    name = pdb_path.stem
    state_dir = pdb_path.parent
    fixed_positions = [
        position for position in range(1, chain_length(pdb_path, score_chain) + 1)
        if position - 1 not in set(h3_indices)
    ]
    fixed_path = state_dir / "fixed_positions.jsonl"
    fixed_path.write_text(
        json.dumps({name: {score_chain: fixed_positions}}) + "\n", encoding="ascii")
    output = state_dir / "mpnn"
    result_path = output / "conditional_probs_only" / f"{name}.npz"
    command = [
        sys.executable,
        str(Path(mpnn_script).resolve()),
        "--pdb_path", pdb_path.as_posix(),
        "--pdb_path_chains", score_chain,
        "--fixed_positions_jsonl", fixed_path.resolve().as_posix(),
        "--path_to_model_weights", str(Path(weights_dir).resolve()),
        "--model_name", model_name,
        "--conditional_probs_only", "1",
        "--conditional_probs_only_backbone", "1",
        "--num_seq_per_target", "1",
        "--batch_size", "1",
        "--seed", str(seed),
        "--suppress_print", "1",
        "--out_folder", output.resolve().as_posix(),
    ]
    if not result_path.exists():
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout)
        if completed.returncode:
            raise RuntimeError(
                f"ProteinMPNN failed for {name}: {completed.stderr[-2000:]}")
    if not result_path.exists():
        raise FileNotFoundError(f"Missing ProteinMPNN output: {result_path}")
    result = np.load(result_path)
    for index in h3_indices:
        if result["design_mask"][index] <= 0 or result["mask"][index] <= 0:
            raise ValueError(f"Invalid H3 MPNN mask at {name} index {index}")
    return result["log_p"][0], command


def chain_length(pdb_path: Path, chain_id: str) -> int:
    """Residue count of ``chain_id`` in a standardized backbone file."""
    count = 0
    seen = set()
    with open(pdb_path, "r", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("TER"):
                seen.clear()
                continue
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain_id:
                continue
            resseq = line[22:26].strip()
            if resseq not in seen:
                seen.add(resseq)
                count += 1
    if count == 0:
        raise ValueError(f"Chain {chain_id} not found in {pdb_path}")
    return count
