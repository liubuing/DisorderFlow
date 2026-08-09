#!/usr/bin/env python
"""Run matched 4HIX generation with current and parent BFN checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PDB_PATH = ROOT / "data/anti_abeta_refs/4HIX.pdb"
CURRENT_CHECKPOINT = (
    ROOT / "logs/bfn_disorder_v4_afdb_supervised_2026_08_05__17_22_45_"
    "disorder_v4_balanced_long_s2032/checkpoints/best.pt"
)
PARENT_CHECKPOINT = ROOT / "data/pretrained_candidates/AntibodyDesignBFN_best.pt"
CALIBRATION = ROOT / "calibration_artifacts/disorder_v4_balanced_s2032.json"
REGION_SPEC = "H:96-107"
NATIVE_H3 = "VRYDHYSGSSDY"
CONTEXT_CHAINS = ["L", "A"]
DEFAULT_SEEDS = [4101, 4111, 4121]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_recovery(sequence):
    if len(sequence) != len(NATIVE_H3):
        raise ValueError(f"H3 length {len(sequence)} != {len(NATIVE_H3)}")
    return sum(a == b for a, b in zip(sequence, NATIVE_H3, strict=True)) / len(NATIVE_H3)


def reset_loader(checkpoint):
    import modules.bfn_loader as loader

    os.environ["DISORDERFLOW_CHECKPOINT"] = str(checkpoint)
    loader._bfn_model = None
    loader._bfn_config = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return loader


def current_disorder_profile(loader, device):
    from modules.idp_disorder_analysis import predict_disorder

    model, config = loader.load_bfn(device)
    result = predict_disorder(
        model,
        config,
        str(PDB_PATH),
        chain_id="A",
        device=device,
        calibration=str(CALIBRATION),
    )
    if result["sequence"] != "DAEFRH":
        raise ValueError(f"Unexpected 4HIX peptide sequence: {result['sequence']}")
    return result["disorder_scores"].astype(float).tolist(), result.get("calibration")


def run_arm(name, checkpoint, seeds, samples, device, profile=None):
    loader = reset_loader(checkpoint)
    rows = []
    started = time.perf_counter()
    for seed in seeds:
        seed_started = time.perf_counter()
        designs = loader.run_bfn_design(
            str(PDB_PATH),
            REGION_SPEC,
            num_samples=samples,
            stochastic=True,
            context_chains=CONTEXT_CHAINS,
            device=device,
            antigen_chains=["A"],
            epitope_disorder_profile=profile,
            disorder_guided=profile is not None,
            sampling_seed=seed,
        )
        elapsed = time.perf_counter() - seed_started
        for index, design in enumerate(designs):
            sequence = design["sequence"]
            rows.append(
                {
                    "arm": name,
                    "seed": seed,
                    "sample_index": index,
                    "sequence": sequence,
                    "native_recovery": sequence_recovery(sequence),
                    "seed_wall_seconds": elapsed,
                    **design,
                }
            )
    return {
        "name": name,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        ),
        "results": rows,
    }


def summarize(arm):
    rows = arm["results"]
    return {
        "n": len(rows),
        "n_unique": len({row["sequence"] for row in rows}),
        "mean_native_recovery": float(np.mean([row["native_recovery"] for row in rows])),
        "mean_ppl": float(np.mean([row["ppl"] for row in rows])),
        "wall_seconds": arm["wall_seconds"],
        "peak_cuda_memory_bytes": arm["peak_cuda_memory_bytes"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=str(ROOT / "results/ablation/4hix_bfn_matched_v1.json"))
    args = parser.parse_args()

    for path in (PDB_PATH, CURRENT_CHECKPOINT, PARENT_CHECKPOINT, CALIBRATION):
        if not path.exists():
            raise FileNotFoundError(path)

    loader = reset_loader(CURRENT_CHECKPOINT)
    profile, calibration = current_disorder_profile(loader, args.device)
    arms = [
        run_arm(
            "current_disorder_on",
            CURRENT_CHECKPOINT,
            args.seeds,
            args.samples,
            args.device,
            profile=profile,
        ),
        run_arm("current_disorder_off", CURRENT_CHECKPOINT, args.seeds, args.samples, args.device),
        run_arm("parent_bfn", PARENT_CHECKPOINT, args.seeds, args.samples, args.device),
    ]
    output = {
        "schema_version": 1,
        "status": "matched_generation_diagnostic_not_design_validation",
        "pdb": str(PDB_PATH.relative_to(ROOT)),
        "pdb_sha256": sha256(PDB_PATH),
        "region_spec": REGION_SPEC,
        "native_h3": NATIVE_H3,
        "context_chains": CONTEXT_CHAINS,
        "seeds": args.seeds,
        "samples_per_seed": args.samples,
        "disorder_profile": profile,
        "calibration": calibration,
        "arms": arms,
        "summary": {arm["name"]: summarize(arm) for arm in arms},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps(output["summary"], indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
