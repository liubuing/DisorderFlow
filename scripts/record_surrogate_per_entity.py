"""Record per-entity surrogate predictions and measured inference timing.

Evaluation-only: reuses the frozen deployment-contract checkpoints and the
already-frozen transfer / holdout LMDBs. Gate decisions live elsewhere; this
script only materializes per-entity predictions for the generator-baseline
comparison and measures per-record wall time for the cost table.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_candidate_interface_checkpoint import evaluate_checkpoint  # noqa: E402
from disorderflow.datasets.confidence_dataset import ConfidenceRegressionDataset  # noqa: E402
from disorderflow.utils.data import CompleteGroupBatchSampler, PaddingCollate  # noqa: E402

CHECKPOINTS = [
    ("2041", "logs/bfn_candidate_interface_multiscaffold_dual_sem_v2_2026_09_01__12_13_52_dual_sem_v2_s2041/checkpoints/1000.pt"),
    ("2053", "logs/bfn_candidate_interface_multiscaffold_dual_sem_v2_2026_09_01__12_29_06_dual_sem_v2_s2053/checkpoints/800.pt"),
    ("2069", "logs/bfn_candidate_interface_multiscaffold_dual_sem_v2_2026_09_01__21_10_10_dual_sem_v2_s2069_retry/checkpoints/800.pt"),
]
COHORTS = {
    "transfer": "data/confidence_candidate_interface_multiscaffold_calibration_ext_dual_sem_v1/calibration.lmdb",
    "holdout": "data/confidence_candidate_interface_multiscaffold_v1_holdout_dual_sem_v2/test.lmdb",
}
OUT = ROOT / "results" / "candidate_interface_per_entity_records"
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for cohort, dataset_rel in COHORTS.items():
        dataset = ConfidenceRegressionDataset({
            "db_path": str(ROOT / dataset_rel),
            "candidate_interface_v1": True,
            "max_residues": 0,
        })
        sampler = CompleteGroupBatchSampler(dataset, 20, shuffle=False)
        loader = DataLoader(
            dataset, batch_sampler=sampler, collate_fn=PaddingCollate(), num_workers=0)
        for seed, ckpt_rel in CHECKPOINTS:
            t0 = time.time()
            evaluation = evaluate_checkpoint(ROOT / ckpt_rel, loader, device)
            elapsed = time.time() - t0
            records = evaluation["entity_aggregated_records"]
            out_path = OUT / f"{cohort}_s{seed}_entity_records.json"
            out_path.write_text(json.dumps({
                "schema_version": "candidate_interface_per_entity_records_v1",
                "cohort": cohort,
                "seed": seed,
                "checkpoint": ckpt_rel,
                "checkpoint_sha256": evaluation["checkpoint_sha256"],
                "device": device,
                "inference_wall_seconds": elapsed,
                "n_entities": len(records),
                "seconds_per_entity": elapsed / len(records),
                "records": [
                    {k: r[k] for k in (
                        "construct_id", "scaffold_family", "pred_pae",
                        "target_pae", "pred_plddt", "target_plddt",
                        "pred_iptm", "target_iptm")}
                    for r in records
                ],
            }, indent=2) + "\n", encoding="ascii")
            print(f"{cohort} s{seed}: {len(records)} entities, "
                  f"{elapsed:.1f}s wall, {elapsed/len(records):.3f}s/entity, device={device}")


if __name__ == "__main__":
    main()
