#!/usr/bin/env python
"""Verify the frozen v2 holdout and controlled two-stage training lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-a", default="configs/train/bfn_multiscaffold_v2_stage_a_random.yml")
    parser.add_argument(
        "--stage-b", default="configs/train/bfn_multiscaffold_v2_stage_b_disorder.yml")
    parser.add_argument(
        "--holdout", default="data/multiscaffold_confirmatory_v2/holdout_manifest.json")
    parser.add_argument(
        "--feasibility", default="data/multiscaffold_confirmatory_v2/feasibility_audit.json")
    parser.add_argument(
        "--lookup", default="data/disorder_supervision/train_afdb_multiscaffold_v2.pkl")
    parser.add_argument(
        "--lookup-audit", default="data/multiscaffold_confirmatory_v2/disorder_lookup_audit.json")
    parser.add_argument(
        "--out", default="results/ablation/multiscaffold_v2_training_contract.json")
    args = parser.parse_args()

    stage_a_path = ROOT / args.stage_a
    stage_b_path = ROOT / args.stage_b
    holdout_path = ROOT / args.holdout
    feasibility_path = ROOT / args.feasibility
    lookup_path = ROOT / args.lookup
    lookup_audit_path = ROOT / args.lookup_audit
    stage_a = yaml.safe_load(stage_a_path.read_text(encoding="utf-8"))
    stage_b = yaml.safe_load(stage_b_path.read_text(encoding="utf-8"))
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
    lookup = pickle.loads(lookup_path.read_bytes())
    lookup_audit = json.loads(lookup_audit_path.read_text(encoding="utf-8"))

    assert stage_a["lineage"]["initialization"] == "random"
    assert stage_a["train"]["freeze_backbone"] is False
    assert stage_b["lineage"]["initialization"] == "stage_a_only"
    assert stage_b["dataset"]["disorder_lookup_train"] == args.lookup
    assert holdout["n_all_axis_components"] >= 12
    assert holdout["representative_ids_sha256"] == hashlib.sha256(
        "\n".join(sorted(holdout["representative_ids"])).encode("ascii")).hexdigest()
    proposed = feasibility["schemes"][feasibility["proposed_scheme"]]
    assert set(holdout["representative_ids"]) <= set(proposed["independent_ids"])
    assert not set(lookup_audit["excluded_ids"]) & set(lookup["profiles"])
    assert set(lookup["profiles"]) == set(lookup["provenance"])

    train_ids = pickle.loads(Path(
        f"{ROOT / stage_a['dataset']['train']['lmdb_path']}-ids").read_bytes())
    val_ids = pickle.loads(Path(
        f"{ROOT / stage_a['dataset']['val']['lmdb_path']}-ids").read_bytes())
    assert not set(holdout["representative_ids"]) & set(train_ids)
    assert not set(holdout["representative_ids"]) & set(val_ids)

    guard_code = (
        "import os,runpy,sys,torch;"
        "original_torch_load=torch.load;"
        "torch.load=lambda f,*a,**k:"
        "((_ for _ in ()).throw(RuntimeError('PRETRAINED_LOAD_FORBIDDEN')) "
        "if isinstance(f,(str,bytes,os.PathLike)) else original_torch_load(f,*a,**k));"
        f"sys.argv=['train.py',{str(stage_a_path)!r},'--debug','--device','cpu','--no_amp','--max-iters','1','--val-freq','100'];"
        f"runpy.run_path({str(ROOT / 'train.py')!r},run_name='__main__')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", guard_code], cwd=ROOT,
        capture_output=True, text=True, timeout=300)
    combined_log = completed.stdout + completed.stderr
    if completed.returncode:
        raise RuntimeError(combined_log[-4000:])
    assert "resume=None, finetune=None, init=None" in combined_log
    assert "Train 2952 | Val 328" in combined_log
    assert "Number of parameters: 11388474" in combined_log

    output = {
        "schema_version": 1,
        "status": "two_stage_training_contract_verified",
        "stage_a": {
            "config": args.stage_a,
            "config_sha256": sha256(stage_a_path),
            "initialization": "random",
            "checkpoint_flags": {"resume": None, "finetune": None, "init": None},
            "freeze_backbone": False,
            "train_records": len(train_ids),
            "val_records": len(val_ids),
            "model_parameters": 11388474,
            "torch_load_guard_smoke_passed": True,
            "one_step_forward_backward_smoke_passed": True,
        },
        "stage_b": {
            "config": args.stage_b,
            "config_sha256": sha256(stage_b_path),
            "only_permitted_initializer": "stage_a_checkpoint",
            "filtered_lookup_sha256": sha256(lookup_path),
            "lookup_records": len(lookup["profiles"]),
            "holdout_homologs_removed": lookup_audit["n_excluded_records"],
        },
        "holdout": {
            "manifest_sha256": sha256(holdout_path),
            "components": holdout["n_all_axis_components"],
            "representatives": holdout["n_representatives"],
            "representative_ids_sha256": holdout["representative_ids_sha256"],
            "exact_overlap_with_stage_a_train_or_val": 0,
        },
        "claim_boundary": "initialization, data-isolation, and one-step optimization smoke only; no full training or benchmark inference performed",
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
