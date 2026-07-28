#!/usr/bin/env python3
"""Evaluate contact prediction and CDR recovery on audited SAbDab2 complexes."""

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from disorderflow.datasets import get_dataset  # noqa: E402
from disorderflow.utils.data import PaddingCollate  # noqa: E402
from disorderflow.utils.misc import seed_all  # noqa: E402
from disorderflow.utils.train import recursive_to  # noqa: E402
from evaluate_v5_1_phase3 import binary_metrics, contact_targets, load_checkpoint  # noqa: E402


def evaluate_sample(model, source_batch, device, seed):
    batch = recursive_to(copy.deepcopy(source_batch), device)
    batch["fixed_t"] = 0.5
    captured = {}

    def hook(_module, _inputs, output):
        captured["receiver"] = output

    handle = model.bfn.receiver.register_forward_hook(hook)
    try:
        seed_all(seed)
        with torch.inference_mode():
            model(batch)
    finally:
        handle.remove()

    output = captured["receiver"]
    logits = output[0][..., :20].float()
    pred_contact = output[8].float()
    cdr = batch["generate_flag"].bool() & batch["mask"].bool()
    if not cdr.any():
        raise RuntimeError("External sample contains no generated CDR residues")
    targets = batch["aa"][cdr].long()
    cdr_logits = logits[cdr]
    labels = contact_targets(batch)
    native_nll = F.cross_entropy(cdr_logits, targets).item()
    return {
        "native_nll": native_nll,
        "native_ppl": math.exp(min(native_nll, 20.0)),
        "recovery": (cdr_logits.argmax(dim=-1) == targets).float().mean().item(),
        "n_cdr": int(cdr.sum().item()),
        "contact_scores": torch.sigmoid(pred_contact[cdr]).cpu().numpy(),
        "contact_labels": labels[cdr].cpu().numpy(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lmdb", default="data/sabdab2_abag_external/test.lmdb")
    parser.add_argument("--audit", default="data/sabdab2_abag_external/audit.json")
    parser.add_argument("--output", default="results/v5_1_corrective/sabdab2_external.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2081)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    model, config = load_checkpoint(args.checkpoint, args.device)
    dataset_config = copy.deepcopy(config.dataset.val)
    dataset_config.lmdb_path = args.lmdb
    dataset = get_dataset(dataset_config)
    if args.max_samples:
        dataset.ids = dataset.ids[:args.max_samples]
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=PaddingCollate())

    records = []
    contact_scores = []
    contact_labels = []
    failures = []
    for index, batch in enumerate(loader):
        try:
            result = evaluate_sample(model, batch, args.device, args.seed + index)
        except Exception as error:  # noqa: BLE001
            failures.append({"index": index, "error": str(error)})
            continue
        records.append({key: value for key, value in result.items() if not key.startswith("contact_")})
        contact_scores.extend(result["contact_scores"].tolist())
        contact_labels.extend(result["contact_labels"].astype(int).tolist())
        if (index + 1) % 25 == 0:
            print(f"Evaluated {index + 1}/{len(dataset)} complexes", flush=True)

    if not records:
        raise RuntimeError("No external SAbDab2 samples could be evaluated")
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    report = {
        "checkpoint": args.checkpoint,
        "weights_kind": "ema",
        "fixed_t": 0.5,
        "n_input_complexes": len(dataset),
        "n_evaluated_complexes": len(records),
        "n_failed_complexes": len(failures),
        "n_cdr_residues": int(sum(record["n_cdr"] for record in records)),
        "independence": audit["interface_eligibility"],
        "scope": {
            "evaluated": ["native_cdr_recovery", "contact_classification"],
            "not_evaluated": ["disorder_conditioning", "contrastive_compatibility"],
            "reason": "No independent per-residue disorder profile is available for these antigens",
        },
        "native_cdr": {
            "mean_nll": float(np.mean([record["native_nll"] for record in records])),
            "mean_ppl": float(np.mean([record["native_ppl"] for record in records])),
            "mean_recovery": float(np.mean([record["recovery"] for record in records])),
        },
        "contact": binary_metrics(contact_scores, contact_labels),
        "failures": failures,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "failures"}, indent=2))


if __name__ == "__main__":
    main()
