#!/usr/bin/env python3
"""Run one real grouped-contrastive BFN forward/backward on CIF coordinates."""

import argparse
import json
import sys
from pathlib import Path

import torch
from easydict import EasyDict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from disorderflow.datasets.statecontrast_structural import StateContrastStructuralDataset
from disorderflow.models import get_model
from disorderflow.utils.data import CompleteGroupBatchSampler, PaddingCollate
from disorderflow.utils.transforms import get_transform


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=(
            "logs/v5_1_corrective/train/"
            "bfn_v5_1_stage5_corrective_2026_07_21__07_19_10_corrective/"
            "checkpoints/best.pt"
        ),
    )
    parser.add_argument("--records-dir", default="data/statecontrast_factory_v2_1")
    parser.add_argument(
        "--parent-lmdb", default="data/statecontrast_structural_smoke_v2/parents.lmdb")
    parser.add_argument(
        "--output", default="data/statecontrast_structural_smoke_v2/bfn_smoke.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--full-group", action="store_true")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "none"), default="bf16")
    args = parser.parse_args()

    torch.manual_seed(20260721)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    transform = get_transform([
        {
            "type": "mask_multiple_cdrs",
            "selection": ["H1", "H2", "H3", "L1", "L2", "L3"],
            "augmentation": False,
        },
        {"type": "merge_chains"},
        {"type": "patch_around_anchor"},
    ])
    dataset_config = EasyDict({
        "records_dir": str(ROOT / args.records_dir),
        "parent_lmdb_path": str(ROOT / args.parent_lmdb),
        "split": "train",
        "expected_schema": "statecontrast_factory_v2_1",
    })
    if not args.full_group:
        dataset_config["include_operations"] = [
            "observed_bound", "cdr_single_mutation:H3"]
    dataset = StateContrastStructuralDataset(dataset_config, transform=transform)
    batch_capacity = 14 if args.full_group else 2
    sampler = CompleteGroupBatchSampler(
        dataset, max_batch_records=batch_capacity, shuffle=False)
    indices = next(iter(sampler))
    batch = PaddingCollate()([dataset[index] for index in indices])
    batch = {key: to_device(value, device) for key, value in batch.items()}

    checkpoint = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    model = get_model(checkpoint["config"].model)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.bfn.loss_weight = {
        "grouped_contrastive": 1.0,
        "grouped_contrastive_margin": 0.5,
        "train_recycles": 1,
        "mask_aug_prob": 0.0,
        "t_clamp": 0.5,
    }
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (
            "contrastive_head" in name or "contrastive_cdr_conv" in name)
    model.to(device).train()
    model.zero_grad(set_to_none=True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    before_step = [parameter.detach().clone() for parameter in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)

    if batch["contrastive_group_id"].unique().numel() != 1:
        raise RuntimeError("Smoke batch is not one complete evidence group")
    ranks = batch["contrastive_rank"].tolist()
    if ranks[0] != 1.0 or any(rank != 0.0 for rank in ranks[1:]):
        raise RuntimeError("Smoke batch does not contain one observed parent followed by counterfactuals")
    antigen_mask = model.bfn._mask_antigen(batch, batch["generate_flag"])
    if not antigen_mask.any(dim=1).all() or not batch["generate_flag"].any(dim=1).all():
        raise RuntimeError("Smoke batch lacks generated CDR or antigen context")

    autocast_enabled = device.type == "cuda" and args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
        losses = model(batch)
        grouped_loss = losses["grouped_contrastive"]
    if not torch.isfinite(grouped_loss) or grouped_loss.item() <= 0:
        raise RuntimeError("Grouped contrastive loss is non-finite or connected-zero")
    grouped_loss.backward()

    gradients = {
        name: float(parameter.grad.float().abs().sum().item())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    if not gradients or not all(torch.isfinite(torch.tensor(value)) for value in gradients.values()):
        raise RuntimeError("Contrastive head gradients are absent or non-finite")
    if sum(gradients.values()) <= 0:
        raise RuntimeError("Contrastive head received zero total gradient")
    optimizer.step()
    parameter_delta = sum(
        float((parameter.detach() - before).float().abs().sum().item())
        for parameter, before in zip(trainable, before_step, strict=True)
    )
    if not torch.isfinite(torch.tensor(parameter_delta)) or parameter_delta <= 0:
        raise RuntimeError("Optimizer step did not update contrastive parameters")

    report = {
        "status": "pass",
        "device": str(device),
        "checkpoint_iteration": checkpoint.get("iteration"),
        "weights_kind": checkpoint.get("weights_kind"),
        "records": len(dataset),
        "groups": len(dataset.group_indices),
        "full_group": args.full_group,
        "amp_dtype": args.amp_dtype,
        "batch_indices": indices,
        "batch_shape": list(batch["aa"].shape),
        "generated_residues": batch["generate_flag"].sum(dim=1).tolist(),
        "antigen_residues": antigen_mask.sum(dim=1).tolist(),
        "grouped_contrastive_loss": float(grouped_loss.detach().float().item()),
        "gradient_tensors": len(gradients),
        "gradient_l1_total": sum(gradients.values()),
        "optimizer": "AdamW",
        "optimizer_lr": 1e-4,
        "parameter_delta_l1": parameter_delta,
        "cuda_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None),
    }
    output = ROOT / args.output
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
