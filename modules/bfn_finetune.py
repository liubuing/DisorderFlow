#!/usr/bin/env python
"""Reward-weighted fine-tuning of BFN model on closed-loop designs.

Converts top-ranked multi-objective designs into a training dataset and
fine-tunes the BFN model using score-weighted NLL loss. High-scoring
designs contribute more to the gradient, steering the model toward
physics-validated sequence space.

Reuses: train.py --finetune + freeze_backbone + sum_weighted_losses
"""

import os, json, math, time
from pathlib import Path
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FineTuneResult:
    checkpoint_path: str
    n_samples: int
    n_epochs: int
    best_loss: float
    final_loss: float
    elapsed_seconds: float


class ScoreWeightedLoss:
    """Apply multi-objective composite scores as per-sample loss weights.

    During BFN fine-tuning, higher-scoring designs get higher weight.
    batch['sample_weight'] must be a float tensor of shape (batch_size,).

    Usage:
        sw_loss = ScoreWeightedLoss(sum_weighted_losses, scale=1.0)
        weighted = sw_loss(loss_dict, batch)
    """

    def __init__(self, base_loss_fn: Callable, scale: float = 1.0):
        self.base_loss_fn = base_loss_fn
        self.scale = scale

    def __call__(self, loss_dict: dict, batch: dict) -> dict:
        weights = batch.get('sample_weight', torch.ones(1))
        if weights.device != next(iter(loss_dict.values())).device:
            weights = weights.to(next(iter(loss_dict.values())).device)

        # Normalize weights to mean=1.0 so overall loss scale doesn't change
        if weights.numel() > 1:
            w_mean = weights.mean()
            if w_mean > 0:
                weights = weights / w_mean
        weights = weights * self.scale

        weighted = {}
        for k, v in loss_dict.items():
            if k == 'overall':
                continue
            if v.dim() > 0 and v.shape[0] == weights.shape[0]:
                weighted[k] = (v * weights).mean()
            else:
                weighted[k] = v.mean() if v.numel() > 1 else v

        weighted['overall'] = sum(weighted.values())
        return weighted


def build_finetune_dataset(
    top_designs: List[Dict],
    output_dir: str,
    target_pdb_path: Optional[str] = None,
    prefix: str = 'cl_finetune',
) -> str:
    """Convert closed-loop top designs to a training-ready PDB directory.

    Each design's AF2-predicted structure (pdb_path) is copied/linked
    into a structured directory for standard BFN dataset loading.

    Also saves a metadata.json with per-sample multi-objective scores
    as training weights.

    Args:
        top_designs: list from ClosedLoopOrchestrator output, each with:
                     sequence, pdb_path, mo_composite_score, dG, iptm, ...
        output_dir: where to write the dataset
        target_pdb_path: optional reference antigen PDB
        prefix: directory name prefix

    Returns:
        Path to the dataset directory
    """
    import shutil

    dataset_dir = os.path.join(output_dir, f'{prefix}_dataset')
    pdb_dir = os.path.join(dataset_dir, 'pdbs')
    os.makedirs(pdb_dir, exist_ok=True)

    metadata = {
        'items': [],
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'prefix': prefix,
        'n_designs': len(top_designs),
    }

    for i, design in enumerate(top_designs):
        seq = design.get('sequence', '')
        pdb_path = design.get('pdb_path', '')
        score = design.get('mo_composite_score', 0.5)
        name = f'{prefix}_{i+1:03d}'

        # Copy or symlink the PDB
        dest_pdb = os.path.join(pdb_dir, f'{name}.pdb')
        src = pdb_path
        if src and os.path.exists(str(src)):
            try:
                shutil.copy2(str(src), dest_pdb)
            except OSError:
                pass  # Skip if copy fails, use placeholder
        else:
            # Write a minimal placeholder — actual training will need real PDBs
            with open(dest_pdb, 'w') as f:
                f.write(f'REMARK Placeholder for {name}\nEND\n')

        metadata['items'].append({
            'name': name,
            'tag': f'CL-{i+1:03d}',
            'sequence': seq,
            'sample_weight': score,
            'mo_composite_score': score,
            'dG': design.get('dG'),
            'iptm': design.get('iptm'),
            'plddt': design.get('plddt'),
        })

    # Write metadata
    with open(os.path.join(dataset_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Write weights file (simple text format for training)
    with open(os.path.join(dataset_dir, 'sample_weights.txt'), 'w') as f:
        for item in metadata['items']:
            f.write(f"{item['name']}\t{item['sample_weight']:.4f}\t{item['sequence']}\n")

    return dataset_dir


def fine_tune_on_designs(
    checkpoint_path: str,
    top_designs: List[Dict],
    output_dir: str,
    model_config_path: Optional[str] = None,
    n_epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-5,
    freeze_backbone: bool = True,
    use_score_weighting: bool = True,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    log_cb: Optional[Callable] = None,
) -> Optional[str]:
    """Fine-tune the BFN model on top-ranked closed-loop designs.

    Uses the existing BFN training infrastructure (train.py's --finetune
    mode) with reward-weighted NLL loss.

    Args:
        checkpoint_path: path to existing BFN checkpoint
        top_designs: list of scored designs from closed-loop ranking
        output_dir: where to save the fine-tuned checkpoint
        model_config_path: path to model config YAML (uses DEFAULT_MODEL_CONFIG if None)
        n_epochs: training epochs
        batch_size: per-device batch size
        lr: learning rate
        freeze_backbone: freeze encoder, only train confidence heads
        use_score_weighting: weight losses by mo_composite_score
        device: 'cuda' or 'cpu'
        log_cb: optional callback for progress messages

    Returns:
        Path to the fine-tuned checkpoint, or None on failure
    """
    def _log(msg):
        if log_cb:
            log_cb(msg)

    _log(f"Building fine-tuning dataset from {len(top_designs)} designs...")

    # Build dataset
    dataset_dir = build_finetune_dataset(top_designs, output_dir)

    _log(f"Loading checkpoint: {checkpoint_path}")

    try:
        from disorderflow.models import get_model
        from disorderflow.utils.misc import load_config as _lc
        from disorderflow.utils.train import recursive_to, sum_weighted_losses, get_optimizer
        from disorderflow.datasets.custom import CustomDataset
    except ImportError as e:
        _log(f"Failed to import BFN modules: {e}")
        return None

    # Load config
    model_config = model_config_path or 'configs/demo_design.yml'
    config, _ = _lc(model_config)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_config = ckpt.get('config', None)

    # Build model
    mc = ckpt_config.model if ckpt_config else config.model
    model = get_model(mc).to(device)

    # Load weights (handle head mismatches)
    ckpt_state = ckpt['model']
    # Strip incompatible heads (same logic as train.py --finetune)
    if any('head_iptm' in k for k in ckpt_state):
        new_iptm_keys = [k for k in model.state_dict() if 'head_iptm' in k]
        old_iptm_keys = [k for k in ckpt_state if 'head_iptm' in k]
        shape_mismatch = False
        for k in new_iptm_keys:
            if k in ckpt_state and ckpt_state[k].shape != model.state_dict()[k].shape:
                shape_mismatch = True
                break
        if shape_mismatch:
            for k in old_iptm_keys:
                ckpt_state.pop(k, None)

    model.load_state_dict(ckpt_state, strict=False)
    model.train()

    # Freeze backbone
    if freeze_backbone:
        head_params = {'head_plddt', 'head_iptm', 'head_pae',
                       'conf_embed', 'iptm_embed', 'pae_embed', 'pair_proj'}
        for name, param in model.named_parameters():
            if not any(hp in name for hp in head_params):
                param.requires_grad = False
        _log(f"Backbone frozen — only confidence heads trainable")

    # Build optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    n_trainable = sum(p.numel() for p in trainable)
    _log(f"Trainable parameters: {n_trainable:,}")

    # Simple training loop (minimal, no dataloader —trains on top designs directly)
    t_start = time.time()
    best_loss = float('inf')
    sw_loss = ScoreWeightedLoss(sum_weighted_losses, scale=1.0) if use_score_weighting else None

    # Since we can't easily load the top_designs PDBs into the full BFN dataset
    # pipeline (which expects preprocessed LMDB), we do a lightweight pass:
    # for each design, run a forward pass on the PDB structure and backprop
    # the score-weighted loss.

    # This is a simplified fine-tuning loop. Production use would require
    # full LMDB dataset construction via the preprocessing pipeline.
    _log("Running score-weighted fine-tuning (simplified mode)...")

    for epoch in range(1, n_epochs + 1):
        epoch_losses = []
        for i, design in enumerate(top_designs):
            pdb_path = design.get('pdb_path', '')
            seq = design.get('sequence', '')
            score = design.get('mo_composite_score', 0.5)

            if not pdb_path or not os.path.exists(str(pdb_path)):
                continue

            try:
                from disorderflow.datasets.protein import preprocess_protein_structure
                from disorderflow.utils.data import PaddingCollate
                from disorderflow.utils.transforms import get_transform

                structure = preprocess_protein_structure(str(pdb_path))
                if structure is None:
                    continue

                # Minimal transforms for training
                transform = get_transform([
                    {'type': 'merge_protein'},
                    {'type': 'patch_protein'},
                ])
                batch = recursive_to(PaddingCollate()([transform(structure)]), device)

                # Add sample weight
                batch['sample_weight'] = torch.tensor([score], device=device)

                # Forward pass
                loss_dict = model(batch)

                # Apply score weighting
                if sw_loss:
                    loss_dict = sw_loss(loss_dict, batch)

                loss = loss_dict.get('overall', sum(loss_dict.values()))
                if isinstance(loss, dict):
                    loss = loss['overall'] if 'overall' in loss else sum(loss.values())

                # Backward
                optimizer.zero_grad()
                if isinstance(loss, torch.Tensor):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()

                epoch_losses.append(loss.item() if isinstance(loss, torch.Tensor) else float(loss))

            except Exception as e:
                _log(f"  Batch {i} error: {e}")
                continue

            if (i + 1) % max(1, len(top_designs) // 2) == 0:
                _log(f"  Epoch {epoch}/{n_epochs} — {i+1}/{len(top_designs)} batches")

        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1) if epoch_losses else float('inf')
        _log(f"  Epoch {epoch}/{n_epochs} avg loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    # Save checkpoint
    os.makedirs(output_dir, exist_ok=True)
    ckpt_name = f'bfn_finetuned_cl_{int(time.time())}.pt'
    ckpt_path = os.path.join(output_dir, ckpt_name)

    torch.save({
        'model': model.state_dict(),
        'config': ckpt_config if ckpt_config else config,
        'finetune_info': {
            'n_designs': len(top_designs),
            'n_epochs': n_epochs,
            'lr': lr,
            'freeze_backbone': freeze_backbone,
            'score_weighted': use_score_weighting,
            'best_loss': best_loss,
        },
    }, ckpt_path)

    elapsed = time.time() - t_start
    _log(f"Fine-tuning complete ({elapsed:.0f}s) — saved to {ckpt_path}")
    _log(f"  Best loss: {best_loss:.4f}")

    return ckpt_path


# ── Convenience function for direct training loop integration ──

def create_finetune_fn(checkpoint_path: str,
                       output_dir: str,
                       n_epochs: int = 3,
                       lr: float = 1e-5,
                       freeze_backbone: bool = True) -> Callable:
    """Create a finetune_fn compatible with ClosedLoopOrchestrator.

    Returns a callable (top_designs, cycle_idx) -> checkpoint_path
    that the orchestrator can invoke between cycles.
    """
    def _finetune(top_designs, cycle_idx):
        cycle_dir = os.path.join(output_dir, f'cycle_{cycle_idx}')
        os.makedirs(cycle_dir, exist_ok=True)
        return fine_tune_on_designs(
            checkpoint_path=checkpoint_path,
            top_designs=top_designs,
            output_dir=cycle_dir,
            n_epochs=n_epochs,
            lr=lr,
            freeze_backbone=freeze_backbone,
        )
    return _finetune
