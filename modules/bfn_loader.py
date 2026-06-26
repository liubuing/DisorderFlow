#!/usr/bin/env python
"""Minimal BFN model loader —no Gradio/UI dependencies.

Loads the BFN model from checkpoint with proper config handling.
Supports XPU and CUDA devices.
"""

import os
import torch
from pathlib import Path


def _detect_device():
    """Auto-detect available device: CUDA > XPU > CPU."""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        return 'xpu'
    return 'cpu'


PROJECT_DIR = Path(__file__).parent.parent
DEFAULT_MODEL_CONFIG = PROJECT_DIR / 'configs' / 'demo_design.yml'
APP_CONFIG_FILE = PROJECT_DIR / 'app_config.yaml'

_bfn_model = None
_bfn_config = None


def _get_checkpoint_path():
    """Read checkpoint path from app_config.yaml."""
    import yaml
    with open(APP_CONFIG_FILE, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg['models']['bfn']['checkpoint']


def load_bfn(device=None):
    """Load BFN model with config from checkpoint.

    Args:
        device: torch device string (default: auto-detect CUDA > XPU > CPU)

    Returns:
        (model, config) tuple. Model is in eval mode.
    """
    if device is None:
        device = _detect_device()
    global _bfn_model, _bfn_config

    if _bfn_model is not None:
        return _bfn_model, _bfn_config

    from disorderflow.models import get_model
    from disorderflow.utils.misc import load_config as _lc

    ckpt_path = _get_checkpoint_path()
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    config, _ = _lc(DEFAULT_MODEL_CONFIG)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    mc = ckpt['config'].model
    if hasattr(ckpt['config'], 'train') and hasattr(ckpt['config'].train, 'loss_weights'):
        mc['loss_weight'] = dict(ckpt['config'].train.loss_weights)

    model = get_model(mc).to(device)

    ckpt_state = ckpt['model']
    # Skip ipTM head keys if architecture mismatches
    if any('head_iptm' in k for k in ckpt_state):
        new_iptm_keys = [k for k in model.state_dict() if 'head_iptm' in k]
        shape_mismatch = False
        for k in new_iptm_keys:
            if k in ckpt_state and ckpt_state[k].shape != model.state_dict()[k].shape:
                shape_mismatch = True
                break
        if shape_mismatch:
            for k in list(ckpt_state.keys()):
                if 'head_iptm' in k:
                    ckpt_state.pop(k)

    model.load_state_dict(ckpt_state, strict=False)
    model.eval()

    _bfn_model = model
    _bfn_config = config
    return model, config


def has_disorder_head(model=None):
    """Check if the loaded model has a disorder prediction head."""
    if model is None:
        model, _ = load_bfn()
    if hasattr(model, 'receiver'):
        return getattr(model.receiver, 'disorder_head', False)
    if hasattr(model, 'bfn') and hasattr(model.bfn, 'receiver'):
        return getattr(model.bfn.receiver, 'disorder_head', False)
    return False


def run_bfn_design(pdb_path, region_spec, num_samples=10, stochastic=True,
                   context_chains=None, device=None, sort_by=None, descending=True,
                   disorder_guided=False, disorder_guided_strength=1.0):
    """Run BFN protein sequence design on specified regions.

    Args:
        pdb_path: path to input PDB file
        region_spec: e.g. "B:26-33,51-58,97-113"
        num_samples: number of design samples
        stochastic: use stochastic sampling (vs deterministic)
        context_chains: list of chain IDs to include as visible context
            (e.g. ['A'] for antigen). These chains are NOT designed.
            None or [] = FixBB mode (no context).
        device: torch device
        disorder_guided: if True, scale the CDR sampling noise per-residue by the
            disorder head's prediction (residues it reads as disordered get more
            entropy → pliable CDRs; ordered framework-facing residues get less).
            Requires a checkpoint with a trained disorder head. See task-A.
        disorder_guided_strength: 0..1+ pliability scaling strength.

    Returns:
        list of dicts, each with: sequence, ppl, entropy, plddt, iptm, pae,
        and optionally disorder_score if model has disorder head.
    """
    import re
    from disorderflow.datasets.protein import preprocess_protein_structure
    from disorderflow.utils.train import recursive_to
    from disorderflow.utils.misc import seed_all
    from disorderflow.utils.data import PaddingCollate
    from disorderflow.utils.transforms import get_transform

    if device is None:
        device = _detect_device()

    AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'

    model, config = load_bfn(device)
    seed_all(getattr(config.sampling, 'seed', 42))

    # Parse region spec
    regions = {}
    for cid, spec in re.findall(r'([A-Za-z0-9]+):([0-9,\-\s]+)', region_spec):
        indices = []
        for seg in spec.split(','):
            seg = seg.strip()
            if not seg:
                continue
            if '-' in seg:
                a, b = seg.split('-')
                indices.extend(range(int(a.strip()), int(b.strip()) + 1))
            else:
                indices.append(int(seg))
        regions[cid] = sorted(set(indices))

    if not regions:
        raise ValueError(f"Invalid region spec: {region_spec}")

    design_chains = list(regions.keys())

    # Merge context chains with design chains for structure loading
    context_chains = context_chains or []
    all_chains = sorted(set(design_chains + context_chains),
                        key=lambda c: design_chains.index(c) if c in design_chains else 999)

    # Preprocess —load all chains (design + context)
    structure = preprocess_protein_structure(pdb_path, chain_ids=all_chains)
    if structure is None:
        raise ValueError(f"Cannot parse structure: {pdb_path}")

    transform = get_transform([
        {'type': 'mask_region', 'regions': regions},
        {'type': 'merge_protein'},
        {'type': 'patch_protein'},
    ])
    batch = recursive_to(PaddingCollate()([transform(structure)]), device)
    gen_mask = batch['generate_flag'][0].bool()
    if gen_mask.sum() == 0:
        raise ValueError("No residues selected for design")

    sample_opt = {'deterministic': not stochastic, 'num_recycles': 3}
    if disorder_guided:
        sample_opt['disorder_guided'] = True
        sample_opt['disorder_guided_strength'] = disorder_guided_strength
    results_list = []

    _has_disorder = has_disorder_head(model)

    for i in range(num_samples):
        with torch.no_grad():
            traj = model.sample(batch, sample_opt=sample_opt)

        pred_aa = traj[0][2][0][gen_mask]
        seq = ''.join(AA_LETTERS[a] if a < 20 else 'X' for a in pred_aa.cpu())

        logits = traj['pred_logits'][0][gen_mask]
        lp = torch.log_softmax(logits[..., :20], dim=-1)
        probs = torch.exp(lp)
        nll = -lp[range(len(pred_aa)), pred_aa].mean()
        ppl = torch.exp(nll).item()

        entropy = -(probs * lp).sum(dim=-1).mean().item()
        max_prob = probs.max(dim=-1).values.mean().item()
        plddt_val = traj['plddt'][0][gen_mask].mean().item()
        plddt_per_res = traj['plddt'][0][gen_mask]
        plddt_std = plddt_per_res.std().item()
        iptm_val = traj['iptm'][0].item()
        pae_val = traj['pae'][0][gen_mask][:, gen_mask].mean().item()

        result = {
            'sequence': seq, 'ppl': ppl, 'entropy': entropy,
            'max_prob': max_prob,
            'plddt': plddt_val, 'plddt_std': plddt_std,
            'iptm': iptm_val, 'pae': pae_val,
        }

        if _has_disorder:
            disorder = traj.get('disorder')
            if disorder is not None:
                d_val = disorder[0][gen_mask].mean().item()
                result['disorder_score'] = d_val

        results_list.append(result)

    # Optional ranking. Without this (the historical default), callers like
    # run_oc_v14_p1.py took designs[:top_n] = the first N *random* samples, so
    # the AF2 yield test was effectively blind sampling — a root cause of the
    # low OC yield (designs all had AF2 ipTM ~0.05). With a calibrated
    # confidence head, sort_by='iptm' picks the highest-self-confidence designs
    # to send to AF2, turning the pipeline into confidence-ranked selection.
    if sort_by is not None and results_list:
        reverse = bool(descending)
        results_list.sort(key=lambda r: r.get(sort_by, float('-inf')), reverse=reverse)

    return results_list
