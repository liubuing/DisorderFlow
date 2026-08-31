#!/usr/bin/env python
"""Minimal BFN model loader —no Gradio/UI dependencies.

Loads the BFN model from checkpoint with proper config handling.
Supports XPU and CUDA devices.
"""

import os
import re
from pathlib import Path

import torch


def _detect_device():
    """Auto-detect available device: CUDA > XPU > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


PROJECT_DIR = Path(__file__).parent.parent
DEFAULT_MODEL_CONFIG = PROJECT_DIR / "configs" / "demo_design.yml"
APP_CONFIG_FILE = PROJECT_DIR / "app_config.yaml"

_bfn_model = None
_bfn_config = None


def _validate_candidate_interface_checkpoint(model, checkpoint_state):
    receiver = model.bfn.receiver
    if receiver.confidence_head_kind != "candidate_interface_v1":
        return
    required = [
        key for key in model.state_dict()
        if ".candidate_interface_" in key
    ]
    missing = [key for key in required if key not in checkpoint_state]
    if missing:
        raise RuntimeError(
            "candidate_interface_v1 checkpoint is missing confidence weights: "
            + ", ".join(missing[:5])
        )


def _get_checkpoint_path():
    """Read checkpoint path from app_config.yaml."""
    override = os.environ.get("DISORDERFLOW_CHECKPOINT")
    if override:
        return str(Path(override).expanduser().resolve())
    import yaml

    with open(APP_CONFIG_FILE, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    path = Path(cfg["models"]["bfn"]["checkpoint"]).expanduser()
    return str(path if path.is_absolute() else PROJECT_DIR / path)


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

    mc = ckpt["config"].model
    if hasattr(ckpt["config"], "train") and hasattr(ckpt["config"].train, "loss_weights"):
        mc["loss_weight"] = dict(ckpt["config"].train.loss_weights)

    model = get_model(mc).to(device)

    ckpt_state = ckpt["model"]
    _validate_candidate_interface_checkpoint(model, ckpt_state)
    # Skip ipTM head keys if architecture mismatches
    if any("head_iptm" in k for k in ckpt_state):
        new_iptm_keys = [k for k in model.state_dict() if "head_iptm" in k]
        shape_mismatch = False
        for k in new_iptm_keys:
            if k in ckpt_state and ckpt_state[k].shape != model.state_dict()[k].shape:
                shape_mismatch = True
                break
        if shape_mismatch:
            for k in list(ckpt_state.keys()):
                if "head_iptm" in k:
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
    if hasattr(model, "receiver"):
        return getattr(model.receiver, "disorder_head", False)
    if hasattr(model, "bfn") and hasattr(model.bfn, "receiver"):
        return getattr(model.bfn.receiver, "disorder_head", False)
    return False


def parse_region_spec(region_spec):
    """Convert 1-based user residue positions to 0-based MaskRegion indices."""
    regions = {}
    for cid, spec in re.findall(r"([A-Za-z0-9]+):([0-9,\-\s]+)", region_spec):
        indices = []
        for segment in spec.split(","):
            segment = segment.strip()
            if not segment:
                continue
            if "-" in segment:
                start_text, end_text = segment.split("-", 1)
                start, end = int(start_text.strip()), int(end_text.strip())
                if start < 1 or end < start:
                    raise ValueError(f"Invalid 1-based residue range: {segment}")
                indices.extend(range(start - 1, end))
            else:
                position = int(segment)
                if position < 1:
                    raise ValueError(f"Invalid 1-based residue position: {position}")
                indices.append(position - 1)
        if indices:
            regions[cid] = sorted(set(indices))
    if not regions:
        raise ValueError(f"Invalid region spec: {region_spec}")
    return regions


def build_region_batch(
    pdb_path, region_spec, context_chains=None, device=None, antigen_chains=None
):
    """Build the masked model batch shared by generation and fixed scoring."""
    from disorderflow.datasets.protein import preprocess_protein_structure
    from disorderflow.utils.data import PaddingCollate
    from disorderflow.utils.train import recursive_to
    from disorderflow.utils.transforms import get_transform

    if device is None:
        device = _detect_device()
    regions = parse_region_spec(region_spec)
    design_chains = list(regions)

    if context_chains is None:
        from Bio import PDB

        parsed = PDB.PDBParser(QUIET=True).get_structure("context", pdb_path)[0]
        context_chains = [
            chain.id for chain in parsed.get_chains() if chain.id not in design_chains
        ]
    else:
        context_chains = list(context_chains)
    overlap = set(context_chains) & set(design_chains)
    if overlap:
        raise ValueError(f"Context chains cannot also be design chains: {sorted(overlap)}")
    all_chains = sorted(
        set(design_chains + context_chains),
        key=lambda chain: design_chains.index(chain) if chain in design_chains else 999,
    )

    structure = preprocess_protein_structure(pdb_path, chain_ids=all_chains)
    if structure is None:
        raise ValueError(f"Cannot parse structure: {pdb_path}")
    transform = get_transform(
        [
            {"type": "mask_region", "regions": regions},
            {"type": "merge_protein"},
            {"type": "patch_protein"},
        ]
    )
    batch = recursive_to(PaddingCollate()([transform(structure)]), device)
    if antigen_chains is not None:
        from disorderflow.utils.protein.constants import Fragment

        antigen_chains = set(antigen_chains)
        unknown = antigen_chains - set(context_chains)
        if unknown:
            raise ValueError(
                f"Antigen chains must be included in context_chains: {sorted(unknown)}"
            )
        fragment_type = torch.full_like(batch["fragment_type"], fill_value=int(Fragment.Light))
        for index, label in enumerate(batch["chain_id"]):
            chain_id = label[0] if isinstance(label, (tuple, list)) else label
            if chain_id in design_chains:
                fragment_type[0, index] = int(Fragment.Heavy)
            elif chain_id in antigen_chains:
                fragment_type[0, index] = int(Fragment.Antigen)
        batch["fragment_type"] = fragment_type
    if not batch["generate_flag"].any():
        raise ValueError("No residues selected for design")
    return batch


def inject_candidate_sequence(batch, candidate_sequence):
    """Inject an exact candidate sequence into the batch's generated positions."""
    aa_letters = "ACDEFGHIKLMNPQRSTVWY"
    candidate = candidate_sequence.strip().upper()
    gen_mask = batch["generate_flag"].bool()
    expected_length = int(gen_mask.sum().item())
    if len(candidate) != expected_length:
        raise ValueError(
            f"Candidate sequence has length {len(candidate)}; "
            f"design region requires {expected_length}"
        )
    invalid = sorted(set(candidate) - set(aa_letters))
    if invalid:
        raise ValueError(f"Candidate sequence contains invalid residues: {''.join(invalid)}")

    batch["aa"] = batch["aa"].clone()
    encoded = torch.tensor(
        [aa_letters.index(residue) for residue in candidate],
        dtype=batch["aa"].dtype,
        device=batch["aa"].device,
    )
    batch["aa"][gen_mask] = encoded
    return batch


def _score_exact_candidate_confidence(model, batch, candidate_sequence, fixed_t=0.5):
    """Post-score an emitted sequence for candidate-interface confidence."""
    score_batch = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    inject_candidate_sequence(score_batch, candidate_sequence)
    candidate_mask = score_batch["generate_flag"].bool()
    antigen_mask = model.bfn._mask_antigen(score_batch, candidate_mask)
    if antigen_mask is None or not antigen_mask.any():
        raise ValueError("candidate_interface_v1 requires explicit antigen context")
    score_batch["mask_antigen"] = antigen_mask
    scored = model.score(score_batch, fixed_t=fixed_t)
    interface_mask = candidate_mask.unsqueeze(-1) & antigen_mask.unsqueeze(-2)
    candidate_plddt = scored["plddt"][candidate_mask]
    return {
        "plddt": candidate_plddt.mean().item(),
        "plddt_std": candidate_plddt.std(unbiased=False).item(),
        "iptm": scored["iptm"].mean().item(),
        "pae": scored["pae"][interface_mask].mean().item(),
        "pae_scope": "candidate_to_antigen",
    }


def score_bfn_candidate(
    pdb_path,
    region_spec,
    candidate_sequence,
    context_chains=None,
    device=None,
    fixed_t=0.5,
    model=None,
):
    """Build and score a supplied candidate without sequence generation."""
    if device is None:
        device = _detect_device()
    if model is None:
        model, _ = load_bfn(device)
    batch = build_region_batch(pdb_path, region_spec, context_chains=context_chains, device=device)
    inject_candidate_sequence(batch, candidate_sequence)
    with torch.no_grad():
        return model.score(batch, fixed_t=fixed_t)


def run_bfn_design(
    pdb_path,
    region_spec,
    num_samples=10,
    stochastic=True,
    context_chains=None,
    device=None,
    sort_by=None,
    descending=True,
    disorder_guided=False,
    disorder_guided_strength=1.0,
    epitope_disorder_profile=None,
    sampling_seed=None,
    antigen_chains=None,
):
    """Run BFN protein sequence design on specified regions.

    Args:
        pdb_path: path to input PDB file
        region_spec: e.g. "B:26-33,51-58,97-113"
        num_samples: number of design samples
        stochastic: use stochastic sampling (vs deterministic)
        context_chains: list of chain IDs to include as visible context
            (e.g. ['A'] for antigen). These chains are NOT designed.
            None includes every non-design chain (Complex mode); [] includes
            no additional chains (FixBB mode).
        antigen_chains: subset of context_chains carrying antigen residues.
            Required when epitope_disorder_profile is supplied through the
            generic protein-design path.
        device: torch device
        disorder_guided: exploratory whole-CDR temperature control using an
            externally supplied epitope disorder profile. The model disorder
            head is unsupported for inference and is never used as fallback.
        disorder_guided_strength: 0..1+ pliability scaling strength.

    Returns:
        list of dicts, each with: sequence, ppl, entropy, plddt, iptm, pae.
    """
    from disorderflow.utils.misc import seed_all

    if device is None:
        device = _detect_device()

    AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY"

    model, config = load_bfn(device)
    sampling_config = getattr(config, "sampling", None)
    seed_all(getattr(sampling_config, "seed", 42))

    batch = build_region_batch(
        pdb_path,
        region_spec,
        context_chains=context_chains,
        device=device,
        antigen_chains=antigen_chains,
    )
    gen_mask = batch["generate_flag"][0].bool()

    sample_opt = {
        "deterministic": not stochastic,
        "num_recycles": 3,
        "return_disorder": False,
    }
    if epitope_disorder_profile is not None:
        mask_antigen = model.bfn._mask_antigen(batch, batch["generate_flag"].bool())
        profile_values = torch.as_tensor(
            epitope_disorder_profile, dtype=torch.float32, device=device
        ).flatten()
        antigen_count = int(mask_antigen.sum().item())
        if len(profile_values) != antigen_count:
            raise ValueError(
                f"Epitope disorder profile has {len(profile_values)} residues; "
                f"context has {antigen_count}"
            )
        condition = torch.zeros_like(batch["aa"], dtype=torch.float32)
        condition[mask_antigen] = profile_values
        batch["mask_antigen"] = mask_antigen
        batch["epitope_disorder_profile"] = condition
        pliability = torch.zeros_like(condition)
        pliability[batch["generate_flag"].bool()] = profile_values.mean()
        sample_opt["disorder_pliability"] = pliability
    if disorder_guided:
        if epitope_disorder_profile is None:
            raise ValueError(
                "disorder_guided requires epitope_disorder_profile; "
                "the model disorder head is unsupported for inference"
            )
        sample_opt["disorder_guided"] = True
        sample_opt["disorder_guided_strength"] = disorder_guided_strength
    results_list = []

    for i in range(num_samples):
        if sampling_seed is not None:
            seed_all(int(sampling_seed) + i)
        with torch.no_grad():
            traj = model.sample(batch, sample_opt=sample_opt)

        pred_aa = traj[0][2][0][gen_mask]
        seq = "".join(AA_LETTERS[a] if a < 20 else "X" for a in pred_aa.cpu())

        logits = traj["pred_logits"][0][gen_mask]
        lp = torch.log_softmax(logits[..., :20], dim=-1)
        probs = torch.exp(lp)
        nll = -lp[range(len(pred_aa)), pred_aa].mean()
        ppl = torch.exp(nll).item()

        entropy = -(probs * lp).sum(dim=-1).mean().item()
        max_prob = probs.max(dim=-1).values.mean().item()
        plddt_val = traj["plddt"][0][gen_mask].mean().item()
        plddt_per_res = traj["plddt"][0][gen_mask]
        plddt_std = plddt_per_res.std().item()
        iptm_val = traj["iptm"][0].item()
        pae_val = traj["pae"][0][gen_mask][:, gen_mask].mean().item()

        result = {
            "sequence": seq,
            "ppl": ppl,
            "entropy": entropy,
            "max_prob": max_prob,
            "plddt": plddt_val,
            "plddt_std": plddt_std,
            "iptm": iptm_val,
            "pae": pae_val,
        }
        if model.bfn.receiver.confidence_head_kind == "candidate_interface_v1":
            result.update(_score_exact_candidate_confidence(model, batch, seq))
            result["confidence_head_kind"] = "candidate_interface_v1"

        contact = traj.get("contact")
        if contact is not None:
            result["contact_score"] = torch.sigmoid(contact[0][gen_mask]).mean().item()

        state_compatibility = traj.get("state_compatibility")
        if state_compatibility is not None:
            result["state_compatibility"] = state_compatibility[0].item()

        results_list.append(result)

    # Optional ranking. Without this (the historical default), callers like
    # run_oc_v14_p1.py took designs[:top_n] = the first N *random* samples, so
    # the AF2 yield test was effectively blind sampling — a root cause of the
    # low OC yield (designs all had AF2 ipTM ~0.05). With a calibrated
    # confidence head, sort_by='iptm' picks the highest-self-confidence designs
    # to send to AF2, turning the pipeline into confidence-ranked selection.
    if sort_by is not None and results_list:
        reverse = bool(descending)
        results_list.sort(key=lambda r: r.get(sort_by, float("-inf")), reverse=reverse)

    return results_list
