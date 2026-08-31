#!/usr/bin/env python
"""JAX-native AlphaFold2 Multimer runner for antibody-epitope validation.

Avoids the TensorFlow dependency by:
1. Mocking tensorflow at import time (alphafold's TF code is not actually
   used for JAX inference — only for MSA feature generation which we do in numpy)
2. Building input features in pure numpy
3. Running AlphaFold multimer model via JAX/Haiku directly

Uses cached model weights from ~/.cache/colabfold/params/.
"""

import os
import sys
import types
import json
import time
import pickle
import logging
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

# ---- Suppress verbose logging ----
logging.basicConfig(level=logging.WARNING)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

# ---- Mock tensorflow before any alphafold import ----
import importlib.machinery

class _MockModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return MagicMock()

for mod_name in ('tensorflow', 'tensorflow.compat', 'tensorflow.compat.v1',
                 'tensorflow.io', 'tensorflow.train'):
    if mod_name not in sys.modules:
        mock = _MockModule(mod_name)
        # Python 3.14 requires __spec__ to be a valid ModuleSpec (None raises ValueError)
        mock.__spec__ = importlib.machinery.ModuleSpec(mod_name, None)
        sys.modules[mod_name] = mock
        # Nested modules
        parent = '.'.join(mod_name.split('.')[:-1])
        if parent and parent in sys.modules:
            setattr(sys.modules[parent], mod_name.split('.')[-1], mock)


# ---- Amino acid constants ----
AA_ORDER = 'ARNDCQEGHILKMFPSTWYV'  # AlphaFold canonical order
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}
AA3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

# AlphaFold residue constants (from common/residue_constants.py)
RESTYPE_ORDER = 'ARNDCQEGHILKMFPSTWYV'
RESTYPE_NUM = len(RESTYPE_ORDER)
NUM_ATOM_TYPES = 37  # all-atom representation

# ---- Model loading ----

# Global cache for loaded model runner and params
_MODEL_RUNNER = None
_MODEL_CONFIG = None
_CACHED_PARAMS = None
_CACHED_FORWARD = None
_CACHED_CFG = None
_CACHED_MODEL_NUMBER = None


def _load_model(data_dir=None):
    """Load AlphaFold multimer model using colabfold's JAX-native loader.

    Uses colabfold.alphafold.models to load model weights and
    alphafold.model.model.RunModel for inference. Cached globally.
    """
    global _MODEL_RUNNER, _MODEL_CONFIG

    if _MODEL_RUNNER is not None:
        return _MODEL_RUNNER, _MODEL_CONFIG

    if data_dir is None:
        data_dir = os.path.expanduser('~/.cache/colabfold')

    from alphafold.model import model, config
    from colabfold.alphafold.models import get_model_haiku_params

    model_type = 'alphafold2_multimer_v3'
    model_number = 1

    # Load model config
    from colabfold.alphafold.models import model_to_config_name
    config_name = model_to_config_name(model_type, model_number)
    model_config = config.model_config(config_name)

    # Single-sequence mode: minimal MSA
    model_config.model.embeddings_and_evoformer.num_msa = 1
    model_config.model.embeddings_and_evoformer.num_extra_msa = 1
    model_config.model.embeddings_and_evoformer.use_cluster_profile = False

    # Disable unused heads for speed
    model_config.model.heads.distogram.weight = 0.0
    model_config.model.heads.masked_msa.weight = 0.0
    model_config.model.heads.experimentally_resolved.weight = 0.0

    # No dropout at eval time
    model_config.model.global_config.eval_dropout = False
    model_config.model.global_config.bfloat16 = False

    # Load Haiku params
    params = get_model_haiku_params(
        model_type=model_type,
        model_number=model_number,
        data_dir=str(data_dir),
    )

    # Create RunModel
    model_runner = model.RunModel(
        model_config,
        params,
    )

    _MODEL_RUNNER = model_runner
    _MODEL_CONFIG = model_config

    return model_runner, model_config


# ---- Feature construction (numpy, no TF) ----

def _sequence_to_onehot(sequence):
    """Convert AA sequence to one-hot [N_res, 21]."""
    n = len(sequence)
    onehot = np.zeros((n, 21), dtype=np.float32)
    for i, aa in enumerate(sequence):
        if aa in AA_TO_IDX:
            onehot[i, AA_TO_IDX[aa]] = 1.0
        # Unknown AA stays all zeros
    return onehot




def build_multimer_features(ab_seq, epi_seq):
    """Build AlphaFold multimer input features for antibody-epitope complex.

    Feature shapes follow the AlphaFold multimer model convention:
    - MSA features: [N_msa, N_res] (msa-major)
    - Sequence features: [N_res]
    - Template features: [N_templ, N_res, ...]

    Provides 2 identical MSA sequences so that sample_msa(num_msa=1)
    leaves at least 1 extra MSA sequence for the extra MSA stack.

    Args are assumed to be pre-filtered (only standard AAs in AA_TO_IDX).
    """
    antibody_chains = ab_seq.split(':')
    sequences = antibody_chains + [epi_seq]
    ab_len = sum(len(sequence) for sequence in antibody_chains)
    epi_len = len(epi_seq)
    N = ab_len + epi_len

    full_seq_int = np.array(
        [AA_TO_IDX[aa] for aa in ''.join(sequences)], dtype=np.int32)

    # ---- Assembly features [N_res] ----
    asym_id = np.concatenate([
        np.full(len(sequence), chain_index, dtype=np.int64)
        for chain_index, sequence in enumerate(sequences)
    ])
    entity_id = asym_id.copy()
    sym_id = asym_id.copy()
    seq_mask = np.ones(N, dtype=np.float32)
    residue_index = np.arange(N, dtype=np.int32)

    # ---- MSA features [N_msa, N_res] ----
    # Provide 2 identical sequences: 1 for sampled msa, 1 for extra_msa
    num_msa = 2
    msa = np.tile(full_seq_int[None, :], (num_msa, 1)).astype(np.int32)
    msa_mask = np.ones((num_msa, N), dtype=np.float32)
    deletion_matrix = np.zeros((num_msa, N), dtype=np.float32)
    bert_mask = np.ones((num_msa, N), dtype=np.float32)

    # ---- Build feature dict ----
    feature_dict = {
        # Sequence identifiers
        'aatype': full_seq_int.astype(np.int64),
        'residue_index': residue_index.astype(np.int32),
        'seq_length': np.array(N, dtype=np.int32),
        'is_distillation': np.array(0.0, dtype=np.float32),

        # Chain/assembly
        'asym_id': asym_id,
        'entity_id': entity_id,
        'sym_id': sym_id,
        'seq_mask': seq_mask,
        'num_ensemble': np.array(1, dtype=np.int32),

        # MSA (msa-major: [N_msa, N_res])
        'msa': msa,
        'msa_mask': msa_mask,
        'deletion_matrix': deletion_matrix,
        'bert_mask': bert_mask,

        # Template features (needed when template.enabled=True)
        # Provide empty template of size 1 for model compatibility
        'template_aatype': np.zeros((1, N), dtype=np.int64),
        'template_all_atom_positions': np.zeros((1, N, 37, 3), dtype=np.float32),
        'template_all_atom_mask': np.zeros((1, N, 37), dtype=np.float32),
        'template_mask': np.zeros((1,), dtype=np.float32),
        'template_pseudo_beta': np.zeros((1, N, 3), dtype=np.float32),
        'template_pseudo_beta_mask': np.zeros((1, N), dtype=np.float32),
        'template_sum_probs': np.zeros((1,), dtype=np.float32),

        # Ground truth (empty)
        'all_atom_positions': np.zeros((N, 37, 3), dtype=np.float32),
        'all_atom_mask': np.zeros((N, 37), dtype=np.float32),
        'resolution': np.array([0.0], dtype=np.float32),
    }

    return feature_dict


# ---- Inference ----

def run_multimer_prediction(ab_seq, epi_seq, data_dir=None,
                            num_recycle=3, jax_random_seed=42,
                            return_structure=False, model_number=1):
    """Run AlphaFold multimer prediction on antibody-epitope complex.

    Uses direct Haiku model application, bypassing the TF feature processing
    pipeline. Features are built in pure numpy and fed directly to the model.

    Args:
        ab_seq: antibody amino acid sequence (1-letter). For multi-chain
            antibodies (e.g. Fab), separate chains with ':' (colon):
            "VH_SEQUENCE:VL_SEQUENCE". Do NOT concatenate without separator.
        epi_seq: epitope amino acid sequence (1-letter)
        data_dir: AlphaFold params directory
        num_recycle: number of recycling iterations
        jax_random_seed: random seed for JAX
        return_structure: if True, also return final_atom_positions (N, 37, 3)
                          and final_atom_mask (N, 37) from structure module

    Returns:
        dict with plddt, ptm, iptm, max_pae, interface_pae, success, elapsed,
        and optionally final_atom_positions, final_atom_mask if return_structure
    """
    import jax
    import jax.numpy as jnp
    # JAX 0.10+ compat: jnp.clip changed arg names (a_max→max, a_min→min).
    # The alphafold package (colabfold 2.3.13) uses the old API.
    _orig_clip = jnp.clip
    def _compat_clip(a, a_min=None, a_max=None, min=None, max=None, **kwargs):
        if min is None and a_min is not None: min = a_min
        if max is None and a_max is not None: max = a_max
        return _orig_clip(a, min=min, max=max, **kwargs)
    jnp.clip = _compat_clip
    import haiku as hk

    global _CACHED_PARAMS, _CACHED_FORWARD, _CACHED_CFG, _CACHED_MODEL_NUMBER

    if data_dir is None:
        data_dir = os.path.expanduser('~/.cache/colabfold')

    from alphafold.model import model as af_model, config as af_config
    from alphafold.model import modules_multimer
    from alphafold.model.utils import flat_params_to_haiku

    model_type = 'alphafold2_multimer_v3'
    model_number = int(model_number)
    if model_number not in range(1, 6):
        raise ValueError("AF2-Multimer model_number must be between 1 and 5")

    # Reuse cached config/params/forward to avoid JAX memory accumulation (crashes
    # after ~40 calls when re-creating Haiku transforms every time).
    if _CACHED_FORWARD is None:
        from colabfold.alphafold.models import model_to_config_name
        config_name = model_to_config_name(model_type, model_number)
        cfg = af_config.model_config(config_name)

        # Configure for single-sequence mode
        cfg.model.embeddings_and_evoformer.num_msa = 1
        cfg.model.embeddings_and_evoformer.num_extra_msa = 1
        cfg.model.embeddings_and_evoformer.template.enabled = False
        cfg.model.embeddings_and_evoformer.use_cluster_profile = False
        cfg.model.num_recycle = num_recycle
        cfg.model.global_config.eval_dropout = False
        cfg.model.global_config.bfloat16 = False
        cfg.model.heads.distogram.weight = 0.0
        cfg.model.heads.masked_msa.weight = 0.0
        cfg.model.heads.experimentally_resolved.weight = 0.0
        cfg.model.calc_extended_ptm = True  # needed to return PAE matrix
        cfg.model.rank_by = 'plddt'

        # Load params once
        npz_path = os.path.join(data_dir, 'params',
                                f'params_model_{model_number}_multimer_v3.npz')
        flat_params = np.load(npz_path, allow_pickle=False)
        params = flat_params_to_haiku(flat_params)

        # Define forward function once
        def forward_fn(batch):
            model = modules_multimer.AlphaFold(cfg.model)
            return model(
                batch=batch,
                is_training=False,
                return_representations=False,
            )

        _CACHED_FORWARD = hk.transform(forward_fn)
        _CACHED_PARAMS = params
        _CACHED_CFG = cfg
        _CACHED_MODEL_NUMBER = model_number
    else:
        if model_number != _CACHED_MODEL_NUMBER:
            raise RuntimeError(
                "A worker cannot switch AF2 model_number after model loading")
        cfg = _CACHED_CFG
        # Update num_recycle in case it changed between calls
        cfg.model.num_recycle = num_recycle

    params = _CACHED_PARAMS

    # Filter sequences (same as build_multimer_features does internally)
    antibody_chains = [
        ''.join(aa for aa in chain.upper() if aa in AA_TO_IDX)
        for chain in ab_seq.split(':')
    ]
    ab_seq_clean = ':'.join(antibody_chains)
    epi_seq_clean = ''.join(aa for aa in epi_seq.upper() if aa in AA_TO_IDX)
    ab_len = sum(len(chain) for chain in antibody_chains)
    epi_len = len(epi_seq_clean)

    # Build features
    feature_dict = build_multimer_features(ab_seq_clean, epi_seq_clean)

    forward = _CACHED_FORWARD
    rng = jax.random.PRNGKey(jax_random_seed)

    # Convert features to JAX arrays
    import numpy as onp
    jax_batch = {}
    for k, v in feature_dict.items():
        if isinstance(v, onp.ndarray):
            jax_batch[k] = jnp.array(v)
        elif isinstance(v, (int, float)):
            jax_batch[k] = jnp.array(v)

    # Run prediction
    t0 = time.time()
    try:
        result = forward.apply(params, rng, jax_batch)
        elapsed = time.time() - t0

        # Confidence metrics are directly available from the model output
        # (newer alphafold computes them in the forward pass)
        plddt = onp.array(result['plddt'])
        mean_plddt = float(plddt.mean()) / 100.0

        ptm = float(onp.array(result.get('ptm', 0.0)).flat[0])
        iptm = float(onp.array(result.get('iptm', 0.0)).flat[0])

        # PAE matrix — check if it's a dict (old format) or array (new format)
        pae_raw = result['predicted_aligned_error']
        if hasattr(pae_raw, 'keys'):
            # Old format: dict with logits/breaks
            from alphafold.common.confidence import get_confidence_metrics
            result_np = jax.tree.map(
                lambda x: onp.array(x) if isinstance(x, jnp.ndarray) else x,
                result)
            mask = onp.ones(ab_len + epi_len)
            conf = get_confidence_metrics(result_np, mask=mask)
            pae = onp.array(conf['predicted_aligned_error'])
            if ptm == 0.0:
                ptm = float(onp.array(conf.get('ptm', 0.0)).flat[0])
            if iptm == 0.0:
                iptm = float(onp.array(conf.get('iptm', 0.0)).flat[0])
        else:
            # New format: direct array
            pae = onp.array(pae_raw)
        max_pae = float(pae.max())

        # Interface PAE: slice antibody chain vs epitope chain
        if pae.shape[0] >= ab_len + epi_len and ab_len < pae.shape[0]:
            interface_pae = float(pae[:ab_len, ab_len:].mean())
        else:
            interface_pae = None

        out = {
            'success': True,
            'plddt': mean_plddt,
            'plddt_per_residue': (plddt / 100.0).tolist(),
            'ptm': ptm,
            'iptm': iptm,
            'max_pae': max_pae,
            'interface_pae': interface_pae,
            'pae_matrix': pae.tolist(),
            'elapsed': elapsed,
        }

        if return_structure:
            sm = result.get('structure_module', {})
            positions = sm.get('final_atom_positions')
            atom_mask = sm.get('final_atom_mask')
            if positions is not None:
                out['final_atom_positions'] = onp.array(positions)
            if atom_mask is not None:
                out['final_atom_mask'] = onp.array(atom_mask)
            from alphafold.common import protein
            b_factors = onp.repeat(plddt[:, None], NUM_ATOM_TYPES, axis=1)
            prediction = protein.from_prediction(
                {key: onp.array(value) for key, value in jax_batch.items()},
                jax.tree.map(lambda value: onp.array(value), result),
                b_factors=b_factors,
                remove_leading_feature_dimension=False,
            )
            out['pdb'] = protein.to_pdb(prediction)

        return out
    except Exception as e:
        import traceback
        elapsed = time.time() - t0
        return {
            'success': False,
            'error': str(e)[:500],
            'traceback': traceback.format_exc()[-500:],
            'elapsed': elapsed,
        }


def validate_antibody_epitope(ab_seq, epi_seq, data_dir=None,
                              num_recycle=3, verbose=False):
    """Convenience: validate a single antibody-epitope pair.

    Args:
        ab_seq: full antibody sequence with designed CDRs
        epi_seq: epitope sequence
        data_dir: AlphaFold params directory
        num_recycle: AF2 recycling steps
        verbose: print timing info

    Returns:
        dict with plddt, iptm, ptm, max_pae, interface_pae, success
    """
    if verbose:
        print(f'  AF2 multimer: {len(ab_seq)}+{len(epi_seq)} AA, '
              f'recycle={num_recycle}', end='', flush=True)
    t0 = time.time()
    result = run_multimer_prediction(
        ab_seq, epi_seq, data_dir=data_dir, num_recycle=num_recycle
    )
    if verbose and result.get('success'):
        ipae = result.get('interface_pae')
        ipae_str = f'iPAE={ipae:.1f}' if ipae is not None else 'iPAE=N/A'
        print(f' | ipTM={result["iptm"]:.3f} pLDDT={result["plddt"]:.3f} '
              f'{ipae_str} ({result["elapsed"]:.0f}s)')
    elif verbose:
        print(f' | FAILED')
    return result
