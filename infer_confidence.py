#!/usr/bin/env python
"""Single-PDB confidence inference using fine-tuned BFN confidence heads.

Predicts pLDDT (per-residue), ipTM (scalar), and PAE (matrix) from a PDB file.

Usage:
  python infer_confidence.py --pdb 1ubq.pdb --checkpoint logs/.../best.pt
  python infer_confidence.py --pdb 1ubq.pdb --checkpoint logs/.../best.pt --output results.json
  python infer_confidence.py --pdb 1ubq.pdb --checkpoint logs/.../best.pt --device cpu
"""
import sys, os, json, argparse
from pathlib import Path

import torch
import numpy as np

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'


def parse_args():
    p = argparse.ArgumentParser(description='BFN confidence inference on a single PDB')
    p.add_argument('--pdb', required=True, help='Path to input PDB file')
    p.add_argument('--checkpoint', required=True, help='Path to BFN confidence checkpoint .pt')
    p.add_argument('--device', default='cuda', help='Device (cpu/cuda)')
    p.add_argument('--output', '-o', default=None, help='Output JSON path (default: print to stdout)')
    p.add_argument('--num_recycles', type=int, default=1, help='BFN recycle rounds')
    return p.parse_args()


def load_pdb_structure(pdb_path):
    """Parse PDB into BFN structure dict."""
    from disorderflow.datasets.protein import preprocess_protein_structure
    structure = preprocess_protein_structure(str(pdb_path))
    if structure is None:
        raise RuntimeError(f'Failed to parse PDB: {pdb_path}')
    return structure


def preprocess_for_confidence(structure):
    """Apply BFN transform pipeline for confidence inference.

    All residues are context (generate_flag=False); no design region.
    """
    from disorderflow.utils.transforms import get_transform
    from disorderflow.utils.train import recursive_to

    # Collect all chain IDs and their residue counts
    n_total = 0
    for chain_entry in structure['chains']:
        n_total += chain_entry['data']['aa'].size(0)

    # Mask all chains as context (empty regions 鈫?no generate_flag set)
    transform = get_transform([
        {'type': 'mask_region', 'regions': {}},
        {'type': 'merge_protein'},
        {'type': 'patch_protein', 'initial_patch_size': n_total + 100,
         'context_size': 0},
    ])
    data = transform(structure)
    data = recursive_to(data, 'cpu')
    return data


def load_model(checkpoint_path, device):
    """Load BFN confidence model from checkpoint."""
    from disorderflow.models import get_model
    from disorderflow.utils.misc import load_config as _lc

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mc = ckpt['config'].model
    if hasattr(ckpt['config'], 'train') and hasattr(ckpt['config'].train, 'loss_weights'):
        mc['loss_weight'] = dict(ckpt['config'].train.loss_weights)

    model = get_model(mc).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def run_inference(model, batch_data, device, num_recycles=1):
    """Run confidence inference on preprocessed batch."""
    from disorderflow.utils.data import PaddingCollate
    from disorderflow.utils.train import recursive_to

    # Build batch of 1
    collated = PaddingCollate()([batch_data])
    collated = recursive_to(collated, device)

    # All residues are context 鈥?no generation
    collated['generate_flag'] = torch.zeros_like(collated['mask_heavyatom'][:, :, 0], dtype=torch.bool)

    with torch.no_grad():
        result = model.sample(collated, sample_opt={
            'deterministic': True,
            'num_recycles': num_recycles,
        })

    return result


def extract_predictions(result, n_native):
    """Extract per-residue pLDDT, ipTM, and PAE from model output."""
    plddt = result['plddt'][0, :n_native].cpu().numpy()
    iptm = result['iptm'][0].cpu().item()
    pae = result['pae'][0, :n_native, :n_native].cpu().numpy()

    return {
        'plddt': plddt.tolist(),
        'iptm': iptm,
        'pae': pae.tolist(),
        'plddt_mean': float(np.mean(plddt)),
        'plddt_min': float(np.min(plddt)),
        'plddt_max': float(np.max(plddt)),
    }


def get_sequence(batch_data):
    """Decode amino acid sequence from BFN aa indices."""
    aa = batch_data['aa'].cpu().numpy()
    seq = ''.join(AA_LETTERS[int(a)] if 0 <= int(a) < 20 else 'X' for a in aa)
    return seq


def main():
    args = parse_args()

    if not os.path.exists(args.pdb):
        print(f'ERROR: PDB not found: {args.pdb}')
        sys.exit(1)
    if not os.path.exists(args.checkpoint):
        print(f'ERROR: Checkpoint not found: {args.checkpoint}')
        sys.exit(1)

    print(f'PDB:        {args.pdb}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Device:     {args.device}')
    print()

    # Step 1: Load and preprocess PDB
    print('Loading PDB...')
    structure = load_pdb_structure(args.pdb)
    pdb_id = structure['id']
    chain_ids = structure.get('all_chain_ids', [])
    print(f'  ID: {pdb_id}, Chains: {", ".join(chain_ids)}')

    print('Preprocessing...')
    batch_data = preprocess_for_confidence(structure)
    seq = get_sequence(batch_data)
    n_res = len(seq)
    print(f'  Sequence length: {n_res} residues')

    # Step 2: Load model
    print('Loading model...')
    model = load_model(args.checkpoint, args.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {total_params:,}')

    # Step 3: Run inference
    print('Running inference...')
    result = run_inference(model, batch_data, args.device, num_recycles=args.num_recycles)

    # Step 4: Extract predictions
    pred = extract_predictions(result, n_res)

    # Step 5: Output
    output = {
        'pdb_id': pdb_id,
        'checkpoint': args.checkpoint,
        'sequence': seq,
        'length': n_res,
        'predictions': pred,
    }

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f'\nResults saved to: {args.output}')
    else:
        # Print summary to stdout
        print(f'\n{"=" * 50}')
        print(f'  BFN Confidence Predictions 鈥?{pdb_id} ({n_res} residues)')
        print(f'{"=" * 50}')
        print(f'  ipTM:         {pred["iptm"]:.4f}')
        print(f'  pLDDT mean:   {pred["plddt_mean"]:.4f}')
        print(f'  pLDDT min:    {pred["plddt_min"]:.4f}')
        print(f'  pLDDT max:    {pred["plddt_max"]:.4f}')
        print(f'\n  Per-residue pLDDT:')
        for i, (aa, p) in enumerate(zip(seq, pred['plddt'])):
            bar = '鈻? * int(p * 20) + '鈻? * (20 - int(p * 20))
            print(f'    {i:4d} {aa}  {p:.4f}  {bar}')

    # Also print basic residue-level stats
    print(f'\nDone. Sequence: {seq[:60]}{"..." if n_res > 60 else ""}')


if __name__ == '__main__':
    main()
