#!/usr/bin/env python
"""
General Protein Sequence Design using BFN.

Usage Examples:
  # Design specific region on chain A
  python design_protein.py input.pdb --design "A:10-25" --config config.yml --device cpu

  # Design multiple regions across chains
  python design_protein.py input.pdb --design "A:10-25 B:5-15" --config config.yml --device cpu

  # Design with stochastic sampling
  python design_protein.py input.pdb --design "A:10-25" --config config.yml --num_samples 5 --stochastic

  # Evaluation mode (compare to native)
  python design_protein.py input.pdb --design "A:10-25" --config config.yml --eval
"""
import os
import re
import argparse
import torch
import numpy as np
from tqdm.auto import tqdm

from antibodydesignbfn.datasets.protein import preprocess_protein_structure
from antibodydesignbfn.models import get_model
from antibodydesignbfn.utils.train import recursive_to
from antibodydesignbfn.utils.misc import load_config, seed_all
from antibodydesignbfn.utils.data import PaddingCollate
from antibodydesignbfn.utils.transforms import get_transform

AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'


def parse_design_regions(design_str):
    """
    Parse design region specification.

    Args:
        design_str: Format "CHAIN:START-END CHAIN:R1,R2,R3" or "CHAIN:START-END"

    Returns:
        dict mapping chain_id -> list of 0-based residue indices

    Examples:
        "A:10-25" -> {'A': [10, 11, ..., 25]}
        "A:10,20,30" -> {'A': [10, 20, 30]}
        "A:10-25 B:5-15" -> {'A': [10,...,25], 'B': [5,...,15]}
    """
    regions = {}
    # Split by chain:spec pairs (chain IDs can be alphanumeric)
    pattern = r'([A-Za-z0-9]+):([0-9,\-\s]+)'
    matches = re.findall(pattern, design_str)

    if not matches:
        raise ValueError(
            f'Invalid design specification: "{design_str}". '
            f'Expected format: "CHAIN:START-END" or "CHAIN:R1,R2,R3"'
        )

    for chain_id, spec in matches:
        indices = []
        for segment in spec.split(','):
            segment = segment.strip()
            if not segment:
                continue
            if '-' in segment:
                parts = segment.split('-')
                if len(parts) != 2:
                    raise ValueError(f'Invalid range: {segment}')
                start, end = int(parts[0].strip()), int(parts[1].strip())
                indices.extend(range(start, end + 1))
            else:
                indices.append(int(segment))
        regions[chain_id] = sorted(set(indices))

    return regions


def design_protein(args):
    # Load config
    config, _ = load_config(args.config)
    seed_all(config.sampling.seed if hasattr(config.sampling, 'seed') else 42)

    # Parse design regions
    design_regions = parse_design_regions(args.design)
    print(f'[INFO] Design regions:')
    for chain_id, indices in design_regions.items():
        print(f'  Chain {chain_id}: positions {indices[0]}..{indices[-1]} ({len(indices)} residues)')

    # Preprocess structure (generic, no renumbering)
    pdb_path = args.pdb_path
    pdb_id = os.path.basename(pdb_path).replace('.pdb', '')
    print(f'[INFO] Processing: {pdb_path}')

    structure = preprocess_protein_structure(pdb_path, chain_ids=list(design_regions.keys()))
    if structure is None:
        print('[ERROR] Failed to parse protein structure')
        return

    print(f'[INFO] Loaded {structure["num_chains"]} chain(s): {structure["all_chain_ids"]}')

    # Build transform pipeline
    transform = get_transform([
        {'type': 'mask_region', 'regions': design_regions},
        {'type': 'merge_protein'},
        {'type': 'patch_protein'},
    ])
    data = transform(structure)

    # Collate
    collate_fn = PaddingCollate()
    batch = collate_fn([data])
    batch = recursive_to(batch, args.device)

    # Get design mask
    gen_mask = batch['generate_flag'][0].bool()
    if gen_mask.sum() == 0:
        print('[ERROR] No residues selected for design. Check your --design regions.')
        return
    print(f'[INFO] Designing {gen_mask.sum().item()} residue(s)')

    # Get native sequence (eval mode)
    native_seq = None
    if args.eval:
        native_aa = batch['aa'][0][gen_mask]
        native_seq = ''.join([AA_LETTERS[aa] if aa < 20 else 'X' for aa in native_aa.cpu().tolist()])
        print(f'[INFO] Native sequence: {native_seq}')

    # Load model
    print(f'[INFO] Loading model: {config.model.checkpoint}')
    ckpt = torch.load(config.model.checkpoint, map_location='cpu', weights_only=False)
    model_config = ckpt['config'].model
    if hasattr(ckpt['config'], 'train') and hasattr(ckpt['config'].train, 'loss_weights'):
        model_config['loss_weight'] = dict(ckpt['config'].train.loss_weights)

    model = get_model(model_config).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Sample
    sample_opt = {'deterministic': not args.stochastic, 'num_recycles': 3}
    mode_str = 'stochastic' if args.stochastic else 'deterministic'
    print(f'[INFO] Generating {args.num_samples} sequence(s), mode: {mode_str}, recycles: 3')

    results = []
    best_ppl = float('inf')
    best_seq = None

    for i in tqdm(range(args.num_samples), disable=args.num_samples == 1):
        with torch.no_grad():
            traj = model.sample(batch, sample_opt=sample_opt)

        aa_new = traj[0][2][0]
        pred_aa = aa_new[gen_mask]
        pred_seq = ''.join([AA_LETTERS[aa] if aa < 20 else 'X' for aa in pred_aa.cpu().tolist()])

        # Perplexity
        pred_logits = traj['pred_logits'][0][gen_mask]
        log_probs = torch.log_softmax(pred_logits[..., :20], dim=-1)
        nll = -log_probs[range(len(pred_aa)), pred_aa].mean()
        perplexity = torch.exp(nll).item()

        # BFN内置置信度 (receiver.py 已做 sigmoid → [0,1])
        plddt_val = traj['plddt'][0][gen_mask].mean().item()
        iptm_val = traj['iptm'][0].item()
        pae_val = traj['pae'][0][gen_mask][:, gen_mask].mean().item()

        result = {
            'sample_id': i,
            'sequence': pred_seq,
            'perplexity': perplexity,
            'plddt': plddt_val,
            'iptm': iptm_val,
            'pae': pae_val,
        }

        if args.eval and native_seq:
            matches = sum(1 for p, n in zip(pred_seq, native_seq) if p == n)
            recovery = matches / len(native_seq) if len(native_seq) > 0 else 0.0
            result['recovery'] = recovery
            print(f'  {i:04d}: {pred_seq} | PPL={perplexity:.2f} | pLDDT={plddt_val:.2f} | ipTM={iptm_val:.2f} | PAE={pae_val:.1f} | Recovery={recovery*100:.1f}%')
        else:
            print(f'  {i:04d}: {pred_seq} | PPL={perplexity:.2f} | pLDDT={plddt_val:.2f} | ipTM={iptm_val:.2f} | PAE={pae_val:.1f}')

        results.append(result)

        if perplexity < best_ppl:
            best_ppl = perplexity
            best_seq = pred_seq

    # Summary
    print(f'\n{"="*60}')
    print(f'[RESULT]')
    print(f'  Best sequence:  {best_seq}')
    print(f'  Best PPL:       {best_ppl:.2f}')
    # Report confidence for best result
    best_result = min(results, key=lambda r: r['perplexity'])
    print(f'  Best pLDDT:     {best_result.get("plddt", 0):.2f}')
    print(f'  Best ipTM:      {best_result.get("iptm", 0):.2f}')
    print(f'  Best PAE:       {best_result.get("pae", 0):.1f}')
    avg_plddt = np.mean([r['plddt'] for r in results])
    avg_iptm = np.mean([r['iptm'] for r in results])
    print(f'  Avg pLDDT:      {avg_plddt:.2f}')
    print(f'  Avg ipTM:       {avg_iptm:.2f}')
    if args.eval and native_seq:
        print(f'  Native:         {native_seq}')
        avg_recovery = np.mean([r['recovery'] for r in results])
        print(f'  Avg Recovery:   {avg_recovery*100:.1f}%')
    print(f'{"="*60}')

    # Save output
    if args.output:
        import json
        output_data = {
            'pdb': pdb_id,
            'design_regions': design_regions,
            'best_sequence': best_seq,
            'best_ppl': best_ppl,
            'results': results,
        }
        if args.eval and native_seq:
            output_data['native'] = native_seq
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f'\n[INFO] Saved to {args.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='General Protein Sequence Design using BFN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Design a specific region on chain A
  python design_protein.py input.pdb --design "A:10-25" --config config.yml --device cpu

  # Design multiple regions
  python design_protein.py input.pdb --design "A:10-25 B:5-15" --config config.yml --device cpu

  # Stochastic sampling with multiple sequences
  python design_protein.py input.pdb --design "A:10-30" --config config.yml --num_samples 5 --stochastic

  # Evaluation mode
  python design_protein.py input.pdb --design "A:10-30" --config config.yml --eval --output results.json
        """
    )
    parser.add_argument('pdb_path', type=str, help='Input PDB file')
    parser.add_argument('--design', '-d', type=str, required=True,
                        help='Design regions: "CHAIN:START-END" or "CHAIN:R1,R2,R3" '
                             '(multiple chains: "A:10-25 B:5-15")')
    parser.add_argument('--config', '-c', type=str, required=True, help='Config YAML file')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output JSON file')
    parser.add_argument('--num_samples', '-n', type=int, default=1,
                        help='Number of sequences (default: 1)')
    parser.add_argument('--device', type=str, default='cpu', help='Device (default: cpu)')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic sampling (default: deterministic)')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluation mode: compare to native sequence')

    args = parser.parse_args()
    design_protein(args)
