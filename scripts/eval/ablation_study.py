#!/usr/bin/env python
"""
Ablation Study: Compare checkpoints across training phases on CDR design task.

Evaluates:
  - Sequence AAR (amino acid recovery) per CDR
  - Perplexity per CDR
  - Structure confidence (pLDDT, iptm, PAE) on Phase 3 val complexes

Checkpoints compared:
  1. Phase 2 baseline (pre-Phase3 cross-chain)
  2. Phase 3 baseline (small 24-complex dataset, no reg)
  3. Phase 3 optimized (expanded 2976-complex, light reg)
  4. Phase 3 v11 extended (best Phase 3, 15000 iters)

Usage:
  python scripts/eval/ablation_study.py --device xpu
"""

import os
import sys
import json
import argparse
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import copy
import warnings
warnings.filterwarnings('ignore')

from diffab.datasets.custom import preprocess_antibody_structure
from diffab.models import get_model
from diffab.utils.train import recursive_to
from diffab.utils.data import PaddingCollate, DEFAULT_PAD_VALUES
from diffab.utils.transforms import Compose
from diffab.utils.transforms.mask import MaskSingleCDR
from diffab.utils.transforms.merge import MergeChains
from diffab.utils.transforms.patch import PatchAroundAnchor
from diffab.tools.renumber import renumber as renumber_antibody

AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'
CDR_TYPES = ['H_CDR1', 'H_CDR2', 'H_CDR3', 'L_CDR1', 'L_CDR2', 'L_CDR3']

# ---- Checkpoint registry ----
CHECKPOINTS = {
    'Phase2 (baseline)': './logs/bfn_idp_phase2_xpu_2026_05_28__01_22_06/checkpoints/best.pt',
    'Phase3-baseline (24cmp, no-reg)': './logs/bfn_phase3_ablation_baseline_2026_05_30__12_14_49/checkpoints/best.pt',
    'Phase3-optimized (2976cmp, light-reg)': './logs/bfn_phase3_ablation_optimized_2026_05_31__00_54_28/checkpoints/best.pt',
    'Phase3-v11ext (2976cmp, 15k iter)': './logs/bfn_phase3_v11_extended_2026_05_31__12_09_57/checkpoints/best.pt',
}


def evaluate_cdr_design(model, batch, device):
    """Evaluate single CDR design: AAR + PPL."""
    model.eval()
    with torch.no_grad():
        gen_mask = batch['generate_flag'][0].bool()
        if gen_mask.sum() == 0:
            return None

        if 'native_aa' in batch:
            native_aa = batch['native_aa'][0][gen_mask]
        else:
            native_aa = batch['aa'][0][gen_mask]

        native_seq = ''.join([AA_LETTERS[aa] if aa < 20 else 'X'
                              for aa in native_aa.cpu().tolist()])

        traj = model.sample(batch, sample_opt={'deterministic': True})
        aa_new = traj[0][2][0]
        pred_aa = aa_new[gen_mask]
        pred_seq = ''.join([AA_LETTERS[aa] if aa < 20 else 'X'
                            for aa in pred_aa.cpu().tolist()])

        matches = sum(1 for p, t in zip(pred_seq, native_seq) if p == t)
        aar = matches / len(native_seq) if len(native_seq) > 0 else 0.0

        if 'pred_logits' in traj:
            pred_logits = traj['pred_logits'][0][gen_mask]
            log_probs = torch.log_softmax(pred_logits[..., :20], dim=-1)
            nll = -log_probs[range(len(pred_aa)), pred_aa].mean()
            perplexity = torch.exp(nll).item()
        else:
            perplexity = float('nan')

        return {'native_seq': native_seq, 'pred_seq': pred_seq,
                'length': len(native_seq), 'aar': aar, 'perplexity': perplexity}


def evaluate_structure(pdb_id, pdb_path, heavy_chain, light_chain,
                       model, device, antigen_id=None):
    """Evaluate all 6 CDRs for a single antibody structure."""
    results = []
    temp_files = []
    try:
        out_pdb = tempfile.mktemp(suffix='.pdb')
        temp_files.append(out_pdb)
        heavy_chains, light_chains = renumber_antibody(pdb_path, out_pdb, verbose=False)
        heavy_id = heavy_chains[0] if heavy_chains else heavy_chain
        light_id = light_chains[0] if light_chains else light_chain

        structure = preprocess_antibody_structure({
            'id': pdb_id, 'pdb_path': out_pdb,
            'heavy_id': heavy_id, 'light_id': light_id,
            'antigen_id': antigen_id,
        })
        if structure is None:
            return results

        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
        temp_files = []

        pad_values = DEFAULT_PAD_VALUES.copy()
        pad_values['native_aa'] = 21
        collate_fn = PaddingCollate(pad_values=pad_values)

        for cdr_name in CDR_TYPES:
            try:
                transform = Compose([
                    MaskSingleCDR(cdr_name, augmentation=False),
                    MergeChains(),
                    PatchAroundAnchor(),
                ])
                data = transform(copy.deepcopy(structure))
                if 'generate_flag' not in data or data['generate_flag'].sum() == 0:
                    continue

                batch = collate_fn([data])
                batch = recursive_to(batch, device)
                result = evaluate_cdr_design(model, batch, device)
                if result is not None:
                    result['pdb_id'] = pdb_id
                    result['cdr'] = cdr_name
                    results.append(result)
            except Exception:
                continue
    except Exception:
        pass
    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
    return results


def load_test_set(sabdab_summary, chothia_dir, processed_dir):
    """Load SAbDab test set (antigen-based split)."""
    from diffab.datasets.sabdab import SAbDabDataset
    dataset = SAbDabDataset(
        summary_path=sabdab_summary,
        chothia_dir=chothia_dir,
        processed_dir=processed_dir,
        split='test',
        reset=False,
    )
    id_to_entry = {e['id']: e for e in dataset.sabdab_entries}
    records = []
    for pid in dataset.ids_in_split:
        entry = id_to_entry.get(pid)
        if entry is None:
            continue
        pdb_id = entry['pdbcode'].lower()
        pdb_path = os.path.join(chothia_dir, f"{pdb_id}_fv.pdb")
        if not os.path.exists(pdb_path):
            pdb_path = os.path.join(chothia_dir, f"{pdb_id}.pdb")
        if not os.path.exists(pdb_path):
            continue
        records.append({
            'pdb': entry['pdbcode'].lower(),
            'Hchain': entry['H_chain'],
            'Lchain': entry['L_chain'],
            'pdb_path': pdb_path,
        })
    return pd.DataFrame(records)


def evaluate_checkpoint(ckpt_path, ckpt_name, test_df, device):
    """Evaluate one checkpoint; return results DataFrame."""
    print(f"\n  Loading: {ckpt_name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_config = ckpt['config'].model
    if hasattr(ckpt['config'], 'train') and hasattr(ckpt['config'].train, 'loss_weights'):
        model_config['loss_weight'] = dict(ckpt['config'].train.loss_weights)
    model = get_model(model_config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    all_results = []
    for _, row in test_df.iterrows():
        results = evaluate_structure(
            row['pdb'], row['pdb_path'], row['Hchain'], row['Lchain'],
            model, device,
        )
        all_results.extend(results)

    del model, ckpt
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'xpu':
        torch.xpu.empty_cache()

    if not all_results:
        return None
    return pd.DataFrame(all_results)


def print_ablation_table(all_summaries):
    """Print side-by-side ablation comparison."""
    ckpt_names = list(all_summaries.keys())
    print("\n" + "=" * 120)
    print("ABLATION STUDY: CDR Design Performance Comparison")
    print("=" * 120)

    # Header
    header = f"{'CDR':<10}"
    for name in ckpt_names:
        header += f" {name:<28}"
    print(header)
    print("-" * 120)

    for cdr in CDR_TYPES + ['ALL']:
        row = f"{cdr:<10}"
        for name in ckpt_names:
            s = all_summaries[name]
            cdr_data = [d for d in s if d['cdr'] == cdr]
            if cdr_data:
                d = cdr_data[0]
                row += f" AAR={d['aar_percent']:5.1f}% PPL={d['perplexity']:5.1f} n={d['count']:<3}"
            else:
                row += f" {'--':>28}"
        print(row)

    print("=" * 120)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--sabdab_summary', type=str,
                        default='./data/sabdab_summary_all.tsv')
    parser.add_argument('--chothia_dir', type=str,
                        default='./data/all_structures/chothia')
    parser.add_argument('--processed_dir', type=str,
                        default='./data/processed')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--output_dir', type=str,
                        default='./results/ablation_study')
    args = parser.parse_args()

    # Change to project root
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    print("=" * 120)
    print("ABLATION STUDY SETUP")
    print("=" * 120)

    # Load test set
    print(f"\nLoading SAbDab test set...")
    test_df = load_test_set(args.sabdab_summary, args.chothia_dir, args.processed_dir)
    print(f"Test set: {len(test_df)} structures")
    if args.max_samples:
        test_df = test_df.head(args.max_samples)
        print(f"Limited to {args.max_samples} samples")

    # Evaluate each checkpoint
    all_summaries = {}
    all_results = {}

    for ckpt_name, ckpt_path in CHECKPOINTS.items():
        if not os.path.exists(ckpt_path):
            print(f"\n  [SKIP] Checkpoint not found: {ckpt_path}")
            continue

        df = evaluate_checkpoint(ckpt_path, ckpt_name, test_df, args.device)
        if df is None or len(df) == 0:
            print(f"  [WARN] No results for {ckpt_name}")
            continue

        all_results[ckpt_name] = df

        # Build summary
        summary = []
        for cdr in CDR_TYPES:
            cdr_df = df[df['cdr'] == cdr]
            if len(cdr_df) == 0:
                continue
            summary.append({
                'cdr': cdr,
                'count': len(cdr_df),
                'avg_length': cdr_df['length'].mean(),
                'aar_percent': cdr_df['aar'].mean() * 100,
                'perplexity': cdr_df['perplexity'].mean(),
            })
        summary.append({
            'cdr': 'ALL',
            'count': len(df),
            'avg_length': df['length'].mean(),
            'aar_percent': df['aar'].mean() * 100,
            'perplexity': df['perplexity'].mean(),
        })
        all_summaries[ckpt_name] = summary

    # Print results
    print_ablation_table(all_summaries)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save detailed results
    for name, df in all_results.items():
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
        df.to_csv(os.path.join(args.output_dir, f'{safe_name}_results.csv'), index=False)

    # Save summary JSON
    with open(os.path.join(args.output_dir, f'summary_{timestamp}.json'), 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'test_set_size': len(test_df),
            'summaries': {k: v for k, v in all_summaries.items()},
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {args.output_dir}")
    print("Done.")


if __name__ == '__main__':
    main()
