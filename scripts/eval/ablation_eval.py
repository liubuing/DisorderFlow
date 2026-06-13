#!/usr/bin/env python
"""
Ablation Study: Comprehensive checkpoint comparison on CDR co-design + confidence.

Two evaluation modes:
  1. CDR Co-Design (Phase3 val set): AAR, PPL, dist, ang per CDR
     鈥?Fair sequence comparison via AAR (unbiased by label_smoothing)
  2. Confidence Prediction (confidence_v11 val set): pLDDT MAE, ipTM MAE, PAE MAE, disorder BCE

Checkpoints:
  01_Phase2_baseline        鈥?Phase2 confidence regression (seq_only, freeze_backbone)
  02_Phase3_baseline_24cmp   鈥?Phase3 co-design (24 cmp, no reg, 2k iter)
  03_Phase3_optimized_2976cmp 鈥?Phase3 co-design (2976 cmp, light reg, 2k iter)
  04_Phase3_v11ext_2976cmp   鈥?Phase3 co-design (2976 cmp, 15k iter, best Phase3)
  05_Phase3_disorder         鈥?Phase3 disorder restoration (freeze_backbone, confidence)
  06_Phase4_negative          鈥?Phase4 negative sampling (freeze_backbone, confidence)
  07_Phase5_calibrate         鈥?Phase5 calibrate (unfreeze backbone, confidence)

Usage:
  python scripts/eval/ablation_eval.py --device xpu
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)
os.chdir(_project_root)

from disorderflow.datasets import get_dataset
from disorderflow.models import get_model
from disorderflow.utils.data import PaddingCollate, DEFAULT_PAD_VALUES
from disorderflow.utils.train import recursive_to, sum_weighted_losses, ValidationLossTape, EMAModel
from disorderflow.utils.transforms import Compose
from disorderflow.utils.transforms.mask import MaskSingleCDR, MaskMultipleCDRs
from disorderflow.utils.transforms.merge import MergeChains
from disorderflow.utils.transforms.patch import PatchAroundAnchor

_LOGDIR = '/home/liubuzi/disorderflow-main/disorderflow-main/logs'

AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'
CDR_TYPES = ['H_CDR1', 'H_CDR2', 'H_CDR3', 'L_CDR1', 'L_CDR2', 'L_CDR3']

CHECKPOINTS = {
    '01_Phase2_baseline': {
        'path': f'{_LOGDIR}/bfn_idp_phase2_xpu_2026_05_28__01_22_06/checkpoints/best.pt',
        'desc': 'Phase2 (pre-cross-chain)',
        'codessign': False,   # seq_only model, skip CDR co-design
        'confidence': True,
    },
    '02_Phase3_baseline_24cmp': {
        'path': f'{_LOGDIR}/bfn_phase3_ablation_baseline_2026_05_30__12_14_49/checkpoints/best.pt',
        'desc': 'Phase3 baseline (24cmp, no reg, 2k iter)',
        'codessign': True,
        'confidence': True,
    },
    '03_Phase3_optimized_2976cmp': {
        'path': f'{_LOGDIR}/bfn_phase3_ablation_optimized_2026_05_31__00_54_28/checkpoints/best.pt',
        'desc': 'Phase3 optimized (2976cmp, light reg, 2k iter)',
        'codessign': True,
        'confidence': True,
    },
    '04_Phase3_v11ext_2976cmp': {
        'path': f'{_LOGDIR}/bfn_phase3_v11_extended_2026_05_31__12_09_57/checkpoints/best.pt',
        'desc': 'Phase3 v11ext (2976cmp, 15k iter, best)',
        'codessign': True,
        'confidence': True,
    },
    '05_Phase3_disorder': {
        'path': f'{_LOGDIR}/bfn_phase3_disorder_2026_05_31__14_52_47/checkpoints/best.pt',
        'desc': 'Phase3 disorder (freeze backbone, conf head)',
        'codessign': False,
        'confidence': True,
    },
    '06_Phase4_negative': {
        'path': f'{_LOGDIR}/bfn_phase4_negative_xpu_2026_05_31__18_27_56/checkpoints/best.pt',
        'desc': 'Phase4 negative (freeze backbone, conf head)',
        'codessign': False,
        'confidence': True,
    },
    '07_Phase5_calibrate': {
        'path': f'{_LOGDIR}/bfn_phase5_calibrate_xpu_2026_05_31__22_38_38/checkpoints/best.pt',
        'desc': 'Phase5 calibrate (unfreeze backbone)',
        'codessign': True,
        'confidence': True,
    },
    '08_Phase3_nosmooth': {
        'path': f'{_LOGDIR}/bfn_phase3_v11_nosmooth_2026_06_01__15_08_09/checkpoints/best.pt',
        'desc': 'Phase3 nosmooth (no label_smoothing)',
        'codessign': True,
        'confidence': True,
    },
}

LOSS_WEIGHTS = {'seq': 3.0, 'dist': 0.5, 'ang': 0.5, 'plddt': 1.0, 'iptm': 1.5, 'pae': 0.3}


def load_model(ckpt_path, device):
    """Load model from checkpoint, apply EMA if available."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ckpt_config = ckpt['config']
    model_config = ckpt_config.model
    model = get_model(model_config)
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    model.eval()

    if hasattr(ckpt_config, 'train') and hasattr(ckpt_config.train, 'ema_decay'):
        ema = EMAModel(model, decay=ckpt_config.train.ema_decay)
        ema.apply(model)

    return model, ckpt_config


# =========================================================================
# Part 1: CDR Co-Design Evaluation on Phase3 Val Set
# =========================================================================

def evaluate_codessign_val(ckpt_path, device):
    """Evaluate CDR co-design on Phase3 val set: AAR, PPL, dist, ang."""
    model, ckpt_config = load_model(ckpt_path, device)

    # Build dataset WITHOUT transforms 鈥?we apply per-CDR transforms manually
    from easydict import EasyDict
    val_cfg = EasyDict({
        'type': 'phase3',
        'lmdb_path': f'{_project_root}/data/phase3_processed/val.lmdb',
        'transform': [
            {'type': 'merge_chains'},
            {'type': 'patch_around_anchor'},
        ],
    })
    val_dataset = get_dataset(val_cfg)

    pad_values = DEFAULT_PAD_VALUES.copy()
    pad_values['native_aa'] = 21
    collate_fn = PaddingCollate(pad_values=pad_values)

    cdr_results = defaultdict(list)
    loss_tape = ValidationLossTape()

    # Disable mask augmentation during evaluation for deterministic results
    saved_aug_prob = model.bfn.loss_weight.get('mask_aug_prob', 0.0)
    model.bfn.loss_weight['mask_aug_prob'] = 0.0

    is_seq_only = model.bfn.receiver.seq_only
    if is_seq_only:
        tqdm.write(f"  [INFO] seq_only model 鈥?AAR via FixBB, no dist/ang")

    # Deterministic mask: all CDRs co-design for loss evaluation
    ALL_CDRS = ['H_CDR1', 'H_CDR2', 'H_CDR3', 'L_CDR1', 'L_CDR2', 'L_CDR3']
    loss_transform = Compose([
        MaskMultipleCDRs(selection=ALL_CDRS, augmentation=False),
        MergeChains(),
        PatchAroundAnchor(),
    ])

    for idx in tqdm(range(len(val_dataset)), desc='  CDR eval'):
        structure = val_dataset.get_structure(idx)
        if structure is None:
            continue

        # --- Per-CDR AAR/PPL evaluation ---
        for cdr_name in CDR_TYPES:
            try:
                import copy
                cdr_transform = Compose([
                    MaskSingleCDR(cdr_name, augmentation=False),
                    MergeChains(),
                    PatchAroundAnchor(),
                ])
                data = cdr_transform(copy.deepcopy(structure))
                if 'generate_flag' not in data or data['generate_flag'].sum() == 0:
                    continue

                gen_mask = data['generate_flag'].bool()
                native_aa = data['aa'][gen_mask]

                batch = collate_fn([data])
                batch = recursive_to(batch, device)

                traj = model.sample(batch, sample_opt={'deterministic': True})
                gen_mask_batch = batch['generate_flag'][0].bool()
                pred_aa = traj[0][2][0][gen_mask_batch]

                matches = (pred_aa == native_aa.to(device)).sum().item()
                aar = matches / len(native_aa) if len(native_aa) > 0 else 0.0

                if 'pred_logits' in traj:
                    logits = traj['pred_logits'][0][gen_mask_batch]
                    log_probs = torch.log_softmax(logits[..., :20], dim=-1)
                    nll = -log_probs[range(len(pred_aa)), pred_aa].mean()
                    ppl = torch.exp(nll).item()
                else:
                    ppl = float('nan')

                cdr_results[cdr_name].append({'aar': aar, 'ppl': ppl, 'length': len(native_aa)})
            except Exception:
                continue

        # --- Loss metrics (all CDRs co-design, deterministic) ---
        try:
            loss_data = loss_transform(copy.deepcopy(structure))
            loss_batch = collate_fn([loss_data])
            loss_batch = recursive_to(loss_batch, device)

            with torch.no_grad():
                loss_dict = model(loss_batch)
                loss_dict.pop('avg_t', None)
                loss = sum_weighted_losses(loss_dict, LOSS_WEIGHTS)
                loss_dict['overall'] = loss
            loss_tape.update(loss_dict, 1)
        except Exception:
            continue

    # Aggregate CDR results
    cdr_summary = {}
    for cdr in CDR_TYPES:
        items = cdr_results.get(cdr, [])
        if items:
            cdr_summary[cdr] = {
                'count': len(items),
                'aar': np.mean([x['aar'] for x in items]) * 100,
                'ppl': np.mean([x['ppl'] for x in items]),
                'avg_length': np.mean([x['length'] for x in items]),
            }

    # Aggregate overall AAR/PPL across all CDRs
    all_items = [x for items in cdr_results.values() for x in items]
    if all_items:
        cdr_summary['ALL'] = {
            'count': len(all_items),
            'aar': np.mean([x['aar'] for x in all_items]) * 100,
            'ppl': np.mean([x['ppl'] for x in all_items]),
            'avg_length': np.mean([x['length'] for x in all_items]),
        }

    # Aggregate loss
    n = loss_tape.total
    loss_avg = {}
    if n > 0:
        loss_avg = {k: (v / n).item() if isinstance(v, torch.Tensor) else v / n
                     for k, v in loss_tape.accumulate.items()}

    model.bfn.loss_weight['mask_aug_prob'] = saved_aug_prob
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'xpu':
        torch.xpu.empty_cache()

    return cdr_summary, loss_avg


# =========================================================================
# Part 2: Confidence Evaluation on Confidence Regression Val Set
# =========================================================================

def evaluate_confidence_val(ckpt_path, device, max_samples=None):
    """Evaluate confidence prediction on confidence_v11 val set."""
    model, ckpt_config = load_model(ckpt_path, device)

    # Disable mask augmentation during evaluation
    saved_aug_prob = model.bfn.loss_weight.get('mask_aug_prob', 0.0)
    model.bfn.loss_weight['mask_aug_prob'] = 0.0

    from easydict import EasyDict
    conf_cfg = EasyDict({
        'type': 'confidence_regression',
        'db_path': f'{_project_root}/data/confidence_merged_v11/confidence_val.lmdb',
        'max_residues': 350,
    })
    conf_dataset = get_dataset(conf_cfg)
    if max_samples and len(conf_dataset) > max_samples:
        import random
        conf_dataset._valid_indices = random.sample(conf_dataset._valid_indices, max_samples)

    conf_loader = DataLoader(conf_dataset, batch_size=1, collate_fn=PaddingCollate(),
                              shuffle=False, num_workers=0)

    loss_tape = ValidationLossTape()

    with torch.no_grad():
        for batch in tqdm(conf_loader, desc='  Conf eval', leave=False):
            batch = recursive_to(batch, device)
            loss_dict = model(batch)
            loss_dict.pop('avg_t', None)
            loss_tape.update(loss_dict, 1)

    n = loss_tape.total
    conf_avg = {}
    if n > 0:
        conf_avg = {k: (v / n).item() if isinstance(v, torch.Tensor) else v / n
                     for k, v in loss_tape.accumulate.items()}

    model.bfn.loss_weight['mask_aug_prob'] = saved_aug_prob
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'xpu':
        torch.xpu.empty_cache()

    return conf_avg


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='./results/ablation')
    parser.add_argument('--conf_max_samples', type=int, default=None,
                        help='Limit confidence val samples for speed')
    parser.add_argument('--skip_codessign', action='store_true')
    parser.add_argument('--skip_confidence', action='store_true')
    args = parser.parse_args()

    print("=" * 110)
    print("ABLATION STUDY")
    print(f"6 checkpoints x 2 evaluation modes")
    print("=" * 110)

    cdr_results = {}
    loss_results = {}
    conf_results = {}

    for name, info in CHECKPOINTS.items():
        ckpt_path = info['path']
        if not os.path.exists(ckpt_path):
            print(f"\n[SKIP] {name}: checkpoint not found")
            continue

        print(f"\n{'鈹€' * 80}")
        print(f"[{name}] {info['desc']}")

        # --- Co-Design Evaluation ---
        if info['codessign'] and not args.skip_codessign:
            try:
                cdr_summary, loss_avg = evaluate_codessign_val(ckpt_path, args.device)
                cdr_results[name] = cdr_summary
                loss_results[name] = loss_avg

                all_aar = cdr_summary.get('ALL', {}).get('aar', 0)
                all_ppl = cdr_summary.get('ALL', {}).get('ppl', float('nan'))
                print(f"  CDR AAR={all_aar:.1f}%  PPL={all_ppl:.2f}  "
                      f"dist={loss_avg.get('dist', float('nan')):.1f}  "
                      f"ang={loss_avg.get('ang', float('nan')):.3f}")
            except Exception as e:
                print(f"  [CDR ERROR] {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  [SKIP] CDR co-design evaluation")

        # --- Confidence Evaluation ---
        if info['confidence'] and not args.skip_confidence:
            try:
                conf_avg = evaluate_confidence_val(ckpt_path, args.device, args.conf_max_samples)
                conf_results[name] = conf_avg

                print(f"  conf  pLDDT={conf_avg.get('plddt', float('nan')):.4f}  "
                      f"ipTM={conf_avg.get('iptm', float('nan')):.4f}  "
                      f"PAE={conf_avg.get('pae', float('nan')):.4f}  "
                      f"disorder={conf_avg.get('disorder', float('nan')):.4f}")
            except Exception as e:
                print(f"  [CONF ERROR] {e}")
                import traceback
                traceback.print_exc()

    # =========================================================================
    # Print Tables
    # =========================================================================

    # --- Table 1: CDR Co-Design ---
    if cdr_results:
        print("\n" + "=" * 110)
        print("TABLE 1: CDR Co-Design Performance (Phase3 val, 5 complexes)")
        print("        AAR = Amino Acid Recovery (higher=better, unbiased by label_smoothing)")
        print("        PPL = Perplexity (lower=better)")
        print("=" * 110)

        # Per-CDR AAR table
        header = f"{'CDR':<10}"
        for name in cdr_results:
            short = name.split('_', 1)[1][:22]
            header += f" {short:<24}"
        print(header)
        print("-" * 110)

        for cdr in CDR_TYPES:
            row = f"{cdr:<10}"
            for name in cdr_results:
                d = cdr_results[name].get(cdr, {})
                if d:
                    row += f" AAR={d['aar']:5.1f}% PPL={d['ppl']:5.1f}"
                else:
                    row += f" {'--':>24}"
            print(row)

        # Summary row
        print("-" * 110)
        row = f"{'OVERALL':<10}"
        best_aar_name = max(cdr_results, key=lambda n: cdr_results[n].get('ALL', {}).get('aar', 0))
        best_ppl_name = min(cdr_results, key=lambda n: cdr_results[n].get('ALL', {}).get('ppl', float('inf')))
        for name in cdr_results:
            d = cdr_results[name].get('ALL', {})
            if d:
                aar_mark = ' 鈽? if name == best_aar_name else ''
                ppl_mark = ' 鈽? if name == best_ppl_name else ''
                row += f" AAR={d['aar']:5.1f}%{aar_mark} PPL={d['ppl']:5.1f}{ppl_mark}"
            else:
                row += f" {'--':>24}"
        print(row)

    # --- Table 2: Structure Loss ---
    if loss_results:
        print("\n" + "=" * 110)
        print("TABLE 2: Structure Co-Design Loss (Phase3 val, lower=better)")
        print("=" * 110)

        metrics = ['overall', 'seq', 'dist', 'ang']
        metric_names = {'overall': 'Overall Loss', 'seq': 'Seq CE', 'dist': 'Dist Huber', 'ang': 'Ang Huber'}

        header = f"{'Metric':<14}"
        for name in loss_results:
            short = name.split('_', 1)[1][:22]
            header += f" {short:<24}"
        print(header)
        print("-" * 110)

        best = {}
        for m in metrics:
            vals = {k: v.get(m, float('inf')) for k, v in loss_results.items()}
            best[m] = min(vals, key=vals.get)

        for m in metrics:
            row = f"{metric_names.get(m, m):<14}"
            for name in loss_results:
                val = loss_results[name].get(m, float('nan'))
                marker = ' 鈽? if name == best.get(m) else ''
                row += f" {val:>22.4f}{marker} "
            print(row)

    # --- Table 3: Confidence Prediction ---
    if conf_results:
        print("\n" + "=" * 110)
        print("TABLE 3: Confidence Prediction (confidence_v11 val, lower=better)")
        print("        Overall = 2.0*pLDDT + 2.5*ipTM + 0.3*PAE  (disorder excluded)")
        print("        Disorder only shown for checkpoints that have the head")
        print("=" * 110)

        # Compute fair Overall from shared metrics only (pLDDT, ipTM, PAE)
        shared_weights = {'plddt': 2.0, 'iptm': 2.5, 'pae': 0.3}

        conf_metrics = ['plddt', 'iptm', 'pae', 'disorder']
        conf_names = {'plddt': 'pLDDT MAE', 'iptm': 'ipTM MAE',
                       'pae': 'PAE MAE', 'disorder': 'Disorder BCE'}

        header = f"{'Metric':<14}"
        for name in conf_results:
            short = name.split('_', 1)[1][:22]
            header += f" {short:<24}"
        print(header)
        print("-" * 110)

        # Overall (fair, shared only)
        row = f"{'Overall':<14}"
        best_overall = float('inf')
        best_name = None
        for name in conf_results:
            overall = sum(shared_weights[k] * conf_results[name].get(k, 0)
                         for k in shared_weights)
            if overall < best_overall:
                best_overall = overall
                best_name = name
            row += f" {overall:>22.4f}"
            if name == best_name:
                row += " 鈽?
            else:
                row += "  "
        print(row)

        best = {}
        for m in conf_metrics:
            vals = {k: v.get(m, float('inf')) for k, v in conf_results.items()
                    if m in v and not np.isnan(v.get(m, float('nan')))}
            if vals:
                best[m] = min(vals, key=vals.get)

        for m in conf_metrics:
            # Only show row if at least one checkpoint has this metric
            available = [name for name in conf_results
                         if m in conf_results[name] and not np.isnan(conf_results[name][m])]
            if not available:
                continue
            row = f"{conf_names.get(m, m):<14}"
            for name in conf_results:
                val = conf_results[name].get(m, float('nan'))
                if np.isnan(val):
                    row += f" {'--':>22}  "
                else:
                    marker = ' 鈽? if name == best.get(m) else ''
                    row += f" {val:>22.4f}{marker} "
            print(row)

    # --- Save ---
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outpath = os.path.join(args.output_dir, f'ablation_full_{timestamp}.json')

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    with open(outpath, 'w') as f:
        json.dump(convert({
            'timestamp': timestamp,
            'cdr_summaries': cdr_results,
            'loss_results': loss_results,
            'confidence_results': conf_results,
        }), f, indent=2)

    print(f"\nResults saved to: {outpath}")
    print("Done.")


if __name__ == '__main__':
    main()
