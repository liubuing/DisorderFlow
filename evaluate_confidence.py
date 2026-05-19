#!/usr/bin/env python
"""Evaluate fine-tuned BFN confidence heads against AF2 ground truth.

Computes Pearson r, Spearman ρ, and MAE for pLDDT, ipTM, and PAE predictions
on the validation set. Outputs per-protein details to JSON.
"""
import sys, os, json, pickle, argparse
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn.functional as F

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_DIR = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate BFN confidence head fine-tuning')
    p.add_argument('--checkpoint', required=True, help='Path to fine-tuned checkpoint .pt')
    p.add_argument('--db_path', default=None,
                   help='Path to validation LMDB (default: data/confidence_dataset/confidence_val.lmdb)')
    p.add_argument('--device', default='cpu', help='Device (cpu/cuda)')
    p.add_argument('--num_recycles', type=int, default=1, help='BFN recycle rounds (default 1)')
    return p.parse_args()


def load_lmdb_entries(db_path):
    entries = []
    if os.path.isdir(db_path) and not os.path.exists(os.path.join(db_path, 'data.mdb')):
        meta_path = os.path.join(db_path, 'meta.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            n_entries = meta['n_entries']
        else:
            n_entries = len([f for f in os.listdir(db_path) if f.endswith('.pkl')])
        for i in range(n_entries):
            with open(os.path.join(db_path, f'{i:08d}.pkl'), 'rb') as f:
                entries.append(pickle.load(f))
    else:
        import lmdb
        env = lmdb.open(db_path, readonly=True, lock=False)
        with env.begin() as txn:
            n_entries = pickle.loads(txn.get(b'__len__'))
            for i in range(n_entries):
                entries.append(pickle.loads(txn.get(f'{i:08d}'.encode())))
        env.close()
    return entries


def load_model(checkpoint_path, device):
    from antibodydesignbfn.models import get_model
    from antibodydesignbfn.utils.misc import load_config as _lc

    config_path = PROJECT_DIR / 'configs' / 'train' / 'bfn_confidence_finetune.yml'
    config, _ = _lc(str(config_path))

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mc = ckpt['config'].model
    if hasattr(ckpt['config'], 'train') and hasattr(ckpt['config'].train, 'loss_weights'):
        mc['loss_weight'] = dict(ckpt['config'].train.loss_weights)

    model = get_model(mc).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def compute_correlations(pred, true, mask=None):
    """Compute Pearson r, Spearman ρ, MAE between pred and true tensors."""
    if mask is not None:
        pred = pred[mask.bool()]
        true = true[mask.bool()]
    else:
        pred = pred.flatten()
        true = true.flatten()

    if pred.numel() < 3:
        return {'pearson_r': None, 'spearman_rho': None, 'mae': None, 'n_points': pred.numel()}

    pred_np = pred.cpu().float().numpy()
    true_np = true.cpu().float().numpy()
    mae = np.mean(np.abs(pred_np - true_np))

    # Guard against constant-value inputs that cause pearsonr/spearmanr to return NaN
    r, rho = float('nan'), float('nan')
    if np.std(pred_np) > 1e-12 and np.std(true_np) > 1e-12:
        r, _ = pearsonr(pred_np, true_np)
        rho, _ = spearmanr(pred_np, true_np)

    return {'pearson_r': float(r), 'spearman_rho': float(rho), 'mae': float(mae), 'n_points': int(pred.numel())}


def evaluate(model, entries, device, num_recycles=1):
    """Run evaluation — run BFN sample() to get confidence predictions, compare against AF2."""
    from antibodydesignbfn.utils.train import recursive_to
    from antibodydesignbfn.utils.data import PaddingCollate

    per_protein = []
    all_pred_plddt = []
    all_true_plddt = []
    all_pred_iptm = []
    all_true_iptm = []

    gc = 'cuda' in str(device)

    for i, entry in enumerate(entries):
        batch = entry['batch']
        pdb_id = entry.get('pdb_id', f'entry_{i}')
        seq = entry.get('sequence', '')
        n_native = len(seq)

        # Attach AF2 teacher targets (kept aside for comparison)
        af2_plddt = entry['af2_plddt']
        af2_iptm = entry['af2_iptm']
        af2_pae = entry.get('af2_pae_matrix', None)

        # Collate to batch of 1
        try:
            batch_c = PaddingCollate()([batch])
            batch_c = recursive_to(batch_c, device)
        except Exception as e:
            print(f'  [{i+1}/{len(entries)}] {pdb_id}: COLLATE ERROR - {e}')
            continue

        N, L_padded = batch_c['aa'].shape

        # Set all residues as context (no generation) so encoder sees full structure.
        # The BFN receiver still runs through the diffusion loop and produces
        # confidence predictions for every residue.
        batch_c['generate_flag'] = torch.zeros_like(batch_c['mask'])

        # Run BFN sample through the WRAPPER (not core directly) —
        # the wrapper computes pair_feat via PairEmbedding before calling core.
        with torch.no_grad():
            try:
                result = model.sample(batch_c, sample_opt={
                    'deterministic': True,
                    'num_recycles': num_recycles,
                })
            except Exception as e:
                print(f'  [{i+1}/{len(entries)}] {pdb_id}: SAMPLE ERROR - {e}')
                import traceback; traceback.print_exc()
                continue

        pred_plddt = result['plddt'][0, :n_native]  # trim padding, (L_native,)
        pred_iptm = result['iptm'][0]                # scalar
        pred_pae = result['pae'][0, :n_native, :n_native]  # (L_native, L_native)

        # AF2 ground truth
        true_plddt = torch.as_tensor(af2_plddt, dtype=torch.float32, device=device)
        if true_plddt.dim() == 0:
            true_plddt = true_plddt.unsqueeze(0)
        true_plddt = true_plddt[:n_native]

        true_iptm = torch.as_tensor(float(af2_iptm), dtype=torch.float32, device=device)

        # ---- Per-residue pLDDT ----
        plddt_metrics = compute_correlations(pred_plddt, true_plddt)

        # ---- Per-protein ipTM ----
        true_iptm_val = true_iptm.item() if true_iptm.dim() == 0 else true_iptm[0].item()
        pred_iptm_val = pred_iptm.item() if pred_iptm.dim() == 0 else pred_iptm[0].item()
        iptm_abs_error = abs(pred_iptm_val - true_iptm_val)

        # ---- PAE matrix ----
        if af2_pae is not None:
            true_pae = torch.as_tensor(af2_pae, dtype=torch.float32, device=device)
            true_pae = true_pae[:n_native, :n_native]
            # BFN PAE is sigmoid [0,1], AF2 target is /31.0 [0,1]
            pae_metrics = compute_correlations(pred_pae, true_pae / 31.0)
        else:
            pae_metrics = {'pearson_r': None, 'spearman_rho': None, 'mae': None, 'n_points': 0}

        # Accumulate global arrays
        all_pred_plddt.append(pred_plddt.cpu().numpy())
        all_true_plddt.append(true_plddt.cpu().numpy())
        all_pred_iptm.append(pred_iptm_val)
        all_true_iptm.append(true_iptm_val)

        res = {
            'pdb_id': pdb_id,
            'length': n_native,
            'plddt_pearson_r': plddt_metrics['pearson_r'],
            'plddt_spearman_rho': plddt_metrics['spearman_rho'],
            'plddt_mae': plddt_metrics['mae'],
            'iptm_abs_error': iptm_abs_error,
            'bfn_iptm': pred_iptm_val,
            'af2_iptm': true_iptm_val,
            'pae_pearson_r': pae_metrics['pearson_r'],
            'pae_spearman_rho': pae_metrics['spearman_rho'],
            'pae_mae': pae_metrics['mae'],
        }
        per_protein.append(res)

        print(f'  [{i+1}/{len(entries)}] {pdb_id} (L={n_native}): '
              f'pLDDT r={plddt_metrics["pearson_r"]:.3f} ρ={plddt_metrics["spearman_rho"]:.3f} MAE={plddt_metrics["mae"]:.4f} | '
              f'ipTM err={iptm_abs_error:.4f} | '
              f'PAE r={pae_metrics["pearson_r"]:.3f}' if pae_metrics['pearson_r'] is not None else 'PAE r=N/A')

    return per_protein, all_pred_plddt, all_true_plddt, all_pred_iptm, all_true_iptm


def main():
    args = parse_args()

    db_path = args.db_path or str(PROJECT_DIR / 'data' / 'confidence_dataset' / 'confidence_val.lmdb')

    print('=' * 60)
    print('  BFN Confidence Head Evaluation')
    print('=' * 60)
    print(f'  Checkpoint:    {args.checkpoint}')
    print(f'  Database:      {db_path}')
    print(f'  Device:        {args.device}')
    print(f'  Num recycles:  {args.num_recycles}')
    print()

    if not os.path.exists(args.checkpoint):
        print(f'ERROR: Checkpoint not found: {args.checkpoint}')
        sys.exit(1)

    if not os.path.exists(db_path):
        print(f'ERROR: Database not found: {db_path}')
        sys.exit(1)

    # Load data
    entries = load_lmdb_entries(db_path)
    print(f'Loaded {len(entries)} validation entries\n')

    # Load model
    print('Loading model...')
    model = load_model(args.checkpoint, args.device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Parameters: {trainable:,} trainable / {total_params:,} total\n')

    # Evaluate
    per_protein, all_pred_plddt, all_true_plddt, all_pred_iptm, all_true_iptm = evaluate(
        model, entries, args.device, num_recycles=args.num_recycles
    )

    # ---- Global Summary ----
    print(f'\n{"=" * 60}')
    print(f'  Global Summary ({len(per_protein)} proteins)')
    print(f'{"=" * 60}')

    if all_pred_plddt:
        cat_pred = np.concatenate(all_pred_plddt)
        cat_true = np.concatenate(all_true_plddt)
        pr, _ = pearsonr(cat_pred, cat_true)
        sr, _ = spearmanr(cat_pred, cat_true)
        mae = np.mean(np.abs(cat_pred - cat_true))
        print(f'  pLDDT (per-residue, N={len(cat_pred)}):')
        print(f'    Pearson r   = {pr:.4f}')
        print(f'    Spearman ρ  = {sr:.4f}')
        print(f'    MAE         = {mae:.4f}')

    if all_pred_iptm:
        pred_arr = np.array(all_pred_iptm)
        true_arr = np.array(all_true_iptm)
        pr_i, _ = pearsonr(pred_arr, true_arr)
        sr_i, _ = spearmanr(pred_arr, true_arr)
        mae_i = np.mean(np.abs(pred_arr - true_arr))
        print(f'\n  ipTM (per-protein, N={len(pred_arr)}):')
        print(f'    Pearson r   = {pr_i:.4f}')
        print(f'    Spearman ρ  = {sr_i:.4f}')
        print(f'    MAE         = {mae_i:.4f}')

    # Per-protein pLDDT summary
    valid_plddt = [r for r in per_protein if r['plddt_pearson_r'] is not None]
    if valid_plddt:
        mean_r = np.mean([r['plddt_pearson_r'] for r in valid_plddt])
        mean_rho = np.mean([r['plddt_spearman_rho'] for r in valid_plddt])
        mean_mae = np.mean([r['plddt_mae'] for r in valid_plddt])
        print(f'\n  Per-protein pLDDT avg:')
        print(f'    Mean Pearson r  = {mean_r:.4f}')
        print(f'    Mean Spearman ρ = {mean_rho:.4f}')
        print(f'    Mean MAE        = {mean_mae:.4f}')

    # ---- Save Results ----
    out = {
        'checkpoint': args.checkpoint,
        'db_path': db_path,
        'n_proteins': len(per_protein),
        'global': {
            'plddt_pearson_r': float(pr) if all_pred_plddt else None,
            'plddt_spearman_rho': float(sr) if all_pred_plddt else None,
            'plddt_mae': float(mae) if all_pred_plddt else None,
            'iptm_pearson_r': float(pr_i) if all_pred_iptm else None,
            'iptm_spearman_rho': float(sr_i) if all_pred_iptm else None,
            'iptm_mae': float(mae_i) if all_pred_iptm else None,
        },
        'per_protein': per_protein,
    }

    out_path = PROJECT_DIR / 'confidence_eval_results.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nResults saved to: {out_path}')


if __name__ == '__main__':
    main()
