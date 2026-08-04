#!/usr/bin/env python3
"""IDP disorder head retraining — balanced, multi-IDP, CPU-capable.

Root cause: V14/V15/V20 training batches never had `disorder_label`,
so the disorder head loss never fired → random-level predictions (AUC 0.47).

This script:
  1. Loads a BFN checkpoint
  2. Freezes all heads except `head_disorder`
  3. Trains on balanced data: 50% SAbDab ordered (label=0) + 50% IDP disordered
  4. IDP labels from: precomputed disorder lookup + high-disorder epitopes
  5. Evaluates AUC-ROC/PR after training

Usage:
  python train_idp_disorder_head.py                    # CPU training
  python train_idp_disorder_head.py --device cuda:0    # GPU
"""

import sys, os, copy, json, time, random

sys.path.insert(0, '.')
sys.path.insert(0, 'modules')

import numpy as np
import torch
import torch.nn.functional as F
import lmdb, pickle, yaml
from easydict import EasyDict
from disorderflow.utils.transforms import get_transform
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.train import recursive_to
from disorderflow.models import get_model
from disorderflow.utils.misc import seed_all

# ── Config ──
CHECKPOINT = os.environ.get(
    'DISORDER_CKPT',
    'logs/bfn_v20_amplify_xpu_2026_07_02__21_32_16_v20_win/checkpoints/best.pt')
DEVICE = 'cpu'
ITERS = 800
LR = 3e-4
BALANCE_RATIO = 0.5
SEED = 42
PRINT_EVERY = 50
DISORDER_LOOKUP = 'data/sabdab_disorder_lookup_experimental.pkl'
SABDAB_LMDB = 'data/sabdab_phase3_processed/train.lmdb'


def _predict_disorder_direct(model, batch):
    """Get disorder prediction by calling receiver directly (cheap, no sampling)."""
    from disorderflow.modules.common.geometry import construct_3d_basis
    device = batch['aa'].device
    N, L = batch['aa'].shape
    theta_seq = torch.zeros(N, L, 22, device=device)
    theta_pos = batch['pos_heavyatom'][:, :, 1].float()
    pos_mean = model.bfn.position_mean
    pos_scale = model.bfn.position_scale
    theta_pos_norm = (theta_pos - pos_mean) / pos_scale
    theta_ori = construct_3d_basis(
        batch['pos_heavyatom'][:, :, 1],
        batch['pos_heavyatom'][:, :, 2],
        batch['pos_heavyatom'][:, :, 0])
    theta_ang = batch.get('torsion', torch.zeros(N, L, 4, device=device))
    t = 0.5 * torch.ones(N, device=device)
    pair_feat = batch.get('pair_feat', torch.zeros(N, L, L, 128, device=device))
    mask_res = batch['mask'].bool()
    backbone_pos = batch['pos_heavyatom'][:, :, :4]
    mask_gen = batch.get('generate_flag', torch.zeros(N, L).bool())
    out = model.bfn.receiver(
        theta_seq, theta_pos_norm, theta_ori, theta_ang, t,
        pair_feat, mask_res, backbone_pos=backbone_pos, mask_gen=mask_gen)
    return out[7]  # pred_disorder at index 7


def load_disorder_lookup():
    if not os.path.exists(DISORDER_LOOKUP):
        print(f"WARNING: disorder lookup not found at {DISORDER_LOOKUP}")
        return {}
    return pickle.load(open(DISORDER_LOOKUP, 'rb'))


def build_ordered_batch(sabdab_env, sabdab_ids, transform):
    """Random SAbDab folded complex → all residues label 0 (ordered)."""
    for _ in range(30):
        sid = random.choice(sabdab_ids)
        try:
            with sabdab_env.begin() as txn:
                data = pickle.loads(txn.get(sid.encode()))
            if data.get('heavy') is None:
                continue
            struct = transform(data)
            batch = recursive_to(PaddingCollate()([struct]), DEVICE)
            n = int(batch['mask'].sum())
            if 50 < n < 500:
                L = batch['aa'].shape[1]
                batch['disorder_label'] = torch.zeros(1, L, device=DEVICE)
                return batch
        except Exception:
            continue
    return None


def build_idp_batch(sabdab_env, sabdab_ids, disorder_lookup, transform,
                    min_disorder=0.3, min_disordered_residues=8):
    """Random high-disorder entry → antigen residues labeled from lookup."""
    for _ in range(50):
        sid = random.choice(sabdab_ids)
        if sid not in disorder_lookup:
            continue
        profile = disorder_lookup[sid]
        if float(np.max(profile)) < min_disorder:
            continue
        n_disordered = int(np.sum(np.asarray(profile) >= min_disorder))
        if n_disordered < min_disordered_residues:
            continue
        try:
            with sabdab_env.begin() as txn:
                data = pickle.loads(txn.get(sid.encode()))
            if data.get('heavy') is None or data.get('antigen') is None:
                continue
            struct = transform(data)
            batch = recursive_to(PaddingCollate()([struct]), DEVICE)
            L = batch['aa'].shape[1]
            ag_len = len(data['antigen']['aa'])
            if ag_len == 0:
                continue
            lab = torch.zeros(L, device=DEVICE)
            cn = batch['chain_nb'][0]
            ag_nb = int(cn[-1].item())
            ag_mask = (cn == ag_nb)
            n_ag = int(ag_mask.sum())
            if n_ag == 0:
                continue
            prof = torch.tensor(profile[:n_ag], dtype=torch.float32, device=DEVICE)
            lab[ag_mask] = prof
            batch['disorder_label'] = lab.unsqueeze(0)
            return batch
        except Exception:
            continue
    return None


def roc_auc(scores, labels):
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    pos_rank_sum = sum(r for r, l in zip(ranks, labels) if l == 1)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def evaluate(model, sabdab_env, sabdab_ids, disorder_lookup, transform,
             n_eval=10):
    """Quick AUC evaluation using direct receiver call (no sampling)."""
    preds, labs = [], []
    model.eval()
    with torch.no_grad():
        for _ in range(n_eval * 2):
            batch = build_ordered_batch(sabdab_env, sabdab_ids, transform)
            if batch is not None:
                try:
                    dis = _predict_disorder_direct(model, batch)
                    if dis is not None:
                        mask = batch['mask'][0].bool()
                        preds.extend(torch.sigmoid(dis[0][mask]).cpu().tolist())
                        labs.extend([0] * mask.sum().item())
                except Exception:
                    pass

            batch = build_idp_batch(sabdab_env, sabdab_ids, disorder_lookup, transform)
            if batch is not None:
                try:
                    dis = _predict_disorder_direct(model, batch)
                    if dis is not None:
                        mask = batch['mask'][0].bool()
                        lbl = batch['disorder_label'][0][mask]
                        preds.extend(torch.sigmoid(dis[0][mask]).cpu().tolist())
                        labs.extend(lbl.cpu().tolist())
                except Exception:
                    pass

            if len(preds) > 2000:
                break

    model.train()
    if len(preds) < 50:
        return {'auc_roc': None, 'n_preds': len(preds)}
    bin_labs = [1 if l >= 0.3 else 0 for l in labs]
    n_pos = sum(bin_labs)
    return {
        'auc_roc': roc_auc(np.array(preds), np.array(bin_labs)),
        'n_preds': len(preds),
        'n_pos': n_pos,
        'pos_rate': n_pos / len(preds) if preds else 0,
        'mean_ordered': float(np.mean([p for p, l in zip(preds, labs) if l < 0.3])),
        'mean_disordered': float(np.mean([p for p, l in zip(preds, labs) if l >= 0.3])),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--checkpoint', default=CHECKPOINT)
    parser.add_argument('--iters', type=int, default=ITERS)
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")
    seed_all(SEED)

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Iterations: {args.iters}")

    ck = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    cfg_d = dict(ck['config'].model) if hasattr(ck['config'], 'model') else ck['config']['model']
    mc = EasyDict(copy.deepcopy(cfg_d))
    lw = (ck['config'].train.loss_weights if hasattr(ck['config'], 'train')
          else ck['config']['train']['loss_weights'])
    mc['loss_weight'] = dict(lw)
    for k in ['seq', 'dist', 'ang', 'plddt', 'iptm', 'pae', 'conf_variance',
              'conf_ranking', 'grouped_margin', 'conf_anticollapse',
              'disorder_rank', 'disorder_align', 'disorder_mismatch_rank']:
        mc['loss_weight'][k] = 0.0
    mc['loss_weight']['disorder'] = 1.0
    mc['loss_weight']['train_recycles'] = 1

    print("Loading model...")
    model = get_model(mc).to(DEVICE)
    model.load_state_dict(ck['model'], strict=False)

    trainable = []
    for name, param in model.named_parameters():
        param.requires_grad = ('head_disorder' in name)
        if param.requires_grad:
            trainable.append(name)
    print(f"Trainable params: {len(trainable)} ({', '.join(trainable[:5])}...)")

    sabdab_env = lmdb.open(SABDAB_LMDB, readonly=True, lock=False, readahead=False, subdir=False)
    sabdab_ids = pickle.load(open(SABDAB_LMDB + '-ids', 'rb'))
    disorder_lookup = load_disorder_lookup()
    print(f"SAbDab entries: {len(sabdab_ids)}, disorder lookup: {len(disorder_lookup)}")

    transform = get_transform([
        {'type': 'mask_multiple_cdrs'},
        {'type': 'merge_chains'},
        {'type': 'patch_around_anchor'},
    ])

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=1e-4)

    print("\nPre-training eval:")
    pre_ev = evaluate(model, sabdab_env, sabdab_ids, disorder_lookup, transform)
    print(f"  AUC-ROC={pre_ev.get('auc_roc', 'N/A')}, n={pre_ev.get('n_preds')}, "
          f"pos_rate={pre_ev.get('pos_rate', 0):.3f}")

    print(f"\nTraining {args.iters} iterations...")
    t0 = time.time()
    losses_hist = []
    n_ordered, n_idp = 0, 0

    for it in range(1, args.iters + 1):
        model.train()
        if random.random() < BALANCE_RATIO:
            batch = build_ordered_batch(sabdab_env, sabdab_ids, transform)
            kind = 'ordered'
        else:
            batch = build_idp_batch(sabdab_env, sabdab_ids, disorder_lookup, transform)
            kind = 'idp'

        if batch is None:
            continue

        losses = model(batch)
        loss = losses.get('disorder') if isinstance(losses, dict) else None
        if loss is None or not torch.isfinite(loss):
            continue

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        losses_hist.append(float(loss))
        if kind == 'ordered':
            n_ordered += 1
        else:
            n_idp += 1

        if it % PRINT_EVERY == 0 or it == 1:
            avg_loss = np.mean(losses_hist[-PRINT_EVERY:]) if losses_hist else float('nan')
            ev = evaluate(model, sabdab_env, sabdab_ids, disorder_lookup, transform, n_eval=5)
            elapsed = time.time() - t0
            print(f"  it{it:5d} loss={avg_loss:.4f} "
                  f"AUC={ev.get('auc_roc', 'N/A')} "
                  f"ord={ev.get('mean_ordered', 0):.3f} "
                  f"dis={ev.get('mean_disordered', 0):.3f} "
                  f"[{elapsed:.0f}s] batches o={n_ordered} i={n_idp}")

    print(f"\nPost-training eval:")
    post_ev = evaluate(model, sabdab_env, sabdab_ids, disorder_lookup, transform, n_eval=15)
    print(f"  AUC-ROC={post_ev.get('auc_roc', 'N/A')}")
    print(f"  Ordered mean: {post_ev.get('mean_ordered', 0):.4f}")
    print(f"  Disordered mean: {post_ev.get('mean_disordered', 0):.4f}")
    sep = post_ev.get('mean_disordered', 0) - post_ev.get('mean_ordered', 0)
    print(f"  Separation Δ: {sep:+.4f}")

    out = 'logs/disorder_head_retrain_cpu.pt'
    os.makedirs('logs', exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'config': ck['config'] if hasattr(ck, 'config') else ck,
        'train_history': {
            'n_iters': args.iters,
            'n_ordered_batches': n_ordered,
            'n_idp_batches': n_idp,
            'pre_eval': pre_ev,
            'post_eval': post_ev,
        }
    }, out)
    print(f"\nSaved → {out}")

    sabdab_env.close()


if __name__ == '__main__':
    main()
