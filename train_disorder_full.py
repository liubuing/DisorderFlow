"""Full BFN receiver fine-tune for IDP disorder prediction.
Fine-tunes the ENTIRE receiver backbone (not just 6-param head) on
DisProt/MobiDB experimental disorder labels from confidence_idp_unified.
"""
import sys, os, random, time, pickle
sys.path.insert(0, '.')
sys.path.insert(0, 'modules')

import numpy as np
import torch, lmdb, yaml
from easydict import EasyDict
from disorderflow.models import get_model
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.train import recursive_to
from disorderflow.utils.transforms import get_transform
from disorderflow.utils.misc import seed_all
from disorderflow.modules.common.geometry import construct_3d_basis

CKPT = 'logs/bfn_v20_amplify_xpu_2026_07_02__21_32_16_v20_win/checkpoints/best.pt'
IDP_LMDB = 'data/confidence_idp_bfn_ready'
SABDAB_LMDB = 'data/sabdab_phase3_processed/train.lmdb'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ITERS = 500
LR = 3e-5  # lower LR for backbone fine-tune
SEED = 42
EVAL_EVERY = 50

seed_all(SEED)
print(f"Device: {DEVICE}")

# ── Load model ──
ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
mc = EasyDict(dict(ck['config'].model))
lw = dict(ck['config'].train.loss_weights)
for k in list(lw.keys()):
    lw[k] = 0.0
lw['disorder'] = 1.0
lw['train_recycles'] = 1
mc['loss_weight'] = lw

model = get_model(mc).to(DEVICE)
model.load_state_dict(ck['model'], strict=False)

# Unfreeze receiver backbone + disorder head
for n, p in model.named_parameters():
    p.requires_grad = ('bfn.receiver' in n and 'head_seq' not in n
                       and 'head_pos' not in n and 'head_ang' not in n
                       and 'head_plddt' not in n and 'head_iptm' not in n
                       and 'head_contact' not in n)

trainable = [n for n, p in model.named_parameters() if p.requires_grad]
print(f"Trainable params: {len(trainable)} ({sum(p.numel() for n,p in model.named_parameters() if p.requires_grad)/1e6:.1f}M)")

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4)

# ── Data ──
idp_env = lmdb.open(IDP_LMDB, readonly=True, lock=False, readahead=False, subdir=True)
sabdab_env = lmdb.open(SABDAB_LMDB, readonly=True, lock=False, readahead=False, subdir=False)

# Load IDP entries (preprocessed, ready for training)
idp_entries = []
with idp_env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        if key == b'__len__':
            continue
        try:
            batch = pickle.loads(value)
            if isinstance(batch, dict) and 'disorder_label' in batch:
                idp_entries.append(batch)
        except:
            continue
print(f"IDP entries ready: {len(idp_entries)}")

sabdab_ids = pickle.load(open(SABDAB_LMDB + '-ids', 'rb'))
random.shuffle(sabdab_ids)

transform_sab = get_transform([{'type': 'mask_multiple_cdrs'}, {'type': 'merge_chains'}, {'type': 'patch_around_anchor'}])

_idp_batch_count = 0

def make_idp_batch():
    for attempt in range(20):
        rec = random.choice(idp_entries)
        try:
            batch = {k: v for k, v in rec.items()}
            # Ensure all tensors are on device
            for k in list(batch.keys()):
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(DEVICE)
            L = batch['aa'].shape[1]
            if L > 500 or L < 20:
                continue
            global _idp_batch_count
            _idp_batch_count += 1
            return batch
        except Exception as e:
            if _idp_batch_count == 0 and attempt < 2:
                print(f"  IDP err[{attempt}]: {str(e)[:80]}", flush=True)
            continue
    return None

def make_ord_batch():
    for _ in range(20):
        sid = random.choice(sabdab_ids)
        try:
            with sabdab_env.begin() as txn:
                data = pickle.loads(txn.get(sid.encode()))
            if data.get('heavy') is None:
                continue
            struct = transform_sab(data)
            batch = recursive_to(PaddingCollate()([struct]), DEVICE)
            n = int(batch['mask'].sum())
            if 50 < n < 500:
                L = batch['aa'].shape[1]
                batch['disorder_label'] = torch.zeros(1, L, device=DEVICE)
                return batch
        except:
            continue
    return None

def quick_eval(n=8):
    preds, labs = [], []
    model.eval()
    with torch.no_grad():
        for _ in range(n):
            for builder, label_val in [(make_ord_batch, 0), (make_idp_batch, 1)]:
                batch = builder()
                if batch is None:
                    continue
                try:
                    dev = batch['aa'].device
                    N, L = batch['aa'].shape
                    theta_seq = torch.zeros(N, L, 22, device=dev)
                    theta_pos = batch['pos_heavyatom'][:, :, 1].float()
                    theta_pos_norm = (theta_pos - model.bfn.position_mean) / model.bfn.position_scale
                    theta_ori = construct_3d_basis(
                        batch['pos_heavyatom'][:, :, 1], batch['pos_heavyatom'][:, :, 2],
                        batch['pos_heavyatom'][:, :, 0])
                    theta_ang = batch.get('torsion', torch.zeros(N, L, 4, device=dev))
                    t_tensor = 0.5 * torch.ones(N, device=dev)
                    pair_feat = batch.get('pair_feat', torch.zeros(N, L, L, 128, device=dev))
                    out = model.bfn.receiver(
                        theta_seq, theta_pos_norm, theta_ori, theta_ang, t_tensor,
                        pair_feat, batch['mask'].bool(),
                        backbone_pos=batch['pos_heavyatom'][:, :, :4],
                        mask_gen=batch.get('generate_flag', torch.zeros(N, L).bool()))
                    dis = out[7]
                    if dis is not None:
                        mask = batch['mask'][0].bool()
                        preds.extend(torch.sigmoid(dis[0][mask]).cpu().tolist())
                        lbl = batch['disorder_label'][0][mask]
                        labs.extend(lbl.cpu().tolist())
                except:
                    pass
    model.train()
    if len(preds) < 20:
        return {'auc': None, 'n': len(preds)}
    preds_np = np.array(preds)
    bin_labs = np.array([1 if l >= 0.5 else 0 for l in labs])
    n_pos, n_neg = bin_labs.sum(), len(bin_labs) - bin_labs.sum()
    if n_pos < 2 or n_neg < 2:
        return {'auc': 0.5, 'n': len(preds)}
    order = sorted(range(len(preds_np)), key=lambda i: preds_np[i])
    ranks = np.zeros(len(preds_np))
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and preds_np[order[j]] == preds_np[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    pos_sum = sum(r for r, l in zip(ranks, bin_labs) if l == 1)
    auc = (pos_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    ord_m = float(np.mean([p for p, l in zip(preds_np, labs) if l < 0.3]))
    dis_m = float(np.mean([p for p, l in zip(preds_np, labs) if l >= 0.3]))
    return {'auc': auc, 'n': len(preds), 'ord_mean': ord_m, 'dis_mean': dis_m, 'sep': dis_m - ord_m}

# ── Pre-train eval ──
print("\nPre-training eval:")
pre = quick_eval(10)
print(f"  AUC={pre.get('auc', 'N/A')} sep={pre.get('sep', 0):+.4f} "
      f"ord={pre.get('ord_mean', 0):.3f} dis={pre.get('dis_mean', 0):.3f} n={pre.get('n', 0)}")

# ── Train ──
print(f"\nFine-tuning {ITERS} iters (full receiver backbone)...")
loss_hist, t0 = [], time.time()
n_ord, n_idp = 0, 0

for it in range(1, ITERS + 1):
    if random.random() < 0.5:
        batch = make_ord_batch()
        kind = 'ord'
    else:
        batch = make_idp_batch()
        kind = 'idp'
    if batch is None:
        continue

    try:
        losses_dict = model(batch)
        loss = losses_dict.get('disorder') if isinstance(losses_dict, dict) else None
    except Exception as e:
        if it <= 5:
            print(f"  model forward err (skip): {str(e)[:80]}", flush=True)
        continue
    if loss is None or not torch.isfinite(loss):
        continue

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()

    loss_hist.append(float(loss))
    if kind == 'ord':
        n_ord += 1
    else:
        n_idp += 1

    if it % EVAL_EVERY == 0:
        ev = quick_eval(5)
        avg_l = np.mean(loss_hist[-20:]) if loss_hist else 0
        print(f"  it{it:4d} loss={avg_l:.5f} AUC={ev.get('auc', 'N/A')} "
              f"sep={ev.get('sep', 0):+.4f} "
              f"ord={ev.get('ord_mean', 0):.3f} dis={ev.get('dis_mean', 0):.3f} "
              f"[{time.time()-t0:.0f}s] batches o={n_ord} i={n_idp}")

# ── Post eval ──
print("\nPost-training eval:")
post = quick_eval(15)
print(f"  AUC={post.get('auc', 'N/A')} sep={post.get('sep', 0):+.4f} "
      f"ord={post.get('ord_mean', 0):.3f} dis={post.get('dis_mean', 0):.3f}")

out_path = 'logs/disorder_receiver_finetune.pt'
os.makedirs('logs', exist_ok=True)
torch.save({'model': model.state_dict(), 'config': ck['config'],
            'pre': pre, 'post': post, 'iters': ITERS}, out_path)
print(f"\nSaved: {out_path}")

idp_env.close()
sabdab_env.close()
