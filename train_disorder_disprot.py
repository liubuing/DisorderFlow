"""Retrain disorder head with DisProt experimental labels from IDP LMDB."""
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
IDP_LMDB = 'data/confidence_idp_unified'
SABDAB_LMDB = 'data/sabdab_phase3_processed/train.lmdb'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ITERS = 500
LR = 1e-4
SEED = 42

seed_all(SEED)

# ── Load model ──
print(f"Device: {DEVICE}")
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
for n, p in model.named_parameters():
    p.requires_grad = 'head_disorder' in n

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
print(f"Trainable: {sum(1 for n,p in model.named_parameters() if p.requires_grad)} params")

# ── Data loaders ──
idp_env = lmdb.open(IDP_LMDB, readonly=True, lock=False, readahead=False, subdir=True)
sabdab_env = lmdb.open(SABDAB_LMDB, readonly=True, lock=False, readahead=False, subdir=False)

# Load IDP entries
idp_entries = []
with idp_env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        if key == b'__len__':
            continue
        try:
            rec = pickle.loads(value)
            mask = rec.get('disorder_mask')
            if mask is not None:
                idp_entries.append(rec)
        except:
            continue
print(f"IDP entries with disorder_mask: {len(idp_entries)}")

sabdab_ids = pickle.load(open(SABDAB_LMDB + '-ids', 'rb'))
random.shuffle(sabdab_ids)

transform_idp = get_transform([
    {'type': 'merge_chains'},
    {'type': 'patch_around_anchor'},
])
transform_sabdab = get_transform([
    {'type': 'mask_multiple_cdrs'},
    {'type': 'merge_chains'},
    {'type': 'patch_around_anchor'},
])

def build_idp_batch():
    for _ in range(30):
        rec = random.choice(idp_entries)
        try:
            struct = transform_idp(rec)
            batch = recursive_to(PaddingCollate()([struct]), DEVICE)
            L = batch['aa'].shape[1]
            mask_t = rec['disorder_mask']
            if hasattr(mask_t, 'numpy'):
                mask_t = mask_t.numpy()
            mask_np = np.array(mask_t, dtype=np.float32)
            if len(mask_np) > L:
                mask_np = mask_np[:L]
            elif len(mask_np) < L:
                mask_np = np.pad(mask_np, (0, L - len(mask_np)))
            # Clamp to [0, 1]
            mask_np = np.clip(mask_np, 0, 1)
            if mask_np.mean() < 0.1:
                continue  # skip mostly-ordered entries
            batch['disorder_label'] = torch.tensor(mask_np, device=DEVICE).unsqueeze(0)
            return batch
        except:
            continue
    return None

def build_ordered_batch():
    for _ in range(30):
        sid = random.choice(sabdab_ids)
        try:
            with sabdab_env.begin() as txn:
                data = pickle.loads(txn.get(sid.encode()))
            if data.get('heavy') is None:
                continue
            struct = transform_sabdab(data)
            batch = recursive_to(PaddingCollate()([struct]), DEVICE)
            n = int(batch['mask'].sum())
            if 50 < n < 500:
                L = batch['aa'].shape[1]
                batch['disorder_label'] = torch.zeros(1, L, device=DEVICE)
                return batch
        except:
            continue
    return None

# ── AUC evaluation ──
def quick_eval(n=10):
    preds, labs = [], []
    model.eval()
    with torch.no_grad():
        for _ in range(n):
            for builder, label_val in [(build_ordered_batch, 0), (build_idp_batch, 1)]:
                batch = builder()
                if batch is None:
                    continue
                try:
                    theta_seq = torch.zeros(1, batch['aa'].shape[1], 22, device=DEVICE)
                    theta_pos = batch['pos_heavyatom'][:, :, 1].float()
                    theta_pos_norm = (theta_pos - model.bfn.position_mean) / model.bfn.position_scale
                    theta_ori = construct_3d_basis(
                        batch['pos_heavyatom'][:,:,1], batch['pos_heavyatom'][:,:,2],
                        batch['pos_heavyatom'][:,:,0])
                    theta_ang = batch.get('torsion', torch.zeros(1, batch['aa'].shape[1], 4, device=DEVICE))
                    t_tensor = 0.5 * torch.ones(1, device=DEVICE)
                    pair_feat = batch.get('pair_feat', torch.zeros(1, batch['aa'].shape[1], batch['aa'].shape[1], 128, device=DEVICE))
                    out = model.bfn.receiver(theta_seq, theta_pos_norm, theta_ori, theta_ang, t_tensor,
                                             pair_feat, batch['mask'].bool(),
                                             backbone_pos=batch['pos_heavyatom'][:, :, :4],
                                             mask_gen=batch.get('generate_flag', torch.zeros(1, batch['aa'].shape[1]).bool()))
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
        return {'auc': None}
    bin_labs = np.array([1 if l >= 0.5 else 0 for l in labs])
    preds_np = np.array(preds)
    n_pos = bin_labs.sum()
    n_neg = len(bin_labs) - n_pos
    if n_pos == 0 or n_neg == 0:
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
    ord_mean = float(np.mean([p for p, l in zip(preds_np, labs) if l < 0.3]))
    dis_mean = float(np.mean([p for p, l in zip(preds_np, labs) if l >= 0.3]))
    return {'auc': auc, 'n': len(preds), 'ord_mean': ord_mean, 'dis_mean': dis_mean, 'sep': dis_mean - ord_mean}

# ── Train ──
print("\nPre-training eval:")
pre = quick_eval(10)
print(f"  AUC={pre.get('auc', 'N/A')} sep={pre.get('sep', 0):+.4f} "
      f"ord={pre.get('ord_mean', 0):.3f} dis={pre.get('dis_mean', 0):.3f}")

print(f"\nTraining {ITERS} iters...")
losses = []
t0 = time.time()
for it in range(1, ITERS + 1):
    batch = build_idp_batch() if random.random() < 0.5 else build_ordered_batch()
    if batch is None:
        continue
    losses_dict = model(batch)
    loss = losses_dict.get('disorder') if isinstance(losses_dict, dict) else None
    if loss is None or not torch.isfinite(loss):
        continue
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    losses.append(float(loss))

    if it % 100 == 0:
        ev = quick_eval(5)
        print(f"  it{it:4d} loss={np.mean(losses[-50:]):.4f} AUC={ev.get('auc', 'N/A')} "
              f"sep={ev.get('sep', 0):+.4f} [{time.time()-t0:.0f}s]")

print("\nPost-training eval:")
post = quick_eval(15)
print(f"  AUC={post.get('auc', 'N/A')} sep={post.get('sep', 0):+.4f} "
      f"ord={post.get('ord_mean', 0):.3f} dis={post.get('dis_mean', 0):.3f}")

out = 'logs/disorder_head_disprot.pt'
os.makedirs('logs', exist_ok=True)
torch.save({'model': model.state_dict(), 'config': ck['config'], 'pre': pre, 'post': post}, out)
print(f"\nSaved: {out}")

idp_env.close()
sabdab_env.close()
