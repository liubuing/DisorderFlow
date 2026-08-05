"""Disorder head training via direct receiver call (no BFN encoder).
Bypasses the full model forward which requires SAbDab-specific keys.
"""
import sys, os, random, time, pickle, lmdb, torch
import numpy as np
sys.path.insert(0, '.')
from disorderflow.modules.common.geometry import construct_3d_basis

CKPT = 'logs/bfn_v20_amplify_xpu_2026_07_02__21_32_16_v20_win/checkpoints/best.pt'
IDP_LMDB = 'data/confidence_idp_bfn_ready'
DEVICE = 'cuda'
ITERS = 500
SEED = 42

# ── Load model ──
from easydict import EasyDict
from disorderflow.models import get_model

ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
mc = EasyDict(dict(ck['config'].model))
lw = dict(ck['config'].train.loss_weights)
for k in lw: lw[k] = 0.0
lw['disorder'] = 1.0
mc['loss_weight'] = lw
model = get_model(mc).to(DEVICE)
model.load_state_dict(ck['model'], strict=False)
model.eval()

# Train only disorder head
for n, p in model.named_parameters():
    p.requires_grad = 'head_disorder' in n
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

# ── Load data ──
env = lmdb.open(IDP_LMDB, readonly=True, lock=False, readahead=False, subdir=True)
idp_entries = []
with env.begin() as txn:
    for k, v in txn.cursor():
        if k == b'__len__': continue
        try:
            rec = pickle.loads(v)
            if 'disorder_label' in rec:
                idp_entries.append(rec)
        except: continue
env.close()
random.seed(SEED)
print(f"IDP entries: {len(idp_entries)}")

# ── Training ──
def forward_disorder(batch, model):
    dev = batch['aa'].device
    N, L = batch['aa'].shape
    theta_seq = torch.zeros(N, L, 22, device=dev)
    theta_pos = batch['pos_heavyatom'][:, :, 1].float()
    theta_pos_norm = (theta_pos - model.bfn.position_mean) / model.bfn.position_scale
    theta_ori = construct_3d_basis(
        batch['pos_heavyatom'][:, :, 1], batch['pos_heavyatom'][:, :, 2],
        batch['pos_heavyatom'][:, :, 0])
    theta_ang = torch.zeros(N, L, 4, device=dev)
    t_t = 0.5 * torch.ones(N, device=dev)
    pair_feat = torch.zeros(N, L, L, 128, device=dev)
    mask_res = batch['mask_heavyatom'][:, :, :4].any(dim=2)
    out = model.bfn.receiver(theta_seq, theta_pos_norm, theta_ori, theta_ang,
                              t_t, pair_feat, mask_res, mask_gen=None)
    pred = out[7]
    return torch.sigmoid(pred)

def quick_eval(n=10):
    preds, labs = [], []
    with torch.no_grad():
        for _ in range(n):
            rec = random.choice(idp_entries)
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in rec.items()}
            L = batch['aa'].shape[1]
            mask = batch['mask_heavyatom'][0, :, :4].any(dim=1)
            try:
                pred = forward_disorder(batch, model)[0][mask]
                label = batch['disorder_label'][0][mask]
                preds.extend(pred.cpu().tolist())
                labs.extend(label.cpu().tolist())
            except: continue
    if len(preds) < 20: return {'auc': None}
    p = np.array(preds)
    bl = np.array([1 if l >= 0.5 else 0 for l in labs])
    n_pos, n_neg = bl.sum(), len(bl) - bl.sum()
    if n_pos < 2 or n_neg < 2: return {'auc': 0.5}
    order = sorted(range(len(p)), key=lambda i: p[i])
    ranks = np.zeros(len(p))
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and p[order[j]] == p[order[i]]: j += 1
        for k in range(i, j): ranks[order[k]] = (i + 1 + j) / 2.0
        i = j
    pos_sum = sum(r for r, l in zip(ranks, bl) if l == 1)
    auc = (pos_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return {'auc': auc, 'n': len(preds),
            'ord_mean': float(np.mean([pp for pp, ll in zip(p, labs) if ll < 0.3])),
            'dis_mean': float(np.mean([pp for pp, ll in zip(p, labs) if ll >= 0.3]))}

print("Pre-train eval:")
pre = quick_eval(20)
print(f"  AUC={pre.get('auc',0):.4f} ord={pre.get('ord_mean',0):.3f} dis={pre.get('dis_mean',0):.3f}")

print(f"\nTraining {ITERS} iters...")
loss_hist, t0 = [], time.time()
for it in range(1, ITERS + 1):
    rec = random.choice(idp_entries)
    batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in rec.items()}
    L = batch['aa'].shape[1]
    mask = batch['mask_heavyatom'][0, :, :4].any(dim=1)
    
    model.train()
    pred = forward_disorder(batch, model)[0][mask]
    target = batch['disorder_label'][0][mask]
    
    loss = torch.nn.functional.smooth_l1_loss(pred, target, beta=0.1, reduction='none')
    loss = (loss * (1.0 + 4.0 * target)).mean()  # focal per-element, then mean
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    loss_hist.append(float(loss))
    
    if it % 100 == 0:
        ev = quick_eval(10)
        print(f"  it{it:4d} loss={np.mean(loss_hist[-50:]):.5f} "
              f"AUC={ev.get('auc',0):.4f} ord={ev.get('ord_mean',0):.3f} dis={ev.get('dis_mean',0):.3f} [{time.time()-t0:.0f}s]")

print("\nPost-train eval:")
post = quick_eval(30)
print(f"  AUC={post.get('auc',0):.4f} ord={post.get('ord_mean',0):.3f} dis={post.get('dis_mean',0):.3f}")
print(f"  Sep: {post.get('dis_mean',0)-post.get('ord_mean',0):+.4f}")

os.makedirs('logs', exist_ok=True)
torch.save({'model': model.state_dict(), 'pre': pre, 'post': post}, 'logs/disorder_receiver_v2.pt')
print(f"\nSaved: logs/disorder_receiver_v2.pt")
