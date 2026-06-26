#!/usr/bin/env python3
"""Minimal disorder-head retrain loop.

Problem (verified, see design-precision-idp-balance-next memory):
  - V14/V15 never had `disorder_label` in the training batch → the disorder
    loss in core.py:331 (`if 'disorder_label' in batch`) never fires → the
    disorder head is an inherited-from-foundation, never-IDP-supervised weight
    that reads the strongly-disordered Aβ42 as nearly ordered (0.28, should be
    >0.5).
  - Wiring it into sample() before fixing this would propagate a wrong signal.

This loop finetunes the V15 disorder head ONLY (seq/dist/ang weight 0 so the
generator/confidence heads don't drift), training it to recognize disorder:
  - Ordered negatives: SAbDab folded crystal complexes → disorder_label = 0.
  - IDP positive: 5IMK+Aβ42 complex → VHH residues label 0, Aβ42 residues
    label = RMSF(x)/maxRMSF clipped to [0,1] (we know Aβ42 RMSF~3.9).

Validation: re-run the disorder probe; Aβ42 antigen disorder prediction should
rise from ~0.28 toward >0.5.
"""
import sys, os, copy, json, time, random
if sys.platform == 'win32':
    import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.'); sys.path.insert(0, 'modules')

import numpy as np, torch, torch.nn.functional as F, lmdb, pickle, re
from easydict import EasyDict
from disorderflow.utils.transforms import get_transform
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.train import recursive_to
from disorderflow.models import get_model
from disorderflow.utils.misc import seed_all

V15 = 'logs/bfn_v15_binding_xpu_2026_06_26__15_52_10/checkpoints/best.pt'
DEVICE = 'cuda'
ITERS = 400
SEED = 42
A42_RMSF_PKL = 'data/abeta_conformations/abeta42_5seed.pkl'
COMPLEX_PDB = None  # built at runtime


def _ab42_disorder_label(n_ab42):
    """Map Aβ42 per-residue RMSF to a [0,1] disorder label."""
    d = pickle.load(open(A42_RMSF_PKL, 'rb'))
    rmsf = np.array(d['rmsf'], dtype=np.float32)  # (42,)
    # normalize: high RMSF → near 1, low → lower. clip.
    lab = np.clip(rmsf / 6.0, 0.0, 1.0)  # max RMSF 5.70 → ~0.95
    if n_ab42 <= len(lab):
        return torch.tensor(lab[:n_ab42], dtype=torch.float32)
    # pad if parser gave more (shouldn't)
    out = torch.zeros(n_ab42, dtype=torch.float32)
    out[:len(lab)] = torch.tensor(lab)
    return out


def build_ordered_batch():
    """A SAbDab folded complex: all valid residues labeled ordered (0)."""
    global _sabdab_env, _sabdab_ids, _tf_sabdab
    for _ in range(20):
        sid = random.choice(_sabdab_ids)
        try:
            with _sabdab_env.begin() as txn:
                data = pickle.loads(txn.get(sid.encode()))
            struct = _tf_sabdab(data)
            batch = recursive_to(PaddingCollate()([struct]), DEVICE)
            n = int(batch['mask'].sum())
            if 50 < n < 400:
                L = batch['aa'].shape[1]
                batch['disorder_label'] = torch.zeros(1, L, device=DEVICE)
                return batch, 'sabdab'
        except Exception:
            continue
    return None, None


def build_ab42_batch():
    """The 5IMK+Aβ42 complex: VHH ordered (0), Aβ42 from RMSF."""
    global _ab42_struct_template
    struct = copy.deepcopy(_ab42_struct_template)
    batch = recursive_to(PaddingCollate()([struct]), DEVICE)
    cn = batch['chain_nb'][0]  # (L,)
    uniq = sorted(set(int(x) for x in cn.tolist()))
    ag_nb = uniq[0]  # antigen (Aβ42) is the first chain in merge order
    L = batch['aa'].shape[1]
    ag_mask = (cn == ag_nb)
    n_ag = int(ag_mask.sum())
    lab = torch.zeros(L, device=DEVICE)
    if n_ag > 0:
        lab[ag_mask] = _ab42_disorder_label(n_ag).to(DEVICE)
    batch['disorder_label'] = lab.unsqueeze(0)  # (1,L)
    return batch, 'ab42'


def main():
    seed_all(SEED)
    ck = torch.load(V15, map_location=DEVICE, weights_only=False)
    mc = EasyDict(copy.deepcopy(dict(ck['config'].model)))
    mc['loss_weight'] = dict(ck['config'].train.loss_weights)
    # Disable everything except disorder for this targeted run. seq/dist/ang=0
    # so the generator + conf heads don't drift; only the disorder head moves.
    for k in ['seq', 'dist', 'ang', 'plddt', 'iptm', 'pae', 'conf_variance',
              'conf_ranking', 'grouped_margin', 'conf_anticollapse']:
        mc['loss_weight'][k] = 0.0
    mc['loss_weight']['disorder'] = 1.0
    mc['loss_weight']['train_recycles'] = 1

    model = get_model(mc).to(DEVICE)
    model.load_state_dict(ck['model'], strict=False)
    model.train()

    # global state for batch builders
    global _sabdab_env, _sabdab_ids, _tf_sabdab, _ab42_struct_template
    _sabdab_env = lmdb.open('data/sabdab_phase3_processed/train.lmdb',
                            readonly=True, lock=False, readahead=False, subdir=False)
    _sabdab_ids = pickle.load(open('data/sabdab_phase3_processed/train.lmdb-ids', 'rb'))
    _sabdab_iter = None
    _tf_sabdab = get_transform([{'type': 'mask_multiple_cdrs'},
                                {'type': 'merge_chains'},
                                {'type': 'patch_around_anchor'}])

    # Pre-build the Aβ42 complex structure template (post-merge dict)
    from antibody_epitope_complex import position_epitope
    from disorderflow.datasets.protein import preprocess_protein_structure
    out_dir = os.path.abspath('oc_validation_results/_complexes'); os.makedirs(out_dir, exist_ok=True)
    res = position_epitope('data/misfolding_targets/5IMK.pdb',
                           'data/misfolding_targets/2NAO_model1_A_1-42.pdb',
                           scaffold_chain='B', epitope_chain='A', distance=18.0,
                           output_dir=out_dir)
    regions = {}
    for cid, spec in re.findall(r'([A-Za-z0-9]+):([0-9,\-\s]+)', 'B:26-33,51-58,97-113'):
        idx = []
        for seg in spec.split(','):
            if '-' in seg:
                a, b = seg.split('-'); idx.extend(range(int(a), int(b) + 1))
            elif seg: idx.append(int(seg))
        regions[cid] = sorted(set(idx))
    raw = preprocess_protein_structure(res['pdb_path'], chain_ids=['A', 'B'])
    _ab42_struct_template = get_transform([
        {'type': 'mask_region', 'regions': regions},
        {'type': 'merge_protein'},
        {'type': 'patch_protein'},
    ])(raw)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-4, weight_decay=1e-4)

    print(f"Training disorder head only. disorder_label: sabdab=0, Aβ42=RMSF/6.")
    t0 = time.time()
    for it in range(1, ITERS + 1):
        # 50/50 mix
        src = random.choice(['sabdab', 'ab42'])
        batch, kind = (build_ab42_batch() if src == 'ab42' else build_ordered_batch())
        if batch is None:
            continue
        with torch.autocast('cuda', dtype=torch.float16):
            losses = model(batch)
        l = losses.get('disorder')
        if l is None or not torch.isfinite(l):
            print(f"  it{it} {kind}: no disorder loss / nan, skip"); continue
        opt.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % 25 == 0 or it == 1:
            # quick Aβ42 disorder check
            with torch.no_grad():
                b2, _ = build_ab42_batch()
                cn = b2['chain_nb'][0]; ag_nb = sorted(set(int(x) for x in cn.tolist()))[0]
                with torch.autocast('cuda', dtype=torch.float16):
                    traj = model.sample(b2, sample_opt={'deterministic': True, 'num_recycles': 1, 'return_disorder': True})
                dis = traj.get('disorder')
                agdis = float(torch.sigmoid(dis[0][cn == ag_nb]).mean()) if dis is not None else float('nan')
            print(f"  it{it:4d} {kind:6s} disorder_loss={float(l):.4f} | Aβ42 pred disorder={agdis:.3f} (target>0.5)  [{time.time()-t0:.0f}s]")
    # save
    out = 'logs/disorder_head_retrain.pt'
    os.makedirs('logs', exist_ok=True)
    torch.save({'model': model.state_dict(), 'config': ck['config']}, out)
    print(f"\nSaved → {out}")
    # final check
    with torch.no_grad():
        b2, _ = build_ab42_batch()
        cn = b2['chain_nb'][0]; ag_nb = sorted(set(int(x) for x in cn.tolist()))[0]
        with torch.autocast('cuda', dtype=torch.float16):
            traj = model.sample(b2, sample_opt={'deterministic': True, 'num_recycles': 1, 'return_disorder': True})
        dis = traj.get('disorder')
        fw = torch.sigmoid(dis[0]).mean().item()
        ag = torch.sigmoid(dis[0][cn == ag_nb]).mean().item()
        print(f"FINAL: framework disorder={fw:.3f} (want ~0) | Aβ42 disorder={ag:.3f} (want >0.5)")


if __name__ == '__main__':
    main()
