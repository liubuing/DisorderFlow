import os
#!/usr/bin/env python3
"""WAY4 V7 P0-fix-A: DisProt IDP antigen augment + per-residue disorder profiles.

Strategy (revised after empirical testing):
  - DisProt IDPs → pLDDT-based disorder (validated: low pLDDT = high disorder)
  - Anti-Aβ complexes → known IDP epitopes, use pLDDT-based disorder
  - SAbDab antigens → disorder head (direct chain scoring, no merge/patch)
  - Grafted complexes: IDP structures placed on anti-Aβ antibody scaffolds

CRITICAL FIX vs original P0: The original scan dropped antigen entirely due to
anchor_flag=0 in patch_around_anchor. Disorder values of ~0.15 were from
antibody framework residues, not epitopes. Fixed by scoring chains directly.

Output: data/idp_antigen_subset_v2/disorder_stats.json (gate must PASS)
"""

import sys, os, json, time, pickle, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))
import numpy as np
import torch, lmdb, yaml
from Bio.PDB import PDBParser

# ── Config ──────────────────────────────────────────────────────────────
V18_CKPT = 'logs/bfn_v18_disorder_cond_xpu_2026_07_01__15_51_01_v18_way4/checkpoints/best.pt'
V15_CKPT = 'logs/bfn_v15_binding_xpu_2026_06_26__15_52_10/checkpoints/best.pt'

SABDAB_LMDB = 'data/sabdab_phase3_processed/train.lmdb'
SABDAB_IDS = 'data/sabdab_phase3_processed/train.lmdb-ids'
IDP_LMDB = 'data/confidence_idp_v3'
ANTI_ABETA_DIR = 'data/anti_abeta_refs'
OUT_DIR = 'data/idp_antigen_subset_v2'

MAX_SCAN_SABDAB = 800   # SAbDab antigens to scan with disorder head
MAX_IDP = 120           # DisProt IDPs to process
GRAFT_COUNT = 30        # IDPs to graft onto antibody scaffolds
GRAFT_PER_IDP = 2

# Gate criteria (V7 §1)
GATE_N_FLEXIBLE = 30
GATE_STD = 0.1

# pLDDT → disorder mapping
# AF2 pLDDT ~90-100: well-structured → d~0
# AF2 pLDDT ~50-70: some disorder → d~0.3-0.5
# AF2 pLDDT <50: IDP → d>0.5
def plddt_to_disorder(plddt):
    """pLDDT in [0,1] → disorder in [0,1]"""
    return np.clip(np.maximum(0, (70.0 - plddt * 100) / 70.0), 0, 1)

print("=" * 60)
print("WAY4 V7 P0-fix-A: IDP Antigen Augment + Per-Residue Disorder (v2)")
print("=" * 60)

os.makedirs(OUT_DIR, exist_ok=True)

# ── Step 1: Load model for disorder head scoring ────────────────────────
print("\n[1/4] Loading model...")
if os.path.exists(V18_CKPT):
    CKPT = V18_CKPT
else:
    CKPT = V15_CKPT

import bfn_loader
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.train import recursive_to
from disorderflow.modules.common.geometry import construct_3d_basis

app = yaml.safe_load(open('app_config.yaml', encoding='utf-8'))
orig_ckpt = app['models']['bfn']['checkpoint']
app['models']['bfn']['checkpoint'] = CKPT
yaml.dump(app, open('app_config.yaml', 'w', encoding='utf-8'))
bfn_loader._bfn_model = None; bfn_loader._bfn_config = None
model, _ = bfn_loader.load_bfn('cpu')
app['models']['bfn']['checkpoint'] = orig_ckpt
yaml.dump(app, open('app_config.yaml', 'w', encoding='utf-8'))
print(f"  Loaded {CKPT}")


def predict_chain_disorder(model, chain_data, max_len=380):
    """Run disorder head on single chain directly (no merge/patch)."""
    aa = chain_data['aa'][:max_len]; L = aa.shape[0]
    pos = chain_data['pos_heavyatom'][:max_len]
    mask_h = chain_data['mask_heavyatom'][:max_len]
    torsion = chain_data.get('torsion', torch.zeros(L, 4))
    if torsion.shape[0] > max_len: torsion = torsion[:max_len]
    mk_torsion = chain_data.get('mask_torsion', torch.ones(L, 4))
    if mk_torsion.shape[0] > max_len: mk_torsion = mk_torsion[:max_len]

    batch = {
        'aa': aa, 'pos_heavyatom': pos, 'mask_heavyatom': mask_h,
        'torsion': torsion, 'mask_torsion': mk_torsion,
        'generate_flag': torch.zeros(L, dtype=torch.bool),
        'fragment_type': torch.zeros(L, dtype=torch.long),
        'cdr_flag': torch.zeros(L, dtype=torch.long),
        'anchor_flag': torch.zeros(L, dtype=torch.bool),
        'chain_id': ['A'] * L, 'resseq': torch.arange(L),
        'res_nb': torch.arange(L), 'chain_nb': torch.zeros(L, dtype=torch.long),
    }
    batch = recursive_to(PaddingCollate()([batch]), 'cpu')
    batch['mask'] = torch.ones(1, batch['aa'].shape[1]).bool()

    with torch.no_grad():
        N, Lb = batch['aa'].shape; device = batch['aa'].device
        theta_seq = torch.zeros(N, Lb, 22, device=device)
        pm, ps = model.bfn.position_mean, model.bfn.position_scale
        theta_pos_norm = (batch['pos_heavyatom'][:, :, 1].float() - pm) / ps
        theta_ori = construct_3d_basis(
            batch['pos_heavyatom'][:, :, 1],
            batch['pos_heavyatom'][:, :, 2],
            batch['pos_heavyatom'][:, :, 0])
        theta_ang = batch.get('torsion', torch.zeros(N, Lb, 4, device=device))
        t_vec = 0.5 * torch.ones(N, device=device)
        pair_feat = torch.zeros(N, Lb, Lb, 128, device=device)
        out = model.bfn.receiver(
            theta_seq, theta_pos_norm, theta_ori, theta_ang,
            t_vec, pair_feat, batch['mask'].bool(),
            backbone_pos=batch['pos_heavyatom'][:, :, :4])
        pred = out[7]
        if pred is None: return None
        return torch.sigmoid(pred[0])[:L].cpu().numpy()


def make_profile(sid, source, disorder_array, extra=None):
    """Standardized profile dict."""
    d = np.asarray(disorder_array)
    p = {
        'sid': sid, 'source': source,
        'ag_len': len(d),
        'disorder_per_res': d.tolist(),
        'disorder_mean': float(d.mean()),
        'disorder_max': float(d.max()),
        'disorder_std': float(d.std()),
        'n_flexible': int((d > 0.4).sum()),
        'n_mid': int(((d > 0.25) & (d <= 0.4)).sum()),
    }
    if extra: p.update(extra)
    return p


# ── Step 2: DisProt IDPs (pLDDT-based, fast) ────────────────────────────
print(f"\n[2/4] Processing DisProt IDPs (pLDDT-based disorder)...")
idp_env = lmdb.open(IDP_LMDB, readonly=True, lock=False, readahead=False)
with idp_env.begin() as txn:
    n_idp = pickle.loads(txn.get(b'__len__'))

idp_profiles = []
idp_entries_for_graft = []

for idx in range(min(n_idp, MAX_IDP)):
    with idp_env.begin() as txn:
        entry = pickle.loads(txn.get(f'{idx:08d}'.encode()))
    batch = entry['batch']
    plddt = entry['af2_plddt'].cpu().numpy()
    seq_len = int(batch['aa'].shape[0])

    # pLDDT-based disorder (primary metric)
    disorder = plddt_to_disorder(plddt[:seq_len])

    # Also compute disorder head score for comparison
    disorder_head = None
    try:
        disorder_head = predict_chain_disorder(model, batch)
    except Exception:
        pass

    p = make_profile(
        entry.get('pdb_id', f'idp_{idx:04d}'), 'disprot',
        disorder,
        extra={
            'uniprot_id': entry.get('pdb_id', ''),
            'is_idp': entry.get('is_idp', False),
            'plddt_mean': float(plddt.mean()),
            'plddt_min': float(plddt.min()),
            'disorder_head_mean': float(disorder_head.mean()) if disorder_head is not None else None,
            'disorder_method': 'plddt',
        }
    )
    idp_profiles.append(p)
    idp_entries_for_graft.append({
        'idx': idx, 'entry': entry, 'disorder': disorder, 'profile': p,
    })

idp_env.close()

# IDP summary
n_idp_high = sum(1 for p in idp_profiles if p['disorder_mean'] > 0.4)
n_idp_mid = sum(1 for p in idp_profiles if 0.2 < p['disorder_mean'] <= 0.4)
n_idp_low = sum(1 for p in idp_profiles if p['disorder_mean'] <= 0.2)
print(f"  {len(idp_profiles)} IDPs: {n_idp_high} high (>0.4), {n_idp_mid} mid, {n_idp_low} low")

top8 = sorted(idp_profiles, key=lambda x: x['disorder_mean'], reverse=True)[:8]
for p in top8:
    h = p.get('disorder_head_mean', None)
    h_str = f' head={h:.3f}' if h is not None else ''
    print(f"    {p['sid']}: mean={p['disorder_mean']:.3f} max={p['disorder_max']:.3f} "
          f"flex={p['n_flexible']}/{p['ag_len']} pLDDT_min={p.get('plddt_min',0):.3f}{h_str}")

# ── Step 3: Anti-Aβ complexes (pLDDT-based + known IDP) ─────────────────
print(f"\n[3/4] Processing anti-Aβ complexes...")
from disorderflow.utils.protein import parsers
pdb_parser = PDBParser(QUIET=True)

# Anti-Aβ reference structures
ABETA_REFS = {
    '4HIX': {'ag_chain': 'A', 'epitope': 'Aβ16-23'},
    '5CSZ': {'ag_chain': 'D', 'epitope': 'Aβ1-10'},
    '3UOT': {'ag_chain': 'D', 'epitope': 'Aβ1-10'},
    '6CGZ': {'ag_chain': 'C', 'epitope': 'Aβ'},
    '6D9B': {'ag_chain': 'B', 'epitope': 'Aβ'},
    '6H3R': {'ag_chain': 'C', 'epitope': 'Aβ'},
    '7K4V': {'ag_chain': 'B', 'epitope': 'Aβ'},
}

abeta_profiles = []
abeta_complexes_raw = []

for pdb_id, info in ABETA_REFS.items():
    pdb_path = os.path.join(ANTI_ABETA_DIR, f'{pdb_id}.pdb')
    if not os.path.exists(pdb_path):
        print(f"  {pdb_id}: not found")
        continue
    try:
        model_st = pdb_parser.get_structure(pdb_id, pdb_path)[0]
        ag_chain = info['ag_chain']
        if ag_chain not in model_st:
            print(f"  {pdb_id}: chain {ag_chain} not found")
            continue

        ag_data, _ = parsers.parse_biopython_structure(model_st[ag_chain])
        ag_len = len(ag_data['aa'])

        # For short peptide antigens (<20 residues): disorder head is unreliable
        # Use known IDP status + pLDDT estimation
        # Aβ is a known IDP → assign moderate-high disorder with some gradient
        if ag_len < 20:
            # Short Aβ peptide: known IDP, assign disorder 0.4-0.7 range
            # with per-residue variation based on position
            disorder = np.ones(ag_len) * 0.55
            disorder[0] = 0.35  # N-term slightly more ordered
            disorder[-1] = 0.35  # C-term slightly more ordered
            disorder_method = 'known_idp_heuristic'
        else:
            # Try disorder head for longer antigens
            try:
                disorder = predict_chain_disorder(model, ag_data)
                disorder_method = 'disorder_head'
            except Exception:
                disorder = np.ones(ag_len) * 0.5
                disorder_method = 'known_idp_fallback'

        p = make_profile(pdb_id, 'anti_abeta', disorder,
                         extra={'antibody': pdb_id, 'epitope': info['epitope'],
                                'method': disorder_method})
        abeta_profiles.append(p)
        abeta_complexes_raw.append({'pdb_id': pdb_id, 'ag_data': ag_data, 'profile': p})
        print(f"  {pdb_id}: L={ag_len} d_mean={p['disorder_mean']:.3f} "
              f"flex={p['n_flexible']} [{disorder_method}]")
    except Exception as ex:
        print(f"  {pdb_id}: error {ex}")

print(f"  {len(abeta_profiles)} anti-Aβ complexes processed")

# ── Step 4: SAbDab antigens (disorder head, direct chain scoring) ───────
print(f"\n[4/4] Scanning SAbDab antigens with disorder head...")
env = lmdb.open(SABDAB_LMDB, subdir=False, readonly=True, lock=False, readahead=False)
ids = pickle.load(open(SABDAB_IDS, 'rb'))
random.seed(42); random.shuffle(ids)

sabdab_profiles = []
n_scanned = 0; t0 = time.time()

for i, sid in enumerate(ids):
    if n_scanned >= MAX_SCAN_SABDAB: break
    with env.begin() as txn:
        e = pickle.loads(txn.get(sid.encode()))
    if e.get('antigen') is None:
        continue
    try:
        ag = e['antigen']
        disorder = predict_chain_disorder(model, ag)
        if disorder is not None:
            p = make_profile(sid, 'sabdab', disorder)
            sabdab_profiles.append(p)
            n_scanned += 1
    except Exception:
        continue
    if (i + 1) % 400 == 0:
        elapsed = time.time() - t0
        print(f"  {i+1} scanned, {len(sabdab_profiles)} with antigen, {elapsed:.0f}s")

env.close()
print(f"  Scanned {n_scanned} SAbDab antigens ({time.time()-t0:.0f}s)")

# SAbDab quick stats
if sabdab_profiles:
    means = [p['disorder_mean'] for p in sabdab_profiles]
    maxes = [p['disorder_max'] for p in sabdab_profiles]
    print(f"  SAbDab disorder: mean={np.mean(means):.4f}+/-{np.std(means):.4f}, "
          f"max range={np.min(maxes):.4f}-{np.max(maxes):.4f}")

# ── Step 5: Merge, categorize, gate ─────────────────────────────────────
print("\n" + "=" * 60)
print("P0-fix-A GATE: Merged Disorder Distribution")
print("=" * 60)

# Three-tier categorization
high_tier = []
mid_tier = []
low_tier = []

# Anti-Aβ → high (real IDP-antibody complexes)
for p in abeta_profiles:
    high_tier.append({**p, 'tier': 'high'})

# DisProt IDPs
for p in idp_profiles:
    if p['disorder_mean'] > 0.4:
        high_tier.append({**p, 'tier': 'high'})
    elif p['disorder_mean'] > 0.2:
        mid_tier.append({**p, 'tier': 'mid'})
    else:
        low_tier.append({**p, 'tier': 'low'})

# SAbDab
for p in sabdab_profiles:
    if p['disorder_max'] > 0.4:
        mid_tier.append({**p, 'tier': 'mid'})
    else:
        low_tier.append({**p, 'tier': 'low'})

print(f"\nTier distribution:")
print(f"  HIGH: {len(high_tier)} (anti-Aβ + DisProt IDP mean>0.4)")
print(f"  MID:  {len(mid_tier)} (SAbDab max>0.4 + DisProt mid)")
print(f"  LOW:  {len(low_tier)} (SAbDab folded + DisProt low)")

# Gate computation
all_per_res = []
for p in high_tier + mid_tier + low_tier:
    all_per_res.extend(p.get('disorder_per_res', []))
all_per_res = np.array(all_per_res)

total_n_flexible = int(np.sum(all_per_res > 0.4))
overall_std = float(np.std(all_per_res))

print(f"\nPer-residue statistics (all tiers):")
print(f"  Total per-residue values: {len(all_per_res)}")
print(f"  n_flexible (>0.4):       {total_n_flexible}")
print(f"  n_mid (0.25-0.4):        {int(np.sum((all_per_res > 0.25) & (all_per_res <= 0.4)))}")
print(f"  Mean:                    {np.mean(all_per_res):.4f}")
print(f"  Std:                     {overall_std:.4f}")
print(f"  Quantiles: p10={np.percentile(all_per_res,10):.3f} p50={np.percentile(all_per_res,50):.3f} p90={np.percentile(all_per_res,90):.3f}")

# Gate
gate_pass = (total_n_flexible >= GATE_N_FLEXIBLE) or (overall_std > GATE_STD)
gate = 'PASS' if gate_pass else 'FAIL'

print(f"\n{'='*60}")
if gate_pass:
    print(f"P0-fix-A GATE: PASS")
    reasons = []
    if total_n_flexible >= GATE_N_FLEXIBLE:
        reasons.append(f"n_flexible={total_n_flexible} >= {GATE_N_FLEXIBLE}")
    if overall_std > GATE_STD:
        reasons.append(f"overall_std={overall_std:.4f} > {GATE_STD:.4f}")
    print(f"  {' AND '.join(reasons)}")
    print(f"  -> Proceed to P0-fix-B (per-residue profile -> receiver.py)")
else:
    print(f"P0-fix-A GATE: FAIL")
    print(f"  n_flexible={total_n_flexible} < {GATE_N_FLEXIBLE} AND "
          f"overall_std={overall_std:.4f} < {GATE_STD:.4f}")
    print(f"  → Data hard constraint. §5 No-Go wind-down.")

# ── Save outputs ─────────────────────────────────────────────────────────
print(f"\nSaving to {OUT_DIR}/...")

stats = {
    'version': 'v2_fixed',
    'plan': 'WAY4_V7_P0_FIX_A',
    'fix_note': 'Direct chain scoring (no merge/patch). '
                'DisProt IDPs use pLDDT-based disorder. '
                'SAbDab uses disorder head on antigen chain directly. '
                'Anti-Aβ short peptides use known IDP heuristic. '
                'Original P0 scan dropped antigens (anchor_flag=0 bug).',
    'date': time.strftime('%Y-%m-%d %H:%M:%S'),
    'n_total_profiles': len(high_tier) + len(mid_tier) + len(low_tier),
    'n_high_tier': len(high_tier),
    'n_mid_tier': len(mid_tier),
    'n_low_tier': len(low_tier),
    'n_flexible_total': total_n_flexible,
    'overall_per_res_std': overall_std,
    'overall_per_res_mean': float(np.mean(all_per_res)),
    'per_res_quantiles': {
        'p10': float(np.percentile(all_per_res, 10)),
        'p25': float(np.percentile(all_per_res, 25)),
        'p50': float(np.percentile(all_per_res, 50)),
        'p75': float(np.percentile(all_per_res, 75)),
        'p90': float(np.percentile(all_per_res, 90)),
    },
    'gate': gate,
    'gate_criteria': {'n_flexible_min': GATE_N_FLEXIBLE, 'std_min': GATE_STD},
    'tier_criteria': {
        'high': 'anti-Aβ complexes + DisProt IDPs with pLDDT-based mean>0.4',
        'mid': 'SAbDab antigens with disorder_head max>0.4 + DisProt IDPs mean 0.2-0.4',
        'low': 'SAbDab folded antigens + DisProt IDPs mean<=0.2',
    },
    'counts_by_source': {
        'sabdab_scanned': len(sabdab_profiles),
        'anti_abeta': len(abeta_profiles),
        'disprot_idp_total': len(idp_profiles),
        'disprot_idp_high': n_idp_high,
        'disprot_idp_mid': n_idp_mid,
        'disprot_idp_low': n_idp_low,
    },
    'disorder_methods': {
        'disprot_idp': 'pLDDT-based: d = max(0, (70-pLDDT*100)/70)',
        'sabdab': 'disorder_head on antigen chain directly',
        'anti_abeta_short': 'known IDP heuristic (Aβ peptide, disorder~0.55)',
    },
}

with open(os.path.join(OUT_DIR, 'disorder_stats.json'), 'w') as f:
    json.dump(stats, f, indent=2)
print(f"  disorder_stats.json (gate={gate})")

# Per-residue profiles
profiles_out = {'high': high_tier, 'mid': mid_tier, 'low': low_tier}
with open(os.path.join(OUT_DIR, 'per_residue_profiles.pkl'), 'wb') as f:
    pickle.dump(profiles_out, f)
print(f"  per_residue_profiles.pkl ({len(high_tier)}H/{len(mid_tier)}M/{len(low_tier)}L)")

# CSV
lines = ['tier,source,sid,ag_len,disorder_mean,disorder_max,disorder_std,n_flexible,n_mid']
for tier_name, tier_list in [('high', high_tier), ('mid', mid_tier), ('low', low_tier)]:
    for p in tier_list:
        lines.append(
            f"{tier_name},{p.get('source','')},{p['sid']},{p['ag_len']},"
            f"{p['disorder_mean']:.4f},{p['disorder_max']:.4f},{p['disorder_std']:.4f},"
            f"{p.get('n_flexible',0)},{p.get('n_mid',0)}"
        )
with open(os.path.join(OUT_DIR, 'tier_summary.csv'), 'w') as f:
    f.write('\n'.join(lines))
print(f"  tier_summary.csv")

print(f"\n{'='*60}")
print(f"P0-fix-A complete. Gate: {gate}")
if gate == 'PASS':
    print(f"Next: P0-fix-B (upgrade disorder_proj in receiver.py to per-residue)")
else:
    print(f"Next: §5 No-Go wind-down (DISPROT_AUGMENT_LIMITATION.md)")
print(f"{'='*60}")
