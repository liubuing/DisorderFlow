import os
#!/usr/bin/env python3
"""Quick probe: does V17c use antigen signal? Compare vs V15 baseline.

Tests 3 SAbDab complexes, 5 designs each, complex vs fixbb mode.
Free: BFN inference only, no AF2.
Output: recovery rate Δ(complex-fixbb) for each model.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))

import numpy as np, torch, lmdb, pickle, random, yaml, time
from disorderflow.utils.transforms import get_transform
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.train import recursive_to
from disorderflow.datasets.protein import preprocess_protein_structure
from disorderflow.utils.misc import seed_all
AA = 'ACDEFGHIKLMNPQRSTVWY'

# Reuse helper functions from probe_antigen_signal_usage
from probe_antigen_signal_usage import write_pdb, get_cdr_real, run_recovery

APP = 'app_config.yaml'
app = yaml.safe_load(open(APP, encoding='utf-8'))
ORIG_CKPT = app['models']['bfn']['checkpoint']

N_COMPLEX = 3
N_SAMPLES = 5
TMPDIR = 'oc_validation_results/_diag'
os.makedirs(TMPDIR, exist_ok=True)

def probe_model(ckpt_path, label):
    """Run antigen signal probe with a specific checkpoint."""
    app = yaml.safe_load(open(APP, encoding='utf-8'))
    app['models']['bfn']['checkpoint'] = ckpt_path
    with open(APP, 'w', encoding='utf-8') as f:
        yaml.dump(app, f, default_flow_style=False)

    import bfn_loader
    bfn_loader._bfn_model = None
    bfn_loader._bfn_config = None
    model, _ = bfn_loader.load_bfn('cpu')
    print(f'[{label}] Loaded {ckpt_path}')

    env = lmdb.open('data/sabdab_phase3_processed/train.lmdb', readonly=True, lock=False, readahead=False, subdir=False)
    ids = pickle.load(open('data/sabdab_phase3_processed/train.lmdb-ids', 'rb'))
    random.seed(5)

    agg = {'complex': [], 'fixbb': []}
    picked = 0

    try:
        random.shuffle(ids)
        for sid in ids:
            if picked >= N_COMPLEX:
                break
            with env.begin() as txn:
                e = pickle.loads(txn.get(sid.encode()))
            if e.get('heavy') is None or e.get('antigen') is None:
                continue
            cdrs = get_cdr_real(e, 'H')
            if not cdrs:
                continue

            pdb_c = os.path.join(TMPDIR, f'{sid}_c.pdb')
            pdb_f = os.path.join(TMPDIR, f'{sid}_f.pdb')
            try:
                write_pdb(e, pdb_c, include_antigen=True)
                write_pdb(e, pdb_f, include_antigen=False)
            except Exception as ex:
                print(f'  {sid}: write_pdb failed {ex}')
                continue

            parts = [f'{s+1}-{e_}' for s, e_, _ in cdrs.values()]
            region_spec = 'H:' + ','.join(parts)
            lengths = [e_ - s for s, e_, _ in cdrs.values()]

            try:
                seed_all(42)
                des_c = run_recovery(model, pdb_c, region_spec, ['P'], N_SAMPLES, 'cuda')
                seed_all(42)
                des_f = run_recovery(model, pdb_f, region_spec, [], N_SAMPLES, 'cuda')
            except Exception as ex:
                print(f'  {sid}: design failed {ex}')
                continue

            for ci, nm in enumerate(['H1', 'H2', 'H3']):
                if nm not in cdrs:
                    continue
                s, e_, real = cdrs[nm]
                l = e_ - s
                off = sum(lengths[:ci])
                rc = []
                for d in des_c:
                    dc = d[off:off+l]
                    if len(dc) == len(real):
                        rc.append(sum(a == b for a, b in zip(dc, real)) / len(real))
                rf = []
                for d in des_f:
                    dc = d[off:off+l]
                    if len(dc) == len(real):
                        rf.append(sum(a == b for a, b in zip(dc, real)) / len(real))
                if rc and rf:
                    agg['complex'].append(np.mean(rc))
                    agg['fixbb'].append(np.mean(rf))

            picked += 1
            print(f'  [{label}] {sid}: {picked}/{N_COMPLEX} OK')
    finally:
        env.close()

    c = np.array(agg['complex'])
    f = np.array(agg['fixbb'])
    delta = (c.mean() - f.mean()) * 100
    print(f'\n=== [{label}] Antigen signal usage (n={len(c)} CDR-scaffold) ===')
    print(f'  complex: {c.mean()*100:.1f}%  fixbb: {f.mean()*100:.1f}%  Δ={delta:+.2f}pp')
    return c.mean(), f.mean(), delta


if __name__ == '__main__':
    # Probe V15 baseline
    v15_ckpt = 'logs/bfn_v15_binding_xpu_2026_06_26__15_52_10/checkpoints/best.pt'
    if os.path.exists(v15_ckpt):
        cm15, fm15, d15 = probe_model(v15_ckpt, 'V15')
    else:
        print('V15 checkpoint not found, skipping baseline')
        cm15, fm15, d15 = None, None, None

    # Probe V17c
    v17c_ckpt = 'logs/v17c_zeroinit/bfn_v17c_zeroinit_xpu_2026_06_26__23_22_18/checkpoints/best.pt'
    cm17, fm17, d17 = probe_model(v17c_ckpt, 'V17c')

    # Restore original checkpoint
    app = yaml.safe_load(open(APP, encoding='utf-8'))
    app['models']['bfn']['checkpoint'] = ORIG_CKPT
    with open(APP, 'w', encoding='utf-8') as f:
        yaml.dump(app, f, default_flow_style=False)

    print('\n============= FINAL COMPARISON =============')
    if cm15 is not None:
        print(f'V15:  complex={cm15*100:.1f}%  fixbb={fm15*100:.1f}%  Δ={d15:+.2f}pp')
    print(f'V17c: complex={cm17*100:.1f}%  fixbb={fm17*100:.1f}%  Δ={d17:+.2f}pp')

    if d17 > 1.0:
        print('\n✅ V17c antigen signal WORKING (Δ > 1pp)')
    elif d17 > 0.1:
        print('\n⚠️  V17c antigen signal MARGINAL (0.1 < Δ < 1pp)')
    else:
        print('\n❌ V17c antigen signal NOT WORKING (Δ ≈ 0)')
        print('   Architecture routing insufficient — need §3.1 approaches')
