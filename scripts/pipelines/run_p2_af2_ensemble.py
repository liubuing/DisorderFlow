import os
#!/usr/bin/env python
"""P2 Ensemble Scoring with AF2 ipTM as scoring source.

Full pipeline:
  1. BFN design 10 CDRs (FixBB)
  2. For each design × 5 epitope conformations: run AF2 → get ipTM
  3. Compute ensemble metrics per design
  4. Compare ensemble ranking vs single-conformation ranking vs PPL ranking

Usage:
  python run_p2_af2_ensemble.py
"""
import sys, os
if sys.platform == 'win32':
    import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))
sys.path.insert(0, 'scripts/build')

import time, json, yaml, numpy as np

CKPT = 'logs/bfn_v12_seqconf_xpu_2026_06_20__02_54_40/checkpoints/best.pt'
N_DESIGNS = 10
N_CONFS = 5  # must be <= 5 (we have 5 seeds in the pickle)

# ── Setup checkpoint ──
cfg_path = os.path.join(PROJECT_ROOT, 'app_config.yaml')
with open(cfg_path) as f:
    app_cfg = yaml.safe_load(f)
orig_ckpt = app_cfg['models']['bfn']['checkpoint']
app_cfg['models']['bfn']['checkpoint'] = CKPT
with open(cfg_path, 'w') as f:
    yaml.dump(app_cfg, f, default_flow_style=False)

try:
    # ── 1. Load models and data ──
    from bfn_loader import run_bfn_design, load_bfn
    from idp_antibody_design import ( _extract_sequence_from_pdb,
        _parse_cdr_ranges, graft_cdrs )
    from af2_jax_runner import run_multimer_prediction
    from disorderflow.utils.misc import seed_all
    import bfn_loader
    bfn_loader._bfn_model = None
    bfn_loader._bfn_config = None
    seed_all(42)
    device = 'cuda'

    # ── 2. Generate CDR designs ──
    print('Step 1: BFN design...')
    t0 = time.time()
    designs = run_bfn_design(
        'data/misfolding_targets/5IMK.pdb', 'B:26-33,51-58,97-113',
        num_samples=N_DESIGNS, stochastic=True, context_chains=None, device=device)
    print(f'  {len(designs)} designs in {time.time()-t0:.0f}s')
    ppls = [d.get('ppl', 0) for d in designs]
    print(f'  PPL range: [{min(ppls):.0f}, {max(ppls):.0f}]')

    # Graft CDRs for AF2 validation
    scaffold_seq = _extract_sequence_from_pdb(
        'data/misfolding_targets/5IMK.pdb', 'B')
    epi_seq = _extract_sequence_from_pdb(
        'data/misfolding_targets/2NAO_model1_A_1-42.pdb', 'A')
    cdr_ranges = _parse_cdr_ranges('B:26-33,51-58,97-113')
    full_abs = []
    for d in designs:
        ab_seq, _ = graft_cdrs(scaffold_seq, d['sequence'], cdr_ranges)
        full_abs.append(ab_seq)

    # ── 3. AF2 multi-seed scoring ──
    # Different seeds → different predicted epitope conformations.
    # This captures IDP conformational heterogeneity.
    seeds = [1, 3, 5, 7, 9][:N_CONFS]
    print(f'\nStep 2: AF2 multi-seed ({N_DESIGNS} designs × {N_CONFS} seeds = {N_DESIGNS*N_CONFS} runs)...')

    results = []
    t_start = time.time()

    for d_idx in range(N_DESIGNS):
        ab_seq = full_abs[d_idx]
        for c_idx, seed in enumerate(seeds):
            idx = d_idx * N_CONFS + c_idx + 1
            total = N_DESIGNS * N_CONFS
            elapsed = time.time() - t_start
            eta = elapsed / max(idx, 1) * (total - idx)
            print(f'  [{idx}/{total}] d{d_idx+1}s{seed} ({elapsed:.0f}s, ETA {eta:.0f}s)...',
                  end=' ', flush=True)

            t_run = time.time()
            r = run_multimer_prediction(
                ab_seq, epi_seq, num_recycle=3, jax_random_seed=seed)
            dt = time.time() - t_run

            if r.get('success'):
                results.append({
                    'design_idx': d_idx, 'conf_idx': c_idx,
                    'iptm': r['iptm'], 'plddt': r['plddt'],
                    'interface_pae': r.get('interface_pae', 99),
                })
                print(f'ipTM={r["iptm"]:.3f} ({dt:.0f}s)')
            else:
                print(f'FAILED ({dt:.0f}s)')
                results.append({
                    'design_idx': d_idx, 'conf_idx': c_idx,
                    'iptm': None, 'plddt': None, 'interface_pae': None,
                })

    # ── 5. Compute ensemble metrics ──
    print('\nStep 4: Computing ensemble metrics...')
    ensemble = []
    for d_idx in range(N_DESIGNS):
        conf_vals = [r for r in results if r['design_idx'] == d_idx and r['iptm'] is not None]
        if not conf_vals:
            continue
        iptms = [r['iptm'] for r in conf_vals]
        plddts = [r['plddt'] for r in conf_vals if r['plddt'] is not None]
        entry = {
            'design_idx': d_idx,
            'sequence': designs[d_idx]['sequence'][:40],
            'ppl': designs[d_idx].get('ppl', 0),
            'bfn_iptm': designs[d_idx].get('iptm', 0),
            'bfn_plddt': designs[d_idx].get('plddt', 0),
            'n_confs': len(conf_vals),
            'af2_iptm_mean': float(np.mean(iptms)),
            'af2_iptm_worst': float(np.min(iptms)),
            'af2_iptm_best': float(np.max(iptms)),
            'af2_iptm_std': float(np.std(iptms)),
            'af2_plddt_mean': float(np.mean(plddts)) if plddts else 0,
            'af2_iptm_single': conf_vals[0]['iptm'],  # first conf
        }
        ensemble.append(entry)

    # ── 6. Compare rankings ──
    print(f'\n{"="*70}')
    print(f'  P2 AF2 ENSEMBLE RESULTS')
    print(f'{"="*70}')
    print(f'  {"Rk":<3} {"Mean":<8} {"Worst":<8} {"Best":<8} {"Std":<8} {"Single":<8} {"PPL":<5} {"Seq":<30}')
    print(f'  {"-"*3} {"-"*8} {"-"*8} {"-"*8} {"-"*8} {"-"*8} {"-"*5} {"-"*30}')

    # Sort by ensemble mean
    by_mean = sorted(ensemble, key=lambda d: d['af2_iptm_mean'], reverse=True)
    for i, d in enumerate(by_mean):
        print(f'  {i+1:<3} {d["af2_iptm_mean"]:.4f}  {d["af2_iptm_worst"]:.4f}  '
              f'{d["af2_iptm_best"]:.4f}  {d["af2_iptm_std"]:.4f}  '
              f'{d["af2_iptm_single"]:.4f}  {d["ppl"]:.0f}    {d["sequence"][:30]}')

    # Analysis
    means = [d['af2_iptm_mean'] for d in ensemble]
    worsts = [d['af2_iptm_worst'] for d in ensemble]
    stds = [d['af2_iptm_std'] for d in ensemble]
    ppls = [d['ppl'] for d in ensemble]

    print(f'\n  Score ranges:')
    print(f'    Mean:  [{min(means):.4f}, {max(means):.4f}]')
    print(f'    Worst: [{min(worsts):.4f}, {max(worsts):.4f}]')
    print(f'    Std:   [{min(stds):.4f}, {max(stds):.4f}]')
    print(f'    PPL:   [{min(ppls):.0f}, {max(ppls):.0f}]')

    if max(means) > min(means) * 1.1:
        print(f'\n  ✅ Ensemble mean ipTM DIFFERENTIATES designs!')
    else:
        print(f'\n  ⚠ Ensemble mean has low differentiation')

    # Correlation analysis
    if len(ensemble) >= 5:
        corr_ppl_mean = np.corrcoef(ppls, means)[0, 1]
        corr_ppl_worst = np.corrcoef(ppls, worsts)[0, 1]
        print(f'\n  Correlations:')
        print(f'    PPL vs Mean:  {corr_ppl_mean:.3f}')
        print(f'    PPL vs Worst: {corr_ppl_worst:.3f}')
        if abs(corr_ppl_mean) > 0.3:
            print(f'    → PPL has some predictive power for ensemble ipTM')
        else:
            print(f'    → PPL is not predictive of ensemble ipTM')

    # Save results
    out = {
        'config': {'n_designs': N_DESIGNS, 'n_confs': N_CONFS},
        'rmsf': {'mean': float(rmsf.mean()), 'min': float(rmsf.min()), 'max': float(rmsf.max())},
        'ensemble': ensemble,
    }
    os.makedirs('oc_validation_results', exist_ok=True)
    with open('oc_validation_results/p2_af2_ensemble.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\n  Results saved to: oc_validation_results/p2_af2_ensemble.json')

finally:
    app_cfg['models']['bfn']['checkpoint'] = orig_ckpt
    with open(cfg_path, 'w') as f:
        yaml.dump(app_cfg, f, default_flow_style=False)
