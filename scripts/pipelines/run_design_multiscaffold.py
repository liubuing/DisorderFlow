import os
#!/usr/bin/env python3
"""Direction-C: multi-scaffold × multi-conformation Aβ42 design, save for AF2.

Breaks the fixed-5IMK + 19/42-Aβ42 ceiling by:
  - Scaffolds: 3STB-A, 5IMK-B (clean VHHs, full backbone).
  - Aβ42: 5 full-backbone seed conformations (data/abeta_conformations/pdbs,
    each parses to full 42 residues — vs 19/42 from the single NMR model).
  - disorder-guided sampling (retrained head, strength 0.3).

Builds Ab+Aβ42 complexes via position_epitope (bug-fixed: only scaffold chain
atoms written; Aβ42 on chain 'P'), runs run_bfn_design in complex mode, ranks
by BFN ipTM (now ρ~+0.6 vs AF2), and writes a phase1-style JSON for p2.

Usage:
  V14_CKPT=logs/disorder_head_retrain.pt python run_design_multiscaffold.py
then:
  python -c "import run_oc_v14_p2 as p2; p2.run_phase2('oc_validation_results/oc_v15C_phase1.json','oc_validation_results/oc_v15C_phase2.json')"
"""
import sys, os, re, time, json, tempfile
if sys.platform == 'win32':
    import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))

import yaml, numpy as np

CKPT = os.environ.get('V14_CKPT', 'logs/disorder_head_retrain.pt')
OUTPUT = os.environ.get('OUTPUT', 'oc_validation_results/oc_v15C_phase1.json')
DG_STRENGTH = float(os.environ.get('DG_STRENGTH', '0.3'))
N_SAMPLES = int(os.environ.get('N_SAMPLES', '6'))
TOP_N = int(os.environ.get('TOP_N', '5'))

# (scaffold_pdb, scaffold_chain, region_spec_for_that_chain)
SCAFFOLDS = [
    ('data/misfolding_targets/3STB.pdb', 'A', 'A:26-33,51-58,97-113'),
    ('data/misfolding_targets/5IMK.pdb', 'B', 'B:26-33,51-58,97-113'),
]
# Aβ42 conformations to try (chain 'P' in each)
ABETA_SEEDS = [
    'data/abeta_conformations/pdbs/abeta42_seed0_42.pdb',
    'data/abeta_conformations/pdbs/abeta42_seed2_42.pdb',
    'data/abeta_conformations/pdbs/abeta42_seed4_42.pdb',
]

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    APP = os.path.join(PROJECT_ROOT, 'app_config.yaml')
    app = yaml.safe_load(open(APP, encoding='utf-8'))
    orig = app['models']['bfn']['checkpoint']
    app['models']['bfn']['checkpoint'] = CKPT
    open(APP, 'w', encoding='utf-8').write(yaml.dump(app, default_flow_style=False))

    try:
        import bfn_loader
        bfn_loader._bfn_model = None; bfn_loader._bfn_config = None
        from bfn_loader import run_bfn_design, load_bfn, has_disorder_head
        from antibody_epitope_complex import position_epitope
        from idp_antibody_design import _parse_cdr_ranges, graft_cdrs, _extract_sequence_from_pdb
        from disorderflow.utils.misc import seed_all
        seed_all(42)
        device = 'cuda'
        model, _ = load_bfn(device)
        print(f"Loaded {CKPT} | disorder head: {has_disorder_head(model)} | dg_strength={DG_STRENGTH}")

        out_dir = os.path.join(PROJECT_ROOT, 'oc_validation_results', '_complexes')
        os.makedirs(out_dir, exist_ok=True)
        af2_input = []
        rank = 0
        for scaff_pdb, scaff_chain, region_spec in SCAFFOLDS:
            scaff_seq = _extract_sequence_from_pdb(scaff_pdb, scaff_chain)
            cdr_ranges = _parse_cdr_ranges(region_spec)
            for epitope_pdb in ABETA_SEEDS:
                base = os.path.splitext(os.path.basename(scaff_pdb))[0]
                epi_base = os.path.splitext(os.path.basename(epitope_pdb))[0]
                res = position_epitope(scaff_pdb, epitope_pdb,
                                       scaffold_chain=scaff_chain, epitope_chain='P',
                                       distance=18.0, output_dir=out_dir)
                complex_pdb = res['pdb_path']
                seed_tag = re.search(r'seed(\d+)', epi_base).group(1)
                tag = f"{base}-ch{scaff_chain}_ab42seed{seed_tag}"
                t0 = time.time()
                designs = run_bfn_design(complex_pdb, region_spec,
                                         num_samples=N_SAMPLES, stochastic=True,
                                         context_chains=['P'], device=device,
                                         sort_by='iptm', descending=True,
                                         disorder_guided=True,
                                         disorder_guided_strength=DG_STRENGTH)
                # take best 2 per (scaff,seed) to keep AF2 budget bounded
                for d in designs[:2]:
                    full_ab, mut = graft_cdrs(scaff_seq, d['sequence'], cdr_ranges)
                    rank += 1
                    af2_input.append({
                        'rank': rank, 'tag': tag, 'sequence': d['sequence'],
                        'full_ab': full_ab,
                        'bfn_plddt': d['plddt'], 'bfn_iptm': d['iptm'],
                        'bfn_ppl': d.get('ppl', 0), 'bfn_pae': d.get('pae', 0),
                        'bfn_entropy': d.get('entropy', 0),
                        'mutations': len(mut),
                    })
                best = designs[0]
                print(f"  {tag}: {len(designs)} designs in {time.time()-t0:.0f}s | "
                      f"best ppl={best['ppl']:.0f} iptm={best['iptm']:.4f} {best['sequence'][:33]}")

        plddts = [r['bfn_plddt'] for r in af2_input]
        iptms = [r['bfn_iptm'] for r in af2_input]
        # epitope sequence for AF2 = full Aβ42 (seed0)
        epi_seq = 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA'
        out = {
            'config': {'scaffolds': [s[0] for s in SCAFFOLDS], 'abeta_seeds': ABETA_SEEDS,
                       'ckpt': CKPT, 'dg_strength': DG_STRENGTH,
                       'n_samples': N_SAMPLES, 'design_top_n_per_combo': 2},
            'af2_config': {'num_recycle': 3, 'use_jax': True, 'colabfold_exe': 'colabfold_batch', 'timeout_per_design': 1800},
            'epi_seq': epi_seq,
            'models': {'v15C': {
                'checkpoint': CKPT, 'label': 'V15C multi-scaffold multi-conf disorder-guided',
                'n_designs': len(af2_input),
                'bfn_plddt_mean': float(np.mean(plddts)), 'bfn_plddt_std': float(np.std(plddts)),
                'bfn_iptm_mean': float(np.mean(iptms)), 'bfn_iptm_std': float(np.std(iptms)),
                'af2_input': af2_input,
            }},
        }
        os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
        json.dump(out, open(OUTPUT, 'w', encoding='utf-8'), indent=2, default=str)
        print(f"\nPhase 1 saved: {OUTPUT}  ({len(af2_input)} designs)")
        print("Next: python -c \"import run_oc_v14_p2 as p2; p2.run_phase2('%s','%s')\"" % (OUTPUT, OUTPUT.replace('phase1','phase2')))
    finally:
        app2 = yaml.safe_load(open(APP, encoding='utf-8'))
        app2['models']['bfn']['checkpoint'] = orig
        open(APP, 'w', encoding='utf-8').write(yaml.dump(app2, default_flow_style=False))


if __name__ == '__main__':
    main()
