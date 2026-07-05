#!/usr/bin/env python
"""Phase 1: BFN design with V14 checkpoint, save for AF2 validation.

V14 = grouped real-AF2 design-variant confidence training (see
configs/train/bfn_v14_grouped_conf_xpu.yml). This runner generates BFN designs
on the Abeta42/5IMK scaffold and saves them for Phase 2 AF2 OC measurement.

Usage:
  python run_oc_v14_p1.py
After training V14 with bfn_v14_grouped_conf_xpu.yml, set V14_CKPT below to the
new best.pt, then run this, then run_oc_v14_p2.py.
"""
import sys
import os
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))

import argparse
import yaml
import time
import json
import numpy as np

# Update after V14 training completes. Defaults to V12 until then (so the runner
# is executable end-to-end and only the OC value changes post-training).
V14_CKPT = os.environ.get('V14_CKPT',
                          'logs/bfn_v14_grouped_conf_xpu_2026_06_26__01_06_36/checkpoints/best.pt')

# Design mode:
#   fixbb   — scaffold only, antigen INVISIBLE to BFN during CDR design (legacy).
#   complex — build an Ab+Aβ42 complex PDB and pass the antigen chain as context
#             so the BFN encoder sees the epitope while designing CDRs. This is
#             the correct setup for binding design; the V14 ckpt was trained
#             antibody-only so this is an OOD baseline until Phase 3 retraining.
ap = argparse.ArgumentParser()
ap.add_argument('--mode', choices=['fixbb', 'complex'], default='fixbb')
ap.add_argument('--n-samples', type=int, default=10)
ap.add_argument('--design-top-n', type=int, default=5)
ap.add_argument('--output', default=None,
                help='output json path; default oc_v14_phase1_<mode>.json')
ap.add_argument('--disorder-guided', action='store_true',
                help='Enable disorder-aware CDR sampling (pliability-scaled entropy).')
ap.add_argument('--dg-strength', type=float, default=1.0,
                help='Disorder-guided pliability scaling strength (0..1+).')
args = ap.parse_args()

MODE = args.mode
OUTPUT = args.output or f'oc_validation_results/oc_v14_phase1_{MODE}.json'
# Keep the legacy default filename for the fixbb mode so old tooling resolves.
if MODE == 'fixbb' and args.output is None:
    OUTPUT = 'oc_validation_results/oc_v14_phase1.json'

DESIGN_CONFIG = {
    'scaffold_pdb': 'data/misfolding_targets/5IMK.pdb',
    'scaffold_chain': 'B',
    'target_pdb': 'data/misfolding_targets/2NAO_model1_A_1-42.pdb',
    'target_chain': 'A',
    'cdr_spec': 'B:26-33,51-58,97-113',
    'context_chains': None,           # filled below for complex mode
    'n_samples': args.n_samples,
    'design_top_n': args.design_top_n,
    'mode': MODE,
}


def auto_detect_device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def build_complex_pdb(scaffold_pdb, scaffold_chain, target_pdb, target_chain,
                      distance=18.0):
    """Build an Ab+epitope complex PDB with the epitope facing the CDR loop.

    Reuses antibody_epitope_complex.position_epitope. Returns (complex_pdb_path).
    The complex keeps the scaffold on its original chain id and writes the
    epitope onto `target_chain` so region_spec 'B:...' still resolves.
    """
    from antibody_epitope_complex import position_epitope
    out_dir = os.path.join(PROJECT_ROOT, 'oc_validation_results', '_complexes')
    os.makedirs(out_dir, exist_ok=True)
    res = position_epitope(
        scaffold_pdb, target_pdb,
        scaffold_chain=scaffold_chain, epitope_chain=target_chain,
        distance=distance, output_dir=out_dir)
    return res['pdb_path'], res.get('distance')


cfg_path = os.path.join(PROJECT_ROOT, 'app_config.yaml')
with open(cfg_path, encoding='utf-8') as f:
    app_cfg = yaml.safe_load(f)
orig = app_cfg['models']['bfn']['checkpoint']
app_cfg['models']['bfn']['checkpoint'] = V14_CKPT
with open(cfg_path, 'w', encoding='utf-8') as f:
    yaml.dump(app_cfg, f, default_flow_style=False)

try:
    from bfn_loader import run_bfn_design, load_bfn, has_disorder_head
    from idp_antibody_design import _extract_sequence_from_pdb, _parse_cdr_ranges, graft_cdrs

    import bfn_loader
    bfn_loader._bfn_model = None
    bfn_loader._bfn_config = None
    from disorderflow.utils.misc import seed_all
    seed_all(42)

    device = auto_detect_device()
    print(f'Device: {device}')
    model, config = load_bfn(device)
    print(f'V14 model loaded. Disorder head: {has_disorder_head(model)}')

    epi_seq = _extract_sequence_from_pdb(DESIGN_CONFIG['target_pdb'], DESIGN_CONFIG['target_chain'])
    scaffold_seq = _extract_sequence_from_pdb(DESIGN_CONFIG['scaffold_pdb'], DESIGN_CONFIG['scaffold_chain'])
    cdr_ranges = _parse_cdr_ranges(DESIGN_CONFIG['cdr_spec'])

    # In complex mode, build an Ab+epitope PDB and make the antigen visible to
    # the BFN encoder during CDR design (context_chains=[target_chain]).
    design_pdb = DESIGN_CONFIG['scaffold_pdb']
    if MODE == 'complex':
        design_pdb, cdist = build_complex_pdb(
            DESIGN_CONFIG['scaffold_pdb'], DESIGN_CONFIG['scaffold_chain'],
            DESIGN_CONFIG['target_pdb'], DESIGN_CONFIG['target_chain'])
        DESIGN_CONFIG['context_chains'] = [DESIGN_CONFIG['target_chain']]
        DESIGN_CONFIG['complex_pdb'] = design_pdb
        DESIGN_CONFIG['complex_distance'] = cdist
        print(f'Complex mode: built {design_pdb} (epitope-CDR ~{cdist:.1f} Å), '
              f'context_chains={DESIGN_CONFIG["context_chains"]}')
    else:
        print('FixBB mode: antigen INVISIBLE during CDR design (legacy)')

    t0 = time.time()
    designs = run_bfn_design(
        design_pdb, DESIGN_CONFIG['cdr_spec'],
        num_samples=DESIGN_CONFIG['n_samples'], stochastic=True,
        context_chains=DESIGN_CONFIG['context_chains'], device=device,
        # V14-unfreeze: rank by self-predicted ipTM so design_top_n picks the
        # highest-confidence designs to send to AF2 (was designs[:n] = blind
        # random sampling → all AF2 ipTM ~0.05). Requires an un-collapsed head.
        sort_by='iptm', descending=True,
        disorder_guided=args.disorder_guided,
        disorder_guided_strength=args.dg_strength)
    dt = time.time() - t0

    plddts = [d['plddt'] for d in designs]
    iptms = [d['iptm'] for d in designs]
    ppls = [d.get('ppl', 0) for d in designs]
    entropies = [d.get('entropy', 0) for d in designs]

    print(f'\nV14 designs: {len(designs)} in {dt:.0f}s')
    print(f'  pLDDT: mean={np.mean(plddts):.4f} std={np.std(plddts):.6f}')
    print(f'  ipTM:  mean={np.mean(iptms):.4f} std={np.std(iptms):.6f}  (V14 target within-design std > 0.05)')
    print(f'  PPL:   mean={np.mean(ppls):.1f} [{min(ppls):.0f}-{max(ppls):.0f}]')
    print(f'  Entropy: mean={np.mean(entropies):.4f} std={np.std(entropies):.4f}')

    af2_input = []
    for i, d in enumerate(designs[:DESIGN_CONFIG['design_top_n']]):
        full_ab, mutations = graft_cdrs(scaffold_seq, d['sequence'], cdr_ranges)
        af2_input.append({
            'rank': i + 1, 'sequence': d['sequence'], 'full_ab': full_ab,
            'bfn_plddt': d['plddt'], 'bfn_iptm': d['iptm'],
            'bfn_ppl': d.get('ppl', 0), 'bfn_pae': d.get('pae', 0),
            'bfn_entropy': d.get('entropy', 0),
            'mutations': len(mutations),
        })
        print(f'  [{i+1}] pLDDT={d["plddt"]:.6f} ipTM={d["iptm"]:.6f} PPL={d.get("ppl",0):.0f} {d["sequence"][:40]}')

    results = {
        'config': DESIGN_CONFIG,
        'af2_config': {'num_recycle': 3, 'use_jax': True, 'colabfold_exe': 'colabfold_batch', 'timeout_per_design': 1800},
        'epi_seq': epi_seq, 'scaffold_seq': scaffold_seq,
        'cdr_ranges': [[s, e, l] for s, e, l in cdr_ranges],
        'models': {
            'v14': {
                'checkpoint': V14_CKPT, 'label': 'V14 Grouped Real-AF2',
                'desc': 'Real-AF2 design-variant training + grouped margin/variance loss',
                'n_designs': len(designs),
                'bfn_plddt_mean': float(np.mean(plddts)),
                'bfn_plddt_std': float(np.std(plddts)),
                'bfn_iptm_mean': float(np.mean(iptms)),
                'bfn_iptm_std': float(np.std(iptms)),
                'af2_input': af2_input,
            }
        }
    }

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nPhase 1 saved: {OUTPUT}')
    print('Next: python run_oc_v14_p2.py')

finally:
    app_cfg['models']['bfn']['checkpoint'] = orig
    with open(cfg_path, 'w', encoding='utf-8') as f:
        yaml.dump(app_cfg, f, default_flow_style=False)
