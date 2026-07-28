#!/usr/bin/env python3
"""V18 Unified Benchmark — 4-metric standardized evaluation (§7.3).

Any new model version MUST report all four metrics simultaneously
(val loss alone is meaningless — V17g proved this: val↓0.32 but Δ went negative).

Metrics:
  OC       — overconfidence ratio: BFN_iptm / AF2_iptm (target ≤2x)
  Spearman — rank correlation between BFN and AF2 scores (target ρ>0.4)
  Δ        — antigen signal: complex recovery − fixbb recovery (target >5pp)
  ceiling  — max AF2 ipTM across designs (current: V14 0.081, V17i 0.121)

Usage:
    python scripts/benchmark.py --generator v14|v17i|mpnn --samples 20
    python scripts/benchmark.py --generator mpnn --bfn-ckpt path/to/v14.pt

Output:
    scripts/benchmark_results/benchmark_<timestamp>.json
    Prints 4-metric summary to stdout.
"""
import sys, os, json, time, argparse, subprocess, tempfile, shutil
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'modules'))

import numpy as np

# ── Defaults ──
V14_CKPT = 'logs/bfn_v14_grouped_conf_xpu_2026_06_26__01_06_36/checkpoints/best.pt'
V17I_CKPT = 'logs/v17c_zeroinit/bfn_v17c_zeroinit_xpu_2026_06_26__23_22_18/checkpoints/4600.pt'
SCAFFOLD = 'data/misfolding_targets/3STB.pdb'
ANTIGEN = 'data/abeta_conformations/pdbs/abeta42_seed0_42.pdb'
EPITOPE = 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA'
CDR_SPEC = 'A:26-33,A:51-58,A:97-113'
AA = 'ACDEFGHIKLMNPQRSTVWY'


def _spearman(a, b):
    if len(a) < 3: return float('nan')
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d2 = ((ra - rb) ** 2).sum()
    return 1 - 6 * d2 / (len(a) * (len(a) ** 2 - 1)) if len(a) > 1 else float('nan')


def run_benchmark(generator='v14', bfn_ckpt=None, samples=20, output=None):
    """Run full 4-metric benchmark."""
    bfn_ckpt = bfn_ckpt or (V17I_CKPT if generator == 'v17i' else V14_CKPT)
    results = {
        'generator': generator,
        'bfn_ckpt': bfn_ckpt,
        'timestamp': datetime.now().isoformat(),
        'samples': samples,
    }
    designs = []

    # ── 1. Generate designs ──
    print(f'[1/4] Generating {samples} designs (generator={generator})...')
    t0 = time.time()

    if generator == 'mpnn':
        # Use ProteinMPNN + BFN hybrid
        from mpnn_bfn_hybrid import build_complex_pdb, run_proteinmpnn, score_with_bfn
        complex_pdb = build_complex_pdb(SCAFFOLD, 'A', ANTIGEN, 'P')
        mpnn_designs = run_proteinmpnn(complex_pdb, 'A P', samples, 0.5, 42)
        designs = score_with_bfn(mpnn_designs, complex_pdb, CDR_SPEC, bfn_ckpt=bfn_ckpt)
        for design in designs:
            design['bfn_iptm'] = design.get('bfn_iptm', design.get('iptm'))
            design['bfn_plddt'] = design.get('bfn_plddt', design.get('plddt_mean'))
        os.unlink(complex_pdb)
    else:
        # BFN generator (V14 or V17i)
        import yaml, torch
        from easydict import EasyDict
        from disorderflow.utils.data import PaddingCollate
        from disorderflow.utils.train import recursive_to
        from disorderflow.utils.transforms import get_transform
        from disorderflow.datasets.protein import preprocess_protein_structure
        from disorderflow.utils.misc import seed_all

        # Load model
        app = yaml.safe_load(open('app_config.yaml', encoding='utf-8'))
        orig_ckpt = app['models']['bfn']['checkpoint']
        app['models']['bfn']['checkpoint'] = bfn_ckpt
        yaml.dump(app, open('app_config.yaml', 'w', encoding='utf-8'))

        import bfn_loader; bfn_loader._bfn_model = None; bfn_loader._bfn_config = None
        model, cfg = bfn_loader.load_bfn('cpu')
        app['models']['bfn']['checkpoint'] = orig_ckpt
        yaml.dump(app, open('app_config.yaml', 'w', encoding='utf-8'))

        # Build complex structure
        def parse_regions(spec):
            r = {}
            for p in spec.split(','):
                ch, rg = p.split(':'); s, e = map(int, rg.split('-'))
                r.setdefault(ch, []).extend(range(s - 1, e))
            return {k: sorted(set(v)) for k, v in r.items()}

        rd = parse_regions(CDR_SPEC)
        tx = get_transform([{'type': 'mask_region', 'regions': rd},
                            {'type': 'merge_protein'}, {'type': 'patch_protein'}])
        ss = preprocess_protein_structure(SCAFFOLD, chain_ids=['A'])
        ags = preprocess_protein_structure(ANTIGEN, chain_ids=['P'])
        merged = {'id': 'A+P', 'chains': ss['chains'] + ags['chains'],
                  'num_chains': ss['num_chains'] + ags['num_chains'],
                  'all_chain_ids': ss['all_chain_ids'] + ags['all_chain_ids']}

        def sample_structure(structure, seed):
            seed_all(seed)
            batch = recursive_to(PaddingCollate()([tx(structure)]), 'cpu')
            gm = batch['generate_flag'][0].bool()
            if gm.sum() == 0:
                raise ValueError('CDR specification selected no residues')
            with torch.no_grad():
                traj = model.sample(batch, {'deterministic': False, 'num_recycles': 3})
            pred_aa = traj[0][2][0][gm]
            logits = traj['pred_logits'][0][gm][..., :20]
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            nll = -log_probs[torch.arange(len(pred_aa)), pred_aa].mean()
            native_aa = batch['aa'][0][gm]
            recovery = (pred_aa == native_aa).float().mean().item()
            return traj, gm, pred_aa, float(torch.exp(nll)), float(
                -(probs * log_probs).sum(dim=-1).mean()), recovery

        for si in range(samples):
            traj, gm, pred_aa, ppl, entropy, complex_recovery = sample_structure(
                merged, 42 + si)
            _, _, _, _, _, fixbb_recovery = sample_structure(ss, 42 + si)
            cdr = ''.join(AA[a] for a in pred_aa.tolist())
            disorder = traj.get('disorder')
            designs.append({
                'sequence': cdr,
                'bfn_iptm': float(traj.get('iptm', torch.tensor([0])).mean()),
                'bfn_plddt': float(traj.get('plddt', torch.zeros(1, gm.sum()))[0].mean()),
                'ppl': ppl,
                'entropy': entropy,
                'disorder_mean': (float(disorder[0][gm].mean())
                                  if disorder is not None else None),
                'complex_recovery': complex_recovery,
                'fixbb_recovery': fixbb_recovery,
            })

    results['n_designs'] = len(designs)
    print(f'  {len(designs)} designs ({time.time()-t0:.0f}s)')

    # ── 2. Compute Δ (antigen signal) ──
    print('[2/4] Computing antigen signal Δ...')
    paired = [d for d in designs
              if d.get('complex_recovery') is not None and d.get('fixbb_recovery') is not None]
    delta = (100.0 * np.mean([d['complex_recovery'] - d['fixbb_recovery'] for d in paired])
             if paired else None)
    results['delta_pp'] = delta
    results['delta_n_pairs'] = len(paired)
    if delta is None:
        results['delta_unavailable_reason'] = (
            'Generator did not produce paired Complex/FixBB measurements')
        print('  Delta unavailable (no paired Complex/FixBB measurements)')
    else:
        print(f'  Delta = {delta:+.1f}pp ({len(paired)} paired samples)')

    # ── 3. AF2 validation ──
    print(f'[3/4] AF2 validation on top-10 designs...')
    from Bio.PDB import PDBParser, PDBIO
    from build_design_variant_dataset import _batch_af2_wsl

    aa3 = {'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','E':'GLU','Q':'GLN',
           'G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE',
           'P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL'}
    af2_dir = tempfile.mkdtemp(prefix='bench_af2_')
    seqs = []

    for i, d in enumerate(designs[:10]):
        # Graft CDR into scaffold
        parser = PDBParser(QUIET=True); s = parser.get_structure('s', SCAFFOLD)
        regions = []
        for part in CDR_SPEC.split(','):
            ch, rng = part.split(':'); start, end = map(int, rng.split('-'))
            if ch == 'A': regions.append((start - 1, end))
        pos = 0
        for start, end in regions:
            for j in range(start, end):
                if pos < len(d['sequence']):
                    try: s[0]['A'][j+1].resname = aa3.get(d['sequence'][pos], 'GLY')
                    except: pass
                    pos += 1
        out_pdb = os.path.join(af2_dir, f'design_{i:02d}.pdb')
        io = PDBIO(); io.set_structure(s); io.save(out_pdb)

        seq = []; seen = set()
        with open(out_pdb) as fh:
            for l in fh:
                if l.startswith('ATOM') and l[12:16].strip() == 'CA':
                    ri = l[22:27]
                    if ri not in seen: seen.add(ri); seq.append(aa3.get(l[17:20].strip(), 'X'))
        seqs.append(''.join(seq))

    af2_results = _batch_af2_wsl(seqs, EPITOPE, num_recycle=1)
    shutil.rmtree(af2_dir, ignore_errors=True)

    af2_iptms = []
    for i, r in enumerate(af2_results):
        if r and r.get('success'):
            designs[i]['af2_iptm'] = r.get('iptm', 0)
            designs[i]['af2_plddt'] = r.get('plddt', 0)
            af2_iptms.append(r.get('iptm', 0))
        else:
            designs[i]['af2_iptm'] = 0
            designs[i]['af2_plddt'] = 0

    # ── 4. Compute OC, Spearman, ceiling ──
    print('[4/4] Computing metrics...')
    bfn_iptms = [d['bfn_iptm'] for d in designs[:10] if d.get('af2_iptm', 0) > 0]
    valid_af2 = [d['af2_iptm'] for d in designs[:10] if d.get('af2_iptm', 0) > 0]

    if bfn_iptms and valid_af2:
        oc = np.mean([b / a for b, a in zip(bfn_iptms, valid_af2) if a > 0.001])
        rho = _spearman(np.array(bfn_iptms), np.array(valid_af2))
        ceiling = max(valid_af2)
    else:
        oc, rho, ceiling = 999, float('nan'), 0

    results['oc_ratio'] = round(oc, 2)
    results['spearman_rho'] = round(rho, 4) if not np.isnan(rho) else None
    results['ceiling'] = round(ceiling, 4)
    results['af2_success'] = f'{len(valid_af2)}/{min(10, len(designs))}'

    # ── Output ──
    os.makedirs('scripts/benchmark_results', exist_ok=True)
    if output is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output = f'scripts/benchmark_results/benchmark_{generator}_{ts}.json'
    results['output'] = output
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n{"="*50}')
    print(f'V18 BENCHMARK — {generator}')
    print(f'{"="*50}')
    print(f'  OC ratio:    {oc:.2f}x  (target ≤2x)')
    print(f'  Spearman ρ:  {rho:.4f}  (target >0.4)')
    delta_text = f'{delta:+.1f}pp' if delta is not None else 'N/A'
    print(f'  Delta signal:{delta_text:>10} (target >5pp)')
    print(f'  Ceiling:     {ceiling:.4f}  (AF2 ipTM max)')
    print(f'{"="*50}')
    print(f'Saved to {output}')
    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='V18 Unified Benchmark')
    ap.add_argument('--generator', choices=['v14', 'v17i', 'mpnn'], default='v14')
    ap.add_argument('--bfn-ckpt', default=None, help='BFN checkpoint for scoring')
    ap.add_argument('--samples', type=int, default=20)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()
    run_benchmark(args.generator, args.bfn_ckpt, args.samples, args.output)
