#!/usr/bin/env python
"""Fast CPU AF2 multimer validation with optimized settings.

Strategy: use minimal sequence context (CDR-only + short epitope),
aggressive MSA limits, and 1 recycle for speed (~5 min/design on CPU).
"""
import os, sys, json, subprocess, time
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
VENV = PROJECT_DIR.parent / 'venv'
COLABFOLD = VENV / 'Scripts' / 'colabfold_batch.exe'

# AQP4 C-terminal epitope (interface region from pipeline analysis)
# Interface residues 270-296 map to the epitope region
AQP4_EPITOPE = (
    "KRLKEAFSKAAQQTKGSYMEVEDNRSQVETDDLILKPGVVHVIDVDRGEELGKKVKQSDPSSH"
)

def load_designs(fasta_path):
    designs = []
    with open(fasta_path) as f:
        header, seq = None, ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header and seq:
                    designs.append((header, seq))
                header = line[1:]
                seq = ''
            elif line:
                seq += line
        if header and seq:
            designs.append((header, seq))
    return designs

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--designs', required=True)
    p.add_argument('--output_dir', default='alphafold_results/aqp4_cdr_validation')
    p.add_argument('--max_designs', type=int, default=6)
    args = p.parse_args()

    designs = load_designs(args.designs)
    print(f'Loaded {len(designs)} CDR designs')
    print(f'Epitope: {len(AQP4_EPITOPE)}aa')
    print(f'CDR: {len(designs[0][1]) if designs else 0}aa each')
    print(f'Complex: {len(AQP4_EPITOPE) + len(designs[0][1]) if designs else 0}aa total')
    print(f'Settings: 1 model, 1 recycle, small MSA\n')

    results = []
    for i, (header, seq) in enumerate(designs[:args.max_designs]):
        tag = f'cdr_val_{i}'
        out_dir = Path(args.output_dir) / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        # Multimer FASTA
        fasta_path = out_dir / f'{tag}.fasta'
        with open(fasta_path, 'w') as f:
            f.write(f'>{tag}\n{AQP4_EPITOPE}:{seq}\n')

        print(f'[{i+1}/{min(len(designs), args.max_designs)}] {tag}')
        print(f'  CDR: {seq}')
        t0 = time.time()

        cmd = [
            str(COLABFOLD),
            str(fasta_path), str(out_dir),
            '--num-models', '1',
            '--num-recycle', '1',
            '--model-type', 'alphafold2_multimer_v3',
            '--rank', 'auto',
            '--stop-at-score', '50',
            '--max-seq', '128',
            '--max-extra-seq', '256',
        ]

        env = os.environ.copy()
        env['PATH'] = str(VENV / 'Scripts') + os.pathsep + env.get('PATH', '')

        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200,
                              cwd=str(PROJECT_DIR), env=env)
            elapsed = time.time() - t0

            if r.returncode != 0:
                print(f'  FAILED (exit {r.returncode}): {r.stderr[-200:]}')
                results.append({'design': tag, 'seq': seq, 'success': False, 'error': r.stderr[-300:]})
                continue

            # Parse scores
            scores_files = sorted(out_dir.glob('*_scores_rank_001_*.json'))
            pdb_files = sorted(out_dir.glob('*_rank_001_*.pdb'))
            if not scores_files:
                results.append({'design': tag, 'seq': seq, 'success': False, 'error': 'no scores JSON'})
                continue

            with open(scores_files[0]) as sf:
                scores = json.load(sf)
            plddt = scores.get('plddt', [])
            mean_plddt = sum(plddt) / max(len(plddt), 1)
            ptm = scores.get('ptm', 0)
            iptm = scores.get('iptm')

            # Per-chain pLDDT (first ~N residues = epitope, rest = CDR)
            n_epi = len(AQP4_EPITOPE)
            epi_plddt = sum(plddt[:n_epi]) / max(len(plddt[:n_epi]), 1) if len(plddt) > 0 else 0
            cdr_plddt = sum(plddt[n_epi:]) / max(len(plddt[n_epi:]), 1) if len(plddt) > n_epi else 0

            print(f'  Time: {elapsed:.0f}s')
            print(f'  pLDDT: total={mean_plddt:.1f}  epitope={epi_plddt:.1f}  CDR={cdr_plddt:.1f}')
            print(f'  pTM={ptm:.3f}  ipTM={iptm:.3f}')

            if iptm and iptm > 0.5 and cdr_plddt > 60:
                quality = 'EXCELLENT'
            elif iptm and iptm > 0.3 and cdr_plddt > 50:
                quality = 'GOOD'
            elif iptm and iptm > 0.15:
                quality = 'MODERATE'
            else:
                quality = 'POOR'
            print(f'  Quality: {quality}')

            results.append({
                'design': tag, 'seq': seq, 'success': True,
                'plddt_mean': mean_plddt / 100.0,
                'plddt_epitope': epi_plddt / 100.0,
                'plddt_cdr': cdr_plddt / 100.0,
                'ptm': ptm, 'iptm': iptm,
                'quality': quality, 'time_s': elapsed,
                'pdb_path': str(pdb_files[0]) if pdb_files else None,
            })

        except subprocess.TimeoutExpired:
            print(f'  TIMEOUT')
            results.append({'design': tag, 'seq': seq, 'success': False, 'error': 'timeout'})
        except Exception as e:
            print(f'  ERROR: {e}')
            results.append({'design': tag, 'seq': seq, 'success': False, 'error': str(e)})

    # Summary
    print(f'\n{"="*70}')
    print(f'  AF2 MULTIMER VALIDATION RESULTS')
    print(f'{"="*70}')
    success = [r for r in results if r.get('success')]
    print(f'  Completed: {len(success)}/{len(results)}')

    if success:
        print(f'\n  {"CDR Design":<12} {"pLDDT(Epi)":>10} {"pLDDT(CDR)":>10} {"ipTM":>8} {"pTM":>8} {"Quality":<12}')
        print(f'  {"-"*65}')
        for r in sorted(success, key=lambda x: x.get('iptm') or 0, reverse=True):
            iptm = r.get('iptm') or 0
            print(f'  {r["design"]:<12} {r["plddt_epitope"]:>10.3f} {r["plddt_cdr"]:>10.3f} {iptm:>8.3f} {r["ptm"]:>8.3f} {r["quality"]:<12}')

    out_path = Path(args.output_dir) / 'cdr_validation_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\n  Results saved: {out_path}')

if __name__ == '__main__':
    main()
