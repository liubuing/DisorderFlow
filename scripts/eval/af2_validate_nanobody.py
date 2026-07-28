#!/usr/bin/env python
"""AF2 multimer validation of AQP4 nanobody designs.

Runs AlphaFold2-multimer-v3 to predict complex structures of AQP4 epitope
with optimized nanobody sequences, then scores interface confidence (ipTM, pLDDT, PAE).
"""
import os, sys, json, subprocess, time
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
VENV = PROJECT_DIR.parent / 'venv'
COLABFOLD = VENV / 'Scripts' / 'colabfold_batch.exe'

# AQP4 full sequence
AQP4_FULL = (
    "MSDRPTARRWGKCGPLCTRENIMVAFKGVWTQAFWKAVTAEFLAMLIFVLLSLGSTINWGGTEKPLPVD"
    "MVLISLCFGLSIATMVQCFGHISGGHINPAVTVAMVCTRKISIAKSVFYIAAQCLGAIIGAGILYLVTPP"
    "SVVGGLGVTMVHGNLTAGHGLLVELIITFQLVFTIFASCDSKRTDVTGSIALAIGFSVAIGHLFAINYTG"
    "ASMNPARSFGPAVIMNWENHWIYWVGPIIGAVLAGALYEYVFCPDVELKRRLKEAFSKAAQQTKGSYMEV"
    "EDNRSQVETDDLILKPGVVHVIDVDRGEELGKKVKQSDPSSH"
)

# Use C-terminal epitope region (contains the identified interface residues)
# Interface residues: KFSKAAQQTKDLILKPGVVHVIDVDR are at ~positions 270-296
EPITOPE_START = 250  # 0-based
EPITOPE = AQP4_FULL[EPITOPE_START:]  # 73 residues, gives good context
print(f"Epitope: residues {EPITOPE_START+1}-{len(AQP4_FULL)} ({len(EPITOPE)}aa)")
print(f"Epitope seq: {EPITOPE[:30]}...{EPITOPE[-30:]}")


def load_designs(fasta_path):
    """Load sequences from a FASTA file."""
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


def run_af2_multimer(epitope_seq, nanobody_seq, output_dir, name='complex'):
    """Run AF2 multimer prediction for epitope + nanobody complex."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ColabFold multimer format: colon-separated chains
    fasta_path = output_dir / f'{name}.fasta'
    with open(fasta_path, 'w') as f:
        f.write(f'>{name}\n{epitope_seq}:{nanobody_seq}\n')

    cmd = [
        str(COLABFOLD),
        str(fasta_path),
        str(output_dir),
        '--num-models', '1',
        '--num-recycle', '3',
        '--model-type', 'alphafold2_multimer_v3',
        '--rank', 'auto',
        '--stop-at-score', '80',
    ]

    env = os.environ.copy()
    env['PATH'] = str(VENV / 'Scripts') + os.pathsep + env.get('PATH', '')

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                            cwd=str(PROJECT_DIR), env=env)

    return result, output_dir


def parse_af2_results(result_dir):
    """Parse AF2 output directory for scores and PDB files."""
    result_dir = Path(result_dir)

    # Find scores JSON
    scores_files = sorted(result_dir.glob('*_scores_rank_001_*.json'))
    pdb_files = sorted(result_dir.glob('*_rank_001_*.pdb'))

    if not scores_files:
        return None

    with open(scores_files[0]) as f:
        scores = json.load(f)

    plddt = scores.get('plddt', [0])
    mean_plddt = sum(plddt) / max(len(plddt), 1)
    ptm = scores.get('ptm', 0)
    iptm = scores.get('iptm')
    max_pae = scores.get('max_pae', 999)
    pae = scores.get('pae', None)

    # Compute per-chain pLDDT
    chain_plddt = {'A': [], 'B': []}
    if 'chain_index' in scores:
        for i, ci in enumerate(scores['chain_index']):
            if i < len(plddt):
                chain_plddt[ci].append(plddt[i])
    elif plddt:
        # Guess: first ~73 residues = chain A (epitope), rest = chain B (nanobody)
        n_epitope = 73
        chain_plddt['A'] = plddt[:n_epitope]
        chain_plddt['B'] = plddt[n_epitope:]

    # Compute interface pLDDT (average of nanobody chain)
    nb_plddt = sum(chain_plddt['B']) / max(len(chain_plddt['B']), 1)

    # Compute interface PAE (between chains, not within)
    if pae is not None and 'chain_index' in scores:
        iptm_pae = None
        # PAE between chain A and B
        chain_idx = scores['chain_index']
        n_a = sum(1 for c in chain_idx if c == 'A')
        if n_a > 0 and len(chain_idx) > n_a:
            inter_pae = []
            for i in range(n_a):
                for j in range(n_a, len(chain_idx)):
                    if i < len(pae) and j < len(pae[i]):
                        inter_pae.append(pae[i][j])
            iptm_pae = sum(inter_pae) / max(len(inter_pae), 1) if inter_pae else None
    else:
        iptm_pae = None

    return {
        'plddt_mean': mean_plddt / 100.0,
        'plddt_nanobody': nb_plddt / 100.0,
        'ptm': ptm,
        'iptm': iptm,
        'max_pae': max_pae,
        'interface_pae': iptm_pae,
        'pdb_path': str(pdb_files[0]) if pdb_files else None,
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description='AF2 multimer validation for nanobody designs')
    p.add_argument('--designs', required=True, help='FASTA file with nanobody designs')
    p.add_argument('--output_dir', default='alphafold_results/aqp4_nanobody_validation',
                   help='Output directory for AF2 results')
    p.add_argument('--max_designs', type=int, default=6, help='Max designs to validate')
    args = p.parse_args()

    designs = load_designs(args.designs)
    print(f'\nLoaded {len(designs)} designs from {args.designs}')
    print(f'Validating up to {args.max_designs} designs\n')

    results = []
    for i, (header, seq) in enumerate(designs[:args.max_designs]):
        print(f'{"="*60}')
        tag = f'design_{i}'
        print(f'[{i+1}/{min(len(designs), args.max_designs)}] {tag}')
        print(f'  Nanobody: {seq[:30]}... ({len(seq)}aa)')

        out_dir = Path(args.output_dir) / tag
        t0 = time.time()

        try:
            result, result_dir = run_af2_multimer(EPITOPE, seq, out_dir, tag)
            elapsed = time.time() - t0

            if result.returncode != 0:
                print(f'  FAILED (exit {result.returncode})')
                print(f'  stderr: {result.stderr[-300:]}')
                results.append({'design': tag, 'seq': seq, 'success': False,
                               'error': result.stderr[-300:]})
                continue

            af2 = parse_af2_results(result_dir)
            if af2 is None:
                print(f'  WARNING: No scores JSON found')
                results.append({'design': tag, 'seq': seq, 'success': False,
                               'error': 'no scores JSON'})
                continue

            print(f'  Time: {elapsed:.0f}s')
            print(f'  pLDDT (overall):    {af2["plddt_mean"]:.3f}')
            print(f'  pLDDT (nanobody):   {af2["plddt_nanobody"]:.3f}')
            print(f'  pTM:                {af2["ptm"]:.3f}')
            print(f'  ipTM:               {af2["iptm"]:.3f}')
            print(f'  max PAE:            {af2["max_pae"]:.1f}')
            if af2.get('interface_pae'):
                print(f'  Interface PAE:      {af2["interface_pae"]:.1f}')

            # Quality assessment
            nb_plddt = af2['plddt_nanobody']
            iptm = af2['iptm'] or 0

            if iptm > 0.6 and nb_plddt > 0.7:
                quality = 'EXCELLENT'
            elif iptm > 0.4 and nb_plddt > 0.6:
                quality = 'GOOD'
            elif iptm > 0.2:
                quality = 'MODERATE'
            else:
                quality = 'POOR'

            print(f'  Quality: {quality}')

            results.append({
                'design': tag, 'seq': seq, 'success': True,
                **af2, 'quality': quality, 'time_s': elapsed,
            })

        except subprocess.TimeoutExpired:
            print(f'  TIMEOUT after {time.time()-t0:.0f}s')
            results.append({'design': tag, 'seq': seq, 'success': False, 'error': 'timeout'})
        except Exception as e:
            print(f'  ERROR: {e}')
            results.append({'design': tag, 'seq': seq, 'success': False, 'error': str(e)})

    # Summary
    print(f'\n{"="*70}')
    print(f'  AF2 VALIDATION SUMMARY')
    print(f'{"="*70}')
    success = [r for r in results if r.get('success')]
    print(f'  Successful: {len(success)}/{len(results)}')
    if success:
        print(f'\n  {"Design":<12} {"pLDDT(nb)":>10} {"ipTM":>8} {"pTM":>8} {"Quality":<12}')
        print(f'  {"-"*50}')
        for r in sorted(success, key=lambda x: x.get('iptm') or 0, reverse=True):
            print(f'  {r["design"]:<12} {r["plddt_nanobody"]:>10.3f} {r.get("iptm") or 0:>8.3f} {r["ptm"]:>8.3f} {r["quality"]:<12}')

    # Save
    out_path = Path(args.output_dir) / 'af2_validation_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\n  Results saved: {out_path}')


if __name__ == '__main__':
    main()
