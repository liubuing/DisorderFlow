#!/usr/bin/env python
"""Fast AF2 validation of AQP4 nanobody CDR designs using C-terminal epitope.

Uses the 93aa C-terminal epitope (AQP4 231-323) which contains the
interface residues. Much faster than full AQP4 (546aa → 316aa complex).

Also uses colabfold_batch from venv_colabfold for reliable execution.
"""
import os, sys, json, time, subprocess, argparse, io, re
from pathlib import Path
import numpy as np

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
os.chdir(str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / 'modules'))

from target_design_helpers import extract_seq_from_pdb

# AQP4 C-terminal epitope (residues 231-323, ~93aa)
# This is the extracellular C-terminal domain that faces the antibody
AQP4_EPITOPE = (
    "VELKRRLKEAFSKAAQQTKGSYMEVEDNRSQVETDDLILKPGVVHVIDVDRGEEKKGKDQSGEVLSSV"
)

# 3GD8 scaffold full sequence (PDB chain A, standard AAs)
GD8_SCAFFOLD_SEQ = None  # Loaded from PDB at runtime

# CDR definitions for 3GD8
GD8_CDR_RANGES = [(26, 33), (51, 58), (97, 110)]


def load_designs_from_json(json_path_or_dir):
    """Load BFN designs from a results JSON or FASTA file."""
    designs = []

    # Try JSON first
    if os.path.isdir(json_path_or_dir):
        # Look for design results in directory
        for f in os.listdir(json_path_or_dir):
            if f.endswith('.json') and 'results' in f:
                json_path_or_dir = os.path.join(json_path_or_dir, f)
                break

    if json_path_or_dir.endswith('.json'):
        with open(json_path_or_dir) as f:
            data = json.load(f)

        if 'top_designs' in data:
            designs = data['top_designs']
        elif isinstance(data, list):
            designs = data
        elif 'aqp4_reference_designs' in data:
            designs = data['aqp4_reference_designs']
        elif 'categories' in data:
            for cat_name, cat in data['categories'].items():
                designs.extend(cat.get('designs', []))
    elif json_path_or_dir.endswith('.fasta') or json_path_or_dir.endswith('.fa'):
        designs = _parse_fasta(json_path_or_dir)

    return designs


def _parse_fasta(fasta_path):
    designs = []
    with open(fasta_path) as f:
        header, seq = None, ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header and seq:
                    # Parse metadata from header
                    meta = {}
                    for part in header.split():
                        if '=' in part:
                            k, v = part.split('=', 1)
                            try: meta[k] = float(v)
                            except: meta[k] = v
                    meta['sequence'] = seq
                    designs.append(meta)
                header = line[1:]
                seq = ''
            elif line and not line.startswith('#'):
                seq += line
        if header and seq:
            meta = {}
            for part in header.split():
                if '=' in part:
                    k, v = part.split('=', 1)
                    try: meta[k] = float(v)
                    except: meta[k] = v
            meta['sequence'] = seq
            designs.append(meta)
    return designs


def graft_cdr_into_scaffold(design_seq, scaffold_seq, design_region_spec):
    """Graft designed CDR sequence into the 3GD8 scaffold at specified positions."""
    # Parse design region into positions
    positions = []
    for cid, spec in re.findall(r'([A-Za-z0-9]+):([0-9,\-\s]+)', design_region_spec):
        for seg in spec.split(','):
            seg = seg.strip()
            if not seg: continue
            if '-' in seg:
                a, b = seg.split('-')
                positions.extend(range(int(a.strip()), int(b.strip()) + 1))
            else:
                positions.append(int(seg))
    positions = sorted(set(positions))

    grafted = list(scaffold_seq)
    for i, pos in enumerate(positions):
        if i < len(design_seq):
            idx = pos - 1  # Convert to 0-indexed
            if idx < len(grafted):
                grafted[idx] = design_seq[i]
            else:
                grafted.append(design_seq[i])

    return ''.join(grafted)


def run_af2_validation(ab_seqs, epitope_seq, output_dir, num_recycle=1,
                       colabfold_exe=None, max_designs=5):
    """Run AF2 multimer (colabfold_batch) on antibody-epitope pairs."""
    os.makedirs(output_dir, exist_ok=True)

    if colabfold_exe is None:
        venv = SCRIPT_DIR / 'venv_colabfold' / 'Scripts' / 'colabfold_batch.exe'
        colabfold_exe = str(venv) if venv.exists() else 'colabfold_batch'

    print(f"  AF2: {colabfold_exe}")
    print(f"  Epitope: {len(epitope_seq)}aa")
    print(f"  Designs to validate: {min(len(ab_seqs), max_designs)}")
    print(f"  Recycle: {num_recycle}")

    results = []
    design_count = 0

    for i, d in enumerate(ab_seqs):
        if design_count >= max_designs:
            break

        seq = d.get('sequence', '')
        if not seq or len(seq) < 5:
            continue

        design_count += 1
        tag = f'design_{design_count:03d}'
        design_dir = os.path.join(output_dir, tag)
        os.makedirs(design_dir, exist_ok=True)

        # Graft CDR into scaffold if we have a scaffold sequence
        if 'full_ab_seq' in d:
            full_ab = d['full_ab_seq']
        elif GD8_SCAFFOLD_SEQ:
            region = d.get('_design_region', 'A:26-33,51-58,97-110')
            full_ab = graft_cdr_into_scaffold(seq, GD8_SCAFFOLD_SEQ, region)
        else:
            full_ab = seq  # Use as-is

        # Write FASTA for AF2 multimer
        fasta_path = os.path.join(design_dir, f'{tag}.fasta')
        with open(fasta_path, 'w') as f:
            f.write(f'>{tag}\n{full_ab}:{epitope_seq}\n')

        print(f"\n  [{design_count}] {tag} | CDR={seq[:25]}...")

        cmd = [
            colabfold_exe,
            '--num-models', '1',
            '--num-recycle', str(num_recycle),
            '--model-type', 'alphafold2_multimer_v3',
            '--rank', 'iptm',
            '--stop-at-score', '80',
            fasta_path, design_dir,
        ]

        try:
            t0 = time.time()
            env = os.environ.copy()
            venv_bin = str(Path(colabfold_exe).parent)
            env['PATH'] = venv_bin + os.pathsep + env.get('PATH', '')

            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=900, cwd=str(SCRIPT_DIR), env=env)
            elapsed = time.time() - t0

            if result.returncode != 0:
                err = result.stderr[-200:] if result.stderr else 'unknown'
                print(f"    FAILED (exit {result.returncode}): {err[:100]}")
                results.append({**d, 'af2_success': False, 'af2_error': err[:200]})
                continue

            # Parse output
            scores = _parse_af2_output(design_dir, len(full_ab), len(epitope_seq))
            if scores is None:
                print(f"    FAILED: No scores output")
                results.append({**d, 'af2_success': False})
                continue

            iptm = scores['iptm'] or 0
            ptm = scores['ptm'] or 0
            plddt = scores['plddt'] or 0

            status = '✓ PASS' if iptm >= 0.80 else ('◐ OK' if iptm >= 0.60 else '✗ LOW')
            print(f"    {status} | ipTM={iptm:.4f} pTM={ptm:.4f} "
                  f"pLDDT={plddt:.3f} maxPAE={scores['max_pae']:.1f} "
                  f"({elapsed:.0f}s)")

            results.append({
                **d,
                'af2_success': True,
                'af2_iptm': iptm,
                'af2_ptm': ptm,
                'af2_plddt': plddt,
                'af2_max_pae': scores['max_pae'],
                'af2_interface_pae': scores.get('interface_pae'),
                'full_ab_seq': full_ab,
                'af2_elapsed': elapsed,
            })

        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT (15min)")
            results.append({**d, 'af2_success': False, 'af2_error': 'timeout'})
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({**d, 'af2_success': False, 'af2_error': str(e)[:200]})

    return results


def _parse_af2_output(result_dir, ab_len, epi_len):
    """Parse colabfold_batch output scores."""
    out = Path(result_dir)
    scores_files = sorted(out.glob('*scores_rank_001*.json'))
    if not scores_files:
        scores_files = sorted(out.glob('*.json'))
        scores_files = [f for f in scores_files if 'scores' in f.name]

    if not scores_files:
        return None

    with open(scores_files[0]) as f:
        scores = json.load(f)

    plddt_array = scores.get('plddt', [0])
    mean_plddt = sum(plddt_array) / max(len(plddt_array), 1) / 100.0

    ptm = scores.get('ptm', 0.0)
    iptm = scores.get('iptm', 0.0)
    if iptm == 0.0 or iptm is None:
        iptm = ptm  # Fallback

    max_pae = scores.get('max_pae', 999.0)

    # Interface PAE
    pae = scores.get('pae', None)
    if pae is not None:
        pae_arr = np.array(pae)
        total = pae_arr.shape[0]
        epi = total - ab_len
        if epi > 0 and ab_len < total:
            interface_pae = float(pae_arr[:ab_len, ab_len:].mean())
        else:
            interface_pae = None
    else:
        interface_pae = None

    return {
        'plddt': mean_plddt,
        'ptm': ptm,
        'iptm': iptm,
        'max_pae': max_pae,
        'interface_pae': interface_pae,
        'scores_file': str(scores_files[0]),
    }


def main():
    parser = argparse.ArgumentParser(description='Fast AF2 validation of AQP4 nanobody designs')
    parser.add_argument('--designs', required=True,
                       help='JSON or FASTA file with BFN designs')
    parser.add_argument('--scaffold-pdb', default='C:/biological/protein/aqp4/3GD8.pdb',
                       help='3GD8 scaffold PDB for CDR grafting')
    parser.add_argument('--design-region', default='A:26-33,51-58,97-110',
                       help='CDR design region spec')
    parser.add_argument('--epitope', default=None,
                       help='Epitope sequence (default: AQP4 C-term 93aa)')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--num-recycle', type=int, default=1,
                       help='AF2 recycles (default: 1 for speed)')
    parser.add_argument('--max-designs', type=int, default=3,
                       help='Max designs to validate (default: 3)')
    args = parser.parse_args()

    global GD8_SCAFFOLD_SEQ
    if os.path.exists(args.scaffold_pdb):
        GD8_SCAFFOLD_SEQ = extract_seq_from_pdb(args.scaffold_pdb, 'A')
        print(f"Scaffold: {len(GD8_SCAFFOLD_SEQ)}aa (3GD8)")

    epitope_seq = args.epitope or AQP4_EPITOPE
    print(f"Epitope: {len(epitope_seq)}aa (AQP4 C-terminal)")

    if args.output_dir is None:
        args.output_dir = f'design_results/aqp4_fast_af2_{int(time.time())}'

    # Load designs
    designs = load_designs_from_json(args.designs)
    print(f"Loaded {len(designs)} designs from {args.designs}")

    # Add design region to each design
    for d in designs:
        d['_design_region'] = args.design_region

    # Run AF2 validation
    results = run_af2_validation(
        designs, epitope_seq, args.output_dir,
        num_recycle=args.num_recycle,
        max_designs=args.max_designs,
    )

    # Summary
    successful = [r for r in results if r.get('af2_success')]
    pass_08 = [r for r in successful if (r.get('af2_iptm') or 0) >= 0.80]

    print(f"\n{'='*60}")
    print(f"  FAST AF2 VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"  Validated: {len(successful)}/{len(results)} successful")
    print(f"  ipTM >= 0.80: {len(pass_08)} designs ★")

    if pass_08:
        print(f"\n  PASSING DESIGNS (≥0.80):")
        for r in sorted(pass_08, key=lambda x: x.get('af2_iptm', 0) or 0, reverse=True):
            print(f"    ipTM={r['af2_iptm']:.4f} pLDDT={r['af2_plddt']:.3f} | {r.get('sequence','')[:40]}")

    # Save results
    out_json = os.path.join(args.output_dir, 'af2_validation_results.json')
    with open(out_json, 'w') as f:
        json.dump([{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in r.items() if k != '_design_region'}
                   for r in results], f, indent=2, default=str)
    print(f"\n  Results: {out_json}")

    # FASTA for passing designs
    if pass_08:
        fasta_out = os.path.join(args.output_dir, 'af2_iptm08_passing.fasta')
        with open(fasta_out, 'w') as f:
            for r in sorted(pass_08, key=lambda x: x.get('af2_iptm',0) or 0, reverse=True):
                f.write(f">pass AF2_ipTM={r['af2_iptm']:.4f} pLDDT={r['af2_plddt']:.3f}\n")
                f.write(f"{r.get('sequence','')}\n")
        print(f"  FASTA: {fasta_out} ({len(pass_08)} designs)")


if __name__ == '__main__':
    main()
