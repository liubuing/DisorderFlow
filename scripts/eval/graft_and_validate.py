#!/usr/bin/env python
"""Graft designed CDR sequences into the 3GD8 scaffold and validate with AF2 multimer.

The design positions (from BFN design report) are scattered surface residues on the
3GD8 scaffold (an AQP4-derived binder). This script grafts each 28aa designed CDR
sequence into those positions, creating a full nanobody for AF2 validation.
"""
import os, sys, json, subprocess, time
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
VENV = PROJECT_DIR.parent / 'venv'
COLABFOLD = VENV / 'Scripts' / 'colabfold_batch.exe'

# 3GD8 scaffold sequence (from PDB chain H, standard AAs only)
SCAFFOLD_SEQ = (
    "QAFWKAVTAEFLAMLIFVLLSLGSTINWGGTEKPLPVDMVLISLCFGLSIATMVQCFGHI"
    "SGGHINPAVTVAMVCTRKISIAKSVFYIAAQCLGAIIGAGILYLVTPPSVVGGLGVTMVH"
    "GNLTAGHGLLVELIITFQLVFTIFASCDSKRTDVTGSIALAIGFSVAIGHLFAINYTGAS"
    "MNPARSFGPAVIMGNWENHWIYWVGPIIGAVLAGGLYEYVFCP"
)

# All interface-facing positions (41 total, from BFN design report)
# First 28 = BFN design region; last 13 = additional interface contacts (not designed)
ALL_INTERFACE_POSITIONS = [
    32,33,34,35, 37,38,39,40, 60,61,62, 64,65, 91,92, 149,
    157,158,159,160, 162,163,164,165, 185,186, 188,189,
    226,227, 229,230, 236, 246,247,248,249, 251,252,253,254
]
# Only the first 28 positions were designed by BFN (the CDR region)
DESIGN_POSITIONS = sorted(ALL_INTERFACE_POSITIONS[:28])

# AQP4 full sequence
AQP4_FULL = (
    "MSDRPTARRWGKCGPLCTRENIMVAFKGVWTQAFWKAVTAEFLAMLIFVLLSLGSTINWGGTEKPLPVD"
    "MVLISLCFGLSIATMVQCFGHISGGHINPAVTVAMVCTRKISIAKSVFYIAAQCLGAIIGAGILYLVTPP"
    "SVVGGLGVTMVHGNLTAGHGLLVELIITFQLVFTIFASCDSKRTDVTGSIALAIGFSVAIGHLFAINYTG"
    "ASMNPARSFGPAVIMNWENHWIYWVGPIIGAVLAGALYEYVFCPDVELKRRLKEAFSKAAQQTKGSYMEV"
    "EDNRSQVETDDLILKPGVVHVIDVDRGEELGKKVKQSDPSSH"
)

# Use C-terminal region that contains the epitope (AQP4 230-323 = 94aa)
# This is where the interface is located based on our pipeline analysis
EPITOPE_START = 230
EPITOPE = AQP4_FULL[EPITOPE_START:]  # residues 231-323 = 93aa

# AA 3-to-1 letter mapping
AA3TO1 = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
    'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
    'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y',
}


def parse_pdb_atoms(pdb_path):
    """Parse PDB atom records."""
    atoms = []
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            try:
                atoms.append({
                    'res': (line[17:20].strip(), int(line[22:26].strip()), line[21].strip()),
                    'atom': line[12:16].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'element': line[76:78].strip() or line[12:14].strip(),
                })
            except (ValueError, IndexError):
                continue
    return atoms


def load_designs(fasta_path):
    """Load sequences from FASTA."""
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


def build_residue_index(pdb_path, chain_id='H'):
    """Build mapping from PDB residue number -> 0-based sequence index."""
    atoms = parse_pdb_atoms(pdb_path)
    seen = set()
    index_map = {}  # residue_number -> 0-based index
    for a in atoms:
        res = a['res']
        if res[2] != chain_id or res[0] not in AA3TO1:
            continue
        if res[1] not in seen:
            seen.add(res[1])
            index_map[res[1]] = len(index_map)
    return index_map


def graft_cdr(scaffold_seq, design_seq, pdb_path, chain_id='H'):
    """Graft designed CDR sequence into scaffold at the design positions.

    Uses PDB residue numbers to identify which positions to replace.
    """
    idx_map = build_residue_index(pdb_path, chain_id)

    # Get sorted list of design positions with their 0-based indices
    pos_pairs = []
    for pdb_num in sorted(DESIGN_POSITIONS):
        if pdb_num in idx_map:
            pos_pairs.append((pdb_num, idx_map[pdb_num]))

    if len(pos_pairs) != len(design_seq):
        print(f"  WARNING: {len(pos_pairs)} design positions found in PDB, "
              f"but design is {len(design_seq)}aa")

    # Build grafted sequence
    grafted = list(scaffold_seq)
    for i, (pdb_num, seq_idx) in enumerate(pos_pairs):
        if i < len(design_seq):
            grafted[seq_idx] = design_seq[i]

    return ''.join(grafted)


def run_af2_multimer(aqp4_seq, nanobody_seq, output_dir, name):
    """Run AF2 multimer prediction."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = output_dir / f'{name}.fasta'
    with open(fasta_path, 'w') as f:
        f.write(f'>{name}\n{aqp4_seq}:{nanobody_seq}\n')

    cmd = [
        str(COLABFOLD),
        str(fasta_path), str(output_dir),
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


def parse_af2_scores(result_dir):
    """Parse AF2 scores JSON."""
    result_dir = Path(result_dir)
    scores_files = sorted(result_dir.glob('*_scores_rank_001_*.json'))
    pdb_files = sorted(result_dir.glob('*_rank_001_*.pdb'))
    if not scores_files:
        return None

    with open(scores_files[0]) as f:
        scores = json.load(f)

    plddt = scores.get('plddt', [0])
    mean_plddt = sum(plddt) / max(len(plddt), 1)

    # Approximate chain split: first len(epitope) residues = chain A, rest = chain B
    n_epitope = len(EPITOPE)
    chain_a_plddt = plddt[:n_epitope] if len(plddt) > n_epitope else plddt
    chain_b_plddt = plddt[n_epitope:] if len(plddt) > n_epitope else []

    a_mean = sum(chain_a_plddt) / max(len(chain_a_plddt), 1) if chain_a_plddt else 0
    b_mean = sum(chain_b_plddt) / max(len(chain_b_plddt), 1) if chain_b_plddt else 0

    return {
        'plddt_mean': mean_plddt / 100.0,
        'plddt_A': a_mean / 100.0,
        'plddt_B': b_mean / 100.0,
        'ptm': scores.get('ptm', 0),
        'iptm': scores.get('iptm'),
        'max_pae': scores.get('max_pae', 999),
        'pdb_path': str(pdb_files[0]) if pdb_files else None,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--designs', required=True, help='FASTA with nanobody CDR designs (28aa each)')
    p.add_argument('--scaffold_pdb', default='C:/biological/protein/aqp4/3GD8.pdb',
                   help='3GD8 scaffold PDB for residue numbering')
    p.add_argument('--output_dir', default='alphafold_results/aqp4_grafted_validation')
    p.add_argument('--max_designs', type=int, default=6)
    args = p.parse_args()

    designs = load_designs(args.designs)
    print(f'Loaded {len(designs)} designs from {args.designs}')
    print(f'Scaffold PDB: {args.scaffold_pdb}')
    print(f'Scaffold sequence: {len(SCAFFOLD_SEQ)}aa')
    print(f'AQP4 epitope: {EPITOPE_START+1}-{len(AQP4_FULL)} ({len(EPITOPE)}aa)')
    print(f'Design positions: {len(DESIGN_POSITIONS)} scattered residues\n')

    results = []
    for i, (header, seq) in enumerate(designs[:args.max_designs]):
        tag = f'grafted_{i}'
        print(f'{"="*60}')
        print(f'[{i+1}/{min(len(designs), args.max_designs)}] {tag}: {header[:60]}')

        # Graft CDR into scaffold
        grafted = graft_cdr(SCAFFOLD_SEQ, seq, args.scaffold_pdb, 'A')
        n_changed = sum(1 for a, b in zip(SCAFFOLD_SEQ, grafted) if a != b)
        print(f'  CDR: {seq}')
        print(f'  Grafted nanobody: {len(grafted)}aa ({n_changed} positions changed)')

        out_dir = Path(args.output_dir) / tag
        t0 = time.time()

        try:
            result, result_dir = run_af2_multimer(EPITOPE, grafted, out_dir, tag)
            elapsed = time.time() - t0

            if result.returncode != 0:
                print(f'  FAILED (exit {result.returncode}): {result.stderr[-200:]}')
                results.append({'design': tag, 'success': False, 'error': result.stderr[-300:]})
                continue

            af2 = parse_af2_scores(result_dir)
            if af2 is None:
                results.append({'design': tag, 'success': False, 'error': 'no scores'})
                continue

            iptm = af2['iptm'] or 0
            nb_plddt = af2['plddt_B']

            if iptm > 0.6 and nb_plddt > 0.7:
                quality = 'EXCELLENT'
            elif iptm > 0.4 and nb_plddt > 0.6:
                quality = 'GOOD'
            elif iptm > 0.2:
                quality = 'MODERATE'
            else:
                quality = 'POOR'

            print(f'  Time: {elapsed:.0f}s')
            print(f'  pLDDT (epitope A):   {af2["plddt_A"]:.3f}')
            print(f'  pLDDT (nanobody B):  {af2["plddt_B"]:.3f}')
            print(f'  pTM:                 {af2["ptm"]:.3f}')
            print(f'  ipTM:                {iptm:.3f}')
            print(f'  Quality: {quality}')

            results.append({'design': tag, 'seq': seq, 'grafted': grafted, 'success': True,
                           'n_changed': n_changed, 'quality': quality, 'time_s': elapsed, **af2})

        except subprocess.TimeoutExpired:
            print(f'  TIMEOUT')
            results.append({'design': tag, 'success': False, 'error': 'timeout'})
        except Exception as e:
            print(f'  ERROR: {e}')
            results.append({'design': tag, 'success': False, 'error': str(e)})

    # Summary
    print(f'\n{"="*70}')
    print(f'  GRAFTED AF2 VALIDATION SUMMARY')
    print(f'{"="*70}')
    success = [r for r in results if r.get('success')]
    print(f'  Successful: {len(success)}/{len(results)}')

    if success:
        print(f'\n  {"Design":<12} {"pLDDT(A)":>10} {"pLDDT(B)":>10} {"ipTM":>8} {"pTM":>8} {"Quality":<12}')
        print(f'  {"-"*60}')
        for r in sorted(success, key=lambda x: x.get('iptm') or 0, reverse=True):
            iptm = r.get('iptm') or 0
            print(f'  {r["design"]:<12} {r["plddt_A"]:>10.3f} {r["plddt_B"]:>10.3f} {iptm:>8.3f} {r["ptm"]:>8.3f} {r["quality"]:<12}')

    out_path = Path(args.output_dir) / 'grafted_validation_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\n  Results saved: {out_path}')


if __name__ == '__main__':
    main()
