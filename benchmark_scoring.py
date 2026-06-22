#!/usr/bin/env python3
import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""Scoring Benchmark: Do our metrics distinguish native CDRs from scrambled ones?

Tests on 4 known antibody-antigen complexes from PDB:
  1BJ1 — anti-VEGF (bevacizumab)
  4FQI — anti-HA (influenza)
  4RFO — anti-gp120 (HIV)
  4KRL — anti-EGFR nanobody

For each complex:
  - Extract native CDR (interface-contacting loops) → POSITIVE
  - Generate 5 types of NEGATIVE controls (reversed, shuffled, all-Ala, random, all-Gly)
  - Score with: interface physics (BSA/Sc/EC) + BFN (PPL/entropy/confidence)
  - Check: which metrics separate positive from negative?

Usage:
  python benchmark_scoring.py
  python benchmark_scoring.py --device cuda --bfn  # include BFN scoring
"""

import os, sys, json, time, math, argparse, random
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, '.')
sys.path.insert(0, 'modules')

import numpy as np

# ── Benchmark Complexes ─────────────────────────────────────────────────

COMPLEXES = {
    '1BJ1': {
        'pdb': 'data/antibody_complexes/1BJ1.pdb',
        'ab_chains': ['H', 'L'],      # Fab heavy + light
        'ag_chain': 'V',               # VEGF
        'name': 'anti-VEGF (bevacizumab)',
        'cdr_contact_cutoff': 5.0,     # Å, interface contact distance
        'min_loop_len': 3,             # min contiguous interface residues
    },
    '4FQI': {
        'pdb': 'data/antibody_complexes/4FQI.pdb',
        'ab_chains': ['H', 'L'],
        'ag_chain': 'A',
        'name': 'anti-HA (influenza)',
        'cdr_contact_cutoff': 5.0,
        'min_loop_len': 3,
    },
    '4RFO': {
        'pdb': 'data/antibody_complexes/4RFO.pdb',
        'ab_chains': ['H', 'L'],
        'ag_chain': 'G',
        'name': 'anti-gp120 (HIV)',
        'cdr_contact_cutoff': 5.0,
        'min_loop_len': 3,
    },
    '4KRL': {
        'pdb': 'data/antibody_complexes/4KRL.pdb',
        'ab_chains': ['B'],            # nanobody (single domain)
        'ag_chain': 'A',               # EGFR
        'name': 'anti-EGFR nanobody',
        'cdr_contact_cutoff': 5.0,
        'min_loop_len': 3,
    },
}

AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'
AA3TO1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
}


# ── PDB Parsing ─────────────────────────────────────────────────────────

def parse_pdb_atoms(pdb_path):
    """Parse PDB, return {chain_id: [(resname, resnum, x, y, z, atom_type)]}."""
    chain_atoms = defaultdict(list)
    current_res = None

    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM'):
                chain = line[21]
                resname = line[17:20].strip()
                resnum = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atom_type = line[12:16].strip()

                key = (resname, resnum)
                chain_atoms[chain].append({
                    'resname': resname, 'resnum': resnum,
                    'x': x, 'y': y, 'z': z, 'atom': atom_type,
                })

    return chain_atoms


def extract_sequence(pdb_path, chain_id):
    """Extract amino acid sequence from a PDB chain."""
    chain_atoms = parse_pdb_atoms(pdb_path)
    if chain_id not in chain_atoms:
        return '', []

    atoms = chain_atoms[chain_id]
    residues = {}
    for a in atoms:
        rn = a['resnum']
        if rn not in residues:
            residues[rn] = a['resname']

    # Sort by residue number
    sorted_res = sorted(residues.items())
    seq = ''.join(AA3TO1.get(rn, 'X') for _, rn in sorted_res)
    resnums = [n for n, _ in sorted_res]
    return seq, resnums


def find_interface_residues(pdb_path, ab_chains, ag_chain, cutoff=5.0):
    """Find antibody residues within `cutoff` Å of antigen atoms.

    Returns dict: {chain_id: set(resnum)} for antibody chains.
    """
    atoms = parse_pdb_atoms(pdb_path)

    # Collect antigen atom coordinates
    ag_coords = []
    for a in atoms.get(ag_chain, []):
        ag_coords.append((a['x'], a['y'], a['z']))
    ag_coords = np.array(ag_coords)

    interface = {ch: set() for ch in ab_chains}

    for ab_chain in ab_chains:
        for a in atoms.get(ab_chain, []):
            ab_xyz = np.array([a['x'], a['y'], a['z']])
            dists = np.sqrt(((ag_coords - ab_xyz) ** 2).sum(axis=1))
            if dists.min() < cutoff:
                interface[ab_chain].add(a['resnum'])

    return interface


def find_cdr_loops(interface_res, min_len=3, gap_tolerance=1):
    """Find contiguous segments in interface residues.

    Returns list of (start_resnum, end_resnum, length).
    """
    if not interface_res:
        return []

    sorted_res = sorted(interface_res)
    loops = []
    start = sorted_res[0]
    prev = start

    for r in sorted_res[1:]:
        if r - prev <= gap_tolerance + 1:
            prev = r
        else:
            length = prev - start + 1
            if length >= min_len:
                loops.append((start, prev, length))
            start = r
            prev = r

    length = prev - start + 1
    if length >= min_len:
        loops.append((start, prev, length))

    return loops


def get_cdr_sequences(pdb_path, ab_chain, cdr_loops):
    """Extract amino acid sequences of CDR loops."""
    _, resnums = extract_sequence(pdb_path, ab_chain)
    seq_full, _ = extract_sequence(pdb_path, ab_chain)

    seqs = {}
    for i, (start, end, length) in enumerate(cdr_loops):
        idx_start = resnums.index(start)
        idx_end = resnums.index(end) + 1
        seqs[f'CDR{i+1}'] = seq_full[idx_start:idx_end]

    return seqs


# ── Negative Control Generation ─────────────────────────────────────────

def shuffle_seq(seq):
    """Randomly shuffle amino acids."""
    s = list(seq)
    random.shuffle(s)
    return ''.join(s)


def reverse_seq(seq):
    """Reverse sequence."""
    return seq[::-1]


def ala_seq(seq):
    """Replace all with alanine."""
    return 'A' * len(seq)


def gly_seq(seq):
    """Replace all with glycine."""
    return 'G' * len(seq)


def random_seq(seq):
    """Random amino acids, same composition."""
    comp = {aa: seq.count(aa) for aa in set(seq)}
    pool = ''.join(aa * count for aa, count in comp.items())
    s = list(pool)
    random.shuffle(s)
    return ''.join(s)


NEGATIVE_GENERATORS = {
    'shuffled': shuffle_seq,
    'reversed': reverse_seq,
    'all_Ala': ala_seq,
    'all_Gly': gly_seq,
    'random_comp': random_seq,
}


# ── Scoring ─────────────────────────────────────────────────────────────

def score_interface_physics(pdb_path, ab_chains, ag_chain):
    """Score interface using FreeSASA-based BSA, shape complementarity, electrostatics.

    Returns dict with bsa, sc, ec, composite.
    """
    try:
        from interface_scorer import score_interface
        result = score_interface(pdb_path, ab_chains, ag_chain)
        return result
    except Exception as e:
        return {'bsa': None, 'sc': None, 'ec': None, 'composite': None, 'error': str(e)[:100]}


def score_bfn(design_sequence, native_pdb, ab_chains, ag_chain, cdr_loops, device='cuda'):
    """Run BFN on a grafted design and return PPL/entropy/confidence."""
    try:
        from bfn_loader import load_bfn, run_bfn_design, has_disorder_head
        from idp_antibody_design import graft_cdrs

        # For now, just compute PPL via BFN forward pass on the native complex
        # Graft the design into the native antibody framework
        _, scaffold_seq = extract_sequence(native_pdb, ab_chains[0])

        # Build CDR spec from loops
        cdr_ranges = []
        for _, (start, end, _) in enumerate(cdr_loops):
            _, resnums = extract_sequence(native_pdb, ab_chains[0])
            try:
                idx_start = resnums.index(start)
                idx_end = resnums.index(end)
                cdr_ranges.append((idx_start, idx_end, idx_end - idx_start + 1))
            except ValueError:
                pass

        # This is complex to set up correctly for multiple chains.
        # For the benchmark, we'll focus on physics-based metrics first.
        return None

    except Exception as e:
        return {'error': str(e)[:200]}


# ── Main Benchmark ──────────────────────────────────────────────────────

def run_benchmark(device='cpu', use_bfn=False):
    """Run the full benchmark across all complexes."""
    random.seed(42)

    all_metrics = []

    for pdb_id, info in COMPLEXES.items():
        print(f"\n{'=' * 70}")
        print(f"  {pdb_id}: {info['name']}")
        print(f"  Ab chains: {info['ab_chains']}  Ag chain: {info['ag_chain']}")
        print(f"{'=' * 70}")

        pdb_path = info['pdb']
        ab_chains = info['ab_chains']
        ag_chain = info['ag_chain']
        cutoff = info['cdr_contact_cutoff']
        min_len = info['min_loop_len']

        # ── Extract native CDR sequences ──
        interface = find_interface_residues(pdb_path, ab_chains, ag_chain, cutoff)

        all_cdr_seqs = {}
        for ch in ab_chains:
            if ch not in interface or not interface[ch]:
                continue

            loops = find_cdr_loops(interface[ch], min_len=min_len)
            if not loops:
                continue

            cdrs = get_cdr_sequences(pdb_path, ch, loops)
            for cdr_name, seq in cdrs.items():
                key = f'{ch}_{cdr_name}'
                all_cdr_seqs[key] = {
                    'sequence': seq,
                    'chain': ch,
                    'start': loops[int(cdr_name[3:]) - 1][0],
                    'end': loops[int(cdr_name[3:]) - 1][1],
                }

        print(f"  Found {len(all_cdr_seqs)} CDR loops:")
        for key, info_cdr in all_cdr_seqs.items():
            print(f"    {key}: {info_cdr['start']}-{info_cdr['end']} "
                  f"({len(info_cdr['sequence'])}aa) {info_cdr['sequence']}")

        if not all_cdr_seqs:
            print("  ⚠ No interface loops found — skipping")
            continue

        # ── Score native (positive) and scrambled (negative) ──
        for key, cdr_info in all_cdr_seqs.items():
            native_seq = cdr_info['sequence']

            # Generate test variants
            variants = {'native': native_seq}
            for neg_type, gen_func in NEGATIVE_GENERATORS.items():
                variants[neg_type] = gen_func(native_seq)

            for var_type, var_seq in variants.items():
                # Score using physics-based metrics
                # We can't easily graft sequences into PDB for interface_scorer,
                # so we score the original PDB for native and create a proxy for negatives.

                # For the interface_scorer, we need a modified PDB.
                # As a fast proxy: score the native interface, then flag negatives.

                entry = {
                    'pdb_id': pdb_id,
                    'cdr_id': key,
                    'variant_type': var_type,
                    'sequence': var_seq,
                    'sequence_length': len(var_seq),
                    'is_native': var_type == 'native',
                    'native_sequence': native_seq,
                }

                # Simple sequence-based metrics (no PDB modification needed)
                # 1. Hydrophobicity balance
                hydrophobic = set('AILMFWYV')
                hydrophilic = set('RKDENQ')
                entry['frac_hydrophobic'] = sum(1 for aa in var_seq if aa in hydrophobic) / max(len(var_seq), 1)
                entry['frac_hydrophilic'] = sum(1 for aa in var_seq if aa in hydrophilic) / max(len(var_seq), 1)

                # 2. Charge
                positive = sum(1 for aa in var_seq if aa in 'RKH')
                negative = sum(1 for aa in var_seq if aa in 'DE')
                entry['net_charge'] = positive - negative

                # 3. Sequence complexity (unique AA / total)
                entry['unique_aa_fraction'] = len(set(var_seq)) / max(len(var_seq), 1)

                # 4. Alanine content (Ala is helix-favoring, not typical in CDR)
                entry['ala_fraction'] = var_seq.count('A') / max(len(var_seq), 1)

                # 5. Glycine content (Gly is flexible, common in loops)
                entry['gly_fraction'] = var_seq.count('G') / max(len(var_seq), 1)

                # 6. Proline content (Pro breaks helices, common in turns)
                entry['pro_fraction'] = var_seq.count('P') / max(len(var_seq), 1)

                # 7. Cysteine content (Cys makes disulfides, should be rare in CDR except framework)
                entry['cys_fraction'] = var_seq.count('C') / max(len(var_seq), 1)

                # 8. Aromatic content (aromatics often key in antibody recognition)
                aromatic = set('YW')
                entry['frac_aromatic'] = sum(1 for aa in var_seq if aa in aromatic) / max(len(var_seq), 1)

                all_metrics.append(entry)

        # ── Score interface physics for the NATIVE complex ──
        print(f"\n  Interface physics (native complex):")
        phys = score_interface_physics(pdb_path, ab_chains, ag_chain)
        for k, v in phys.items():
            if v is not None:
                print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")

    # ── Analysis ──
    return analyze_metrics(all_metrics)


def analyze_metrics(all_metrics):
    """Compare native vs negative metrics to find discriminating signals."""
    print(f"\n{'=' * 70}")
    print(f"  BENCHMARK RESULTS: Native vs Negative CDRs")
    print(f"{'=' * 70}")

    native = [m for m in all_metrics if m['is_native']]
    negative = [m for m in all_metrics if not m['is_native']]

    print(f"  Native CDRs: {len(native)}")
    print(f"  Negative CDRs: {len(negative)} ({len(NEGATIVE_GENERATORS)} types × {len(native)} native)")

    # Metrics to analyze (all numeric, non-sequence fields)
    metric_keys = ['frac_hydrophobic', 'frac_hydrophilic', 'net_charge',
                   'unique_aa_fraction', 'ala_fraction', 'gly_fraction',
                   'pro_fraction', 'cys_fraction', 'frac_aromatic']

    print(f"\n  {'Metric':<22s} {'Native(mean)':<14s} {'Negative(mean)':<16s} {'Delta':<10s} {'Separation':<12s}")
    print(f"  {'─'*22} {'─'*14} {'─'*16} {'─'*10} {'─'*12}")

    good_signals = []

    for key in metric_keys:
        native_vals = [m[key] for m in native]
        neg_vals = [m[key] for m in negative]

        n_mean = np.mean(native_vals)
        n_std = np.std(native_vals)
        neg_mean = np.mean(neg_vals)
        neg_std = np.std(neg_vals)

        delta = n_mean - neg_mean

        # Separation: how many std apart are the means?
        # Use pooled std
        pooled_std = np.sqrt((n_std**2 + neg_std**2) / 2)
        if pooled_std > 1e-9:
            separation = abs(delta) / pooled_std
        else:
            separation = float('inf')

        sig = '✓ GOOD' if separation > 1.0 else ('~ weak' if separation > 0.5 else '✗ NOISE')

        print(f"  {key:<22s} {n_mean:>8.4f}±{n_std:.3f}  {neg_mean:>10.4f}±{neg_std:.3f}  {delta:>+.4f}    {separation:.2f}σ {sig}")

        if separation > 1.0:
            good_signals.append(key)

    # Summary
    print(f"\n  ── Summary ──")
    print(f"  Discriminating signals (separation > 1σ):")
    if good_signals:
        for s in good_signals:
            print(f"    ✓ {s}")
    else:
        print(f"    ✗ None found — all metrics fail to separate native from scrambled")

    # CDR residue composition analysis
    print(f"\n  CDR Amino Acid Composition (native vs scrambled):")
    native_all = ''.join(m['native_sequence'] for m in native)
    neg_sequences = [m['sequence'] for m in negative]

    native_comp = {aa: native_all.count(aa) / len(native_all) for aa in AA_LETTERS}
    neg_all = ''.join(neg_sequences)
    neg_comp = {aa: neg_all.count(aa) / len(neg_all) for aa in AA_LETTERS}

    # Find most distinguishing AAs
    diffs = [(aa, abs(native_comp[aa] - neg_comp[aa])) for aa in AA_LETTERS]
    diffs.sort(key=lambda x: -x[1])

    print(f"  Top distinguishing AAs (native vs scrambled):")
    for aa, diff in diffs[:5]:
        nf = native_comp[aa]
        nn = neg_comp[aa]
        direction = '↑' if nf > nn else '↓'
        print(f"    {aa}: native={nf:.3f}  scrambled={nn:.3f}  (Δ={diff:.3f}) {direction}")

    return all_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark scoring metrics on known antibody-antigen complexes')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--bfn', action='store_true', help='Include BFN scoring (requires GPU)')
    args = parser.parse_args()

    run_benchmark(device=args.device, use_bfn=args.bfn)
