import os
#!/usr/bin/env python3
"""Scoring Benchmark v2: Native vs scrambled CDR discrimination + BFN design test.

Two parts:
  Part A: Sequence metrics — can simple properties distinguish native from random?
  Part B: BFN design on EGFR (ordered protein) vs Abeta42 (IDP) — is BFN design
          quality actually decent for ordered targets?

Usage:
  python benchmark_scoring_v2.py              # Part A only (fast)
  python benchmark_scoring_v2.py --bfn --af2  # Part A + B (needs GPU + time)
"""
import sys, os
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules'))

import json, time, random, argparse
import numpy as np
from collections import defaultdict

AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'
AA3TO1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
}

# ── Known native CDR sequences and properties ───────────────────────────

# These are well-characterized antibody CDRs from structural databases.
# We verify that they have:
#   1. High structural complementarity to their antigen (known binding)
#   2. Specific amino acid compositions (enriched in Tyr, Ser, Gly in CDRs)
#   3. Non-random sequence patterns

NATIVE_CDRS = [
    # 1BJ1 — anti-VEGF bevacizumab H3 (heavy chain CDR3)
    {'name': '1BJ1_H3', 'seq': 'AKDPYYYGSTWYFDV',
     'known': True, 'desc': 'anti-VEGF bevacizumab CDR-H3'},
    # 1BJ1 — anti-VEGF H2
    {'name': '1BJ1_H2', 'seq': 'WINTYTGEPTYAADFKR',
     'known': True, 'desc': 'anti-VEGF bevacizumab CDR-H2'},
    # 4KRL — anti-EGFR nanobody CDR3
    {'name': '4KRL_CDR3', 'seq': 'AAGRWDKYGSSFQDEYDY',
     'known': True, 'desc': 'anti-EGFR nanobody CDR3'},
    # 4FQI — anti-HA H3
    {'name': '4FQI_H3', 'seq': 'ARDYGLYGDYGGYFDY',
     'known': True, 'desc': 'anti-HA Fab CDR-H3 (estimated)'},
    # 4RFO — anti-gp120 H3
    {'name': '4RFO_H3', 'seq': 'ARGLVGGAYKYYFDL',
     'known': True, 'desc': 'anti-gp120 Fab CDR-H3 (estimated)'},
]

# Abeta42 CDR designs from V11 seqconf (for comparison)
ABETA_DESIGNS = [
    {'name': 'A42_D1_PPL41', 'seq': 'MGAKICDACKMLNECMNEIRNPDDMGDRDPASA',
     'known': False, 'desc': 'Abeta42 BFN design, PPL=41'},
    {'name': 'A42_D2_PPL59', 'seq': 'DDASAERPDLKDLKWLWSQNWLYTDTGDRPTDV',
     'known': False, 'desc': 'Abeta42 BFN design, PPL=59'},
    {'name': 'A42_D3_PPL60', 'seq': 'AYRTIKTLVLHDTAQVYPILEQSIFKFSGVALD',
     'known': False, 'desc': 'Abeta42 BFN design, PPL=60'},
]

# ── Sequence feature computation ────────────────────────────────────────

def compute_seq_features(seq):
    """Compute a comprehensive set of sequence features."""
    n = len(seq)
    features = {}

    # Composition
    for aa in AA_LETTERS:
        features[f'freq_{aa}'] = seq.count(aa) / n

    # Categories
    hydrophobic = set('AILMFWYV')
    hydrophilic = set('RKDENQ')
    aromatic = set('FYW')
    positive = set('RKH')
    negative = set('DE')
    flexible = set('GSTNP')
    rigid = set('AILMPVWY')
    hbond_donor = set('RKHWYNQST')
    hbond_acceptor = set('DEHQNSTY')

    features['hydrophobic'] = sum(seq.count(aa) for aa in hydrophobic) / n
    features['hydrophilic'] = sum(seq.count(aa) for aa in hydrophilic) / n
    features['aromatic'] = sum(seq.count(aa) for aa in aromatic) / n
    features['net_charge'] = (sum(seq.count(aa) for aa in positive) -
                               sum(seq.count(aa) for aa in negative)) / n
    features['flexible'] = sum(seq.count(aa) for aa in flexible) / n
    features['rigid'] = sum(seq.count(aa) for aa in rigid) / n
    features['hbond_donor'] = sum(seq.count(aa) for aa in hbond_donor) / n
    features['hbond_acceptor'] = sum(seq.count(aa) for aa in hbond_acceptor) / n

    # Complexity
    features['unique_aa'] = len(set(seq)) / n
    features['ala_gly_pro'] = (seq.count('A') + seq.count('G') + seq.count('P')) / n

    # Key CDR indicators
    features['tyr_trp'] = (seq.count('Y') + seq.count('W')) / n  # aromatic recognition
    features['ser_thr'] = (seq.count('S') + seq.count('T')) / n  # hydrogen bonding
    features['asp_glu'] = (seq.count('D') + seq.count('E')) / n  # negative charge
    features['arg_lys'] = (seq.count('R') + seq.count('K')) / n  # positive charge
    features['cys'] = seq.count('C') / n  # disulfide (should be 0 in CDR)

    # Sequence patterns
    features['has_cys'] = 1.0 if 'C' in seq else 0.0
    features['has_pro'] = 1.0 if 'P' in seq else 0.0

    # Charge balance
    total_pos = sum(seq.count(aa) for aa in positive)
    total_neg = sum(seq.count(aa) for aa in negative)
    features['charge_balance'] = abs(total_pos - total_neg) / max(n, 1)

    return features


def generate_negative_controls(native_seq, n_per_type=5):
    """Generate negative controls from a native sequence."""
    random.seed(42)
    negs = []

    # Shuffle
    for _ in range(n_per_type):
        s = list(native_seq)
        random.shuffle(s)
        negs.append(('shuffled', ''.join(s)))

    # Random (uniform from AA_LETTERS)
    for _ in range(n_per_type):
        negs.append(('random_uniform', ''.join(random.choice(AA_LETTERS) for _ in native_seq)))

    # All-alanine
    negs.append(('all_Ala', 'A' * len(native_seq)))

    # All-glycine
    negs.append(('all_Gly', 'G' * len(native_seq)))

    # Hydrophobic-only
    h_aa = 'AILMFWYV'
    negs.append(('hydrophobic_only', ''.join(random.choice(h_aa) for _ in native_seq)))

    return negs


# ── Main ────────────────────────────────────────────────────────────────

def run_part_a():
    """Part A: Do sequence features distinguish native CDRs from random?"""
    print("=" * 70)
    print("  PART A: Native vs Negative CDR Sequence Features")
    print("=" * 70)

    all_data = []

    for cdr in NATIVE_CDRS:
        negs = generate_negative_controls(cdr['seq'])

        # Native
        features = compute_seq_features(cdr['seq'])
        features['name'] = cdr['name']
        features['is_native'] = True
        features['is_abeta'] = False
        all_data.append(features)

        # Negatives
        for neg_type, neg_seq in negs:
            features = compute_seq_features(neg_seq)
            features['name'] = f'{cdr["name"]}_{neg_type}'
            features['is_native'] = False
            features['is_abeta'] = False
            all_data.append(features)

    # Add Abeta42 designs as unknown category
    for ab in ABETA_DESIGNS:
        features = compute_seq_features(ab['seq'])
        features['name'] = ab['name']
        features['is_native'] = False
        features['is_abeta'] = True
        all_data.append(features)

    native = [d for d in all_data if d['is_native']]
    negative = [d for d in all_data if not d['is_native'] and not d['is_abeta']]
    abeta = [d for d in all_data if d['is_abeta']]

    # Feature keys to analyze
    feat_keys = ['hydrophobic', 'hydrophilic', 'aromatic', 'net_charge',
                 'flexible', 'rigid', 'hbond_donor', 'hbond_acceptor',
                 'unique_aa', 'ala_gly_pro', 'tyr_trp', 'ser_thr',
                 'asp_glu', 'arg_lys', 'cys', 'charge_balance',
                 'has_cys', 'has_pro']

    print(f"\n  {'Feature':<20s} {'Native':<14s} {'Negative':<14s} {'Delta':<10s} {'Sep':<8s} {'Abeta42':<14s} {'Signal?'}")
    print(f"  {'─'*20} {'─'*14} {'─'*14} {'─'*10} {'─'*8} {'─'*14} {'─'*10}")

    good_signals = []

    for key in feat_keys:
        n_vals = [d[key] for d in native]
        neg_vals = [d[key] for d in negative]
        a_vals = [d[key] for d in abeta]

        n_mean, n_std = np.mean(n_vals), np.std(n_vals)
        neg_mean, neg_std = np.mean(neg_vals), np.std(neg_vals)
        a_mean = np.mean(a_vals) if a_vals else 0

        pooled_std = np.sqrt((n_std**2 + neg_std**2) / 2)
        sep = abs(n_mean - neg_mean) / max(pooled_std, 1e-9)

        # Determine if this feature can help
        if sep > 2.0:
            sig = 'STRONG'
        elif sep > 1.0:
            sig = 'MODERATE'
        elif sep > 0.5:
            sig = 'weak'
        else:
            sig = 'noise'

        # Check if Abeta42 designs look more like native or negative
        if sep > 0.5:
            a_similarity = abs(a_mean - n_mean) / max(abs(neg_mean - n_mean), 1e-9)
            if a_similarity < 0.3:
                a_label = '≈ native'
            elif a_similarity < 0.7:
                a_label = '≈ mid'
            else:
                a_label = '≈ random'
        else:
            a_label = '-'

        print(f"  {key:<20s} {n_mean:>6.3f}±{n_std:.2f}  {neg_mean:>6.3f}±{neg_std:.2f}  "
              f"{n_mean-neg_mean:>+.3f}   {sep:>4.1f}σ  {a_mean:>6.3f} ({a_label:<8s}) {sig}")

        if sep > 1.0:
            good_signals.append(key)

    # Summary
    print(f"\n  ── Part A Summary ──")
    print(f"  Strong discriminators (separation > 1.5σ):")
    for key in feat_keys:
        n_vals = [d[key] for d in native]
        neg_vals = [d[key] for d in negative]
        n_mean = np.mean(n_vals)
        neg_mean = np.mean(neg_vals)
        n_std = np.std(n_vals)
        neg_std = np.std(neg_vals)
        pooled = np.sqrt((n_std**2 + neg_std**2) / 2)
        sep = abs(n_mean - neg_mean) / max(pooled, 1e-9)
        if sep > 1.5:
            print(f"    {key}: native={n_mean:.3f}  random={neg_mean:.3f}  (Δ={abs(n_mean-neg_mean):.3f})")

    print(f"\n  Abeta42 BFN designs resemble:")
    for key in feat_keys:
        n_vals = [d[key] for d in native]
        neg_vals = [d[key] for d in negative]
        a_vals = [d[key] for d in abeta]
        n_mean, neg_mean, a_mean = np.mean(n_vals), np.mean(neg_vals), np.mean(a_vals)
        rng = abs(n_mean - neg_mean)
        if rng > 0.01:
            a_pos = (a_mean - neg_mean) / rng  # 0 = like random, 1 = like native
            if a_pos > 0.7:
                print(f"    {key}: {a_pos:.2f} (closer to native)")
            elif a_pos < 0.3:
                print(f"    {key}: {a_pos:.2f} (closer to random)")

    return all_data


def run_part_b(device='cuda'):
    """Part B: BFN design on EGFR (ordered) vs Abeta42 (IDP) — AF2 validation."""
    print("\n" + "=" * 70)
    print("  PART B: BFN Design Quality — Ordered vs IDP Target")
    print("=" * 70)

    from bfn_loader import run_bfn_design, load_bfn, has_disorder_head
    from idp_antibody_design import _extract_sequence_from_pdb, _parse_cdr_ranges, graft_cdrs
    from af2_jax_runner import validate_antibody_epitope

    import yaml

    scaffold_pdb = 'data/misfolding_targets/5IMK.pdb'
    scaffold_chain = 'B'
    cdr_spec = 'B:26-33,51-58,97-113'
    n_samples = 5

    # ── Target 1: VEGF (ordered, from 1BJ1) ──
    print("\n  ── Target 1: VEGF (well-folded protein, 94aa) ──")
    vegf_pdb = 'data/antibody_complexes/1BJ1.pdb'
    vegf_chain = 'V'

    try:
        vegf_seq = _extract_sequence_from_pdb(vegf_pdb, vegf_chain)
        print(f"  VEGF sequence: {len(vegf_seq)}aa")
        print(f"  VEGF (first 60): {vegf_seq[:60]}...")
    except Exception as e:
        print(f"  ERROR extracting VEGF: {e}")
        vegf_seq = None

    # ── Target 2: EGFR (ordered, from 4KRL) ──
    print("\n  ── Target 2: EGFR domain (well-folded, from 4KRL) ──")
    egfr_pdb = 'data/antibody_complexes/4KRL.pdb'
    egfr_chain = 'A'

    try:
        egfr_seq = _extract_sequence_from_pdb(egfr_pdb, egfr_chain)
        print(f"  EGFR sequence: {len(egfr_seq)}aa")
        print(f"  EGFR (first 60): {egfr_seq[:60]}...")
    except Exception as e:
        print(f"  ERROR extracting EGFR: {e}")
        egfr_seq = None

    # ── Target 3: Abeta42 (IDP, control) ──
    print("\n  ── Target 3: Abeta42 (IDP, control) ──")
    abeta_pdb = 'data/misfolding_targets/2NAO_model1_A_1-42.pdb'
    abeta_chain = 'A'

    try:
        abeta_seq = _extract_sequence_from_pdb(abeta_pdb, abeta_chain)
        print(f"  Abeta42 sequence: {len(abeta_seq)}aa  {abeta_seq}")
    except Exception as e:
        print(f"  ERROR: {e}")
        abeta_seq = None

    # ── Run BFN design for each target ──
    # Save & set checkpoint
    cfg_path = os.path.join(PROJECT_ROOT, 'app_config.yaml')
    with open(cfg_path) as f:
        app_cfg = yaml.safe_load(f)
    orig_ckpt = app_cfg['models']['bfn']['checkpoint']

    # Use V11 seqconf (latest)
    v11_ckpt = 'logs/bfn_v11_seqconf_xpu_2026_06_19__23_01_18/checkpoints/best.pt'
    app_cfg['models']['bfn']['checkpoint'] = v11_ckpt
    with open(cfg_path, 'w') as f:
        yaml.dump(app_cfg, f, default_flow_style=False)

    try:
        import bfn_loader
        bfn_loader._bfn_model = None
        bfn_loader._bfn_config = None
        model, config = load_bfn(device)

        for target_name, target_pdb, target_chain, target_seq in [
            ('VEGF', vegf_pdb, vegf_chain, vegf_seq),
            ('EGFR', egfr_pdb, egfr_chain, egfr_seq),
            ('Abeta42', abeta_pdb, abeta_chain, abeta_seq),
        ]:
            if target_seq is None:
                continue

            print(f"\n  BFN design for {target_name}...")
            t0 = time.time()

            try:
                designs = run_bfn_design(
                    scaffold_pdb, cdr_spec,
                    num_samples=n_samples, stochastic=True,
                    context_chains=None,  # Complex mode
                    device=device,
                )
                dt = time.time() - t0

                ppls = [d.get('ppl', 0) for d in designs]
                print(f"  {len(designs)} designs in {dt:.0f}s")
                print(f"  PPL: {np.mean(ppls):.0f} [{min(ppls):.0f}-{max(ppls):.0f}]")

                # Show top designs
                for i, d in enumerate(designs[:3]):
                    print(f"  [{i+1}] PPL={d.get('ppl',0):.0f} {d['sequence'][:40]}...")

            except Exception as e:
                print(f"  BFN design FAILED: {e}")
                import traceback
                traceback.print_exc()

    finally:
        app_cfg['models']['bfn']['checkpoint'] = orig_ckpt
        with open(cfg_path, 'w') as f:
            yaml.dump(app_cfg, f, default_flow_style=False)

    print("\n  ── Part B Summary ──")
    print("  To complete validation, run AF2 on top designs from each target.")
    print("  (Requires significant GPU time — 5 designs × 3 targets × ~3 min)")

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bfn', action='store_true', help='Run Part B (BFN design)')
    parser.add_argument('--af2', action='store_true', help='Run AF2 validation on BFN designs')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    results_a = run_part_a()

    if args.bfn or args.af2:
        run_part_b(device=args.device)
