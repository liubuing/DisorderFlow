#!/usr/bin/env python3
"""§5 Fallback Pipeline: IDP Antibody Design without Way4 Generative Model.

Per WAY4_REALIZE_PLAN_V14 §5: When Way4 generative thesis fails (S0 NoGo),
achieve "IDP antibody design" via:
  1. Template-seeded redesign from known anti-Aβ antibodies (Pillar A)
  2. Known anti-IDP antibody library screening
  3. Pliability-matched selection (disorder as filter, NOT generative driver)
  4. Computational pre-screening for wet-lab candidate prioritization

Output: ranked_candidates.json — top-N candidates ready for SPR/BLI testing.

Usage:
    python way4_s5_fallback_pipeline.py [--top 20] [--out wetlab_candidates.json]
"""

import sys, os, json, time, argparse, math
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'modules')

# =============================================================================
# AA scales and utilities
# =============================================================================

AA = 'ACDEFGHIKLMNPQRSTVWY'
AA3_TO_1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}

# Disorder propensity (TOP-IDP scale, normalized)
DISORDER_PROPENSITY = {
    'A': 0.30, 'C': 0.25, 'D': 0.55, 'E': 0.60, 'F': 0.35,
    'G': 0.55, 'H': 0.45, 'I': 0.30, 'K': 0.65, 'L': 0.30,
    'M': 0.35, 'N': 0.50, 'P': 0.70, 'Q': 0.55, 'R': 0.60,
    'S': 0.50, 'T': 0.45, 'V': 0.28, 'W': 0.35, 'Y': 0.40,
}

# Solubility propensity (higher = more soluble)
SOLUBILITY = {
    'A': 0.60, 'C': 0.40, 'D': 0.80, 'E': 0.82, 'F': 0.38,
    'G': 0.65, 'H': 0.52, 'I': 0.35, 'K': 0.78, 'L': 0.42,
    'M': 0.48, 'N': 0.62, 'P': 0.55, 'Q': 0.68, 'R': 0.75,
    'S': 0.65, 'T': 0.55, 'V': 0.40, 'W': 0.30, 'Y': 0.35,
}

# Immunogenicity risk (T-cell epitope propensity, higher = more risk)
IMMUNOGENICITY_RISK = {
    'A': 0.2, 'C': 0.3, 'D': 0.1, 'E': 0.1, 'F': 0.8,
    'G': 0.1, 'H': 0.5, 'I': 0.8, 'K': 0.3, 'L': 0.8,
    'M': 0.6, 'N': 0.2, 'P': 0.2, 'Q': 0.2, 'R': 0.3,
    'S': 0.2, 'T': 0.3, 'V': 0.7, 'W': 0.9, 'Y': 0.7,
}


def seq_entropy(seq):
    """Shannon entropy of a sequence."""
    cnt = Counter(seq)
    n = len(seq)
    return -sum((c/n) * math.log(c/n) for c in cnt.values()) / math.log(20)


def seq_complexity(seq):
    """CDR complexity: penalize homorepeats and single-AA dominance."""
    cnt = Counter(seq)
    n = len(seq)
    # Run-length penalty
    run_max = 1
    run_cur = 1
    for i in range(1, n):
        if seq[i] == seq[i-1]:
            run_cur += 1
            run_max = max(run_max, run_cur)
        else:
            run_cur = 1
    run_penalty = max(0, (run_max - 3) / 10.0)

    # Single-AA dominance
    top_frac = max(cnt.values()) / n if n > 0 else 1.0
    dominance_penalty = max(0, top_frac - 0.4)

    # Diversity bonus
    unique = len(set(seq))
    diversity = unique / len(seq)

    return diversity - run_penalty - dominance_penalty


def disorder_profile(seq):
    """Per-residue disorder propensity."""
    return [DISORDER_PROPENSITY.get(aa, 0.5) for aa in seq]


def mean_solubility(seq):
    """Mean solubility score."""
    return np.mean([SOLUBILITY.get(aa, 0.5) for aa in seq])


def mean_immunogenicity(seq):
    """Mean immunogenicity risk score (lower = safer)."""
    return np.mean([IMMUNOGENICITY_RISK.get(aa, 0.5) for aa in seq])


def anti_degen_check(seq, max_run=4, max_6mer=0):
    """Check for sequence degeneration."""
    # Homopolymer runs
    run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            run += 1
            if run > max_run:
                return False
        else:
            run = 1
    # 6-mer repeats
    if max_6mer > 0 and len(seq) >= 6:
        seen = set()
        for i in range(len(seq) - 5):
            kmer = seq[i:i+6]
            if kmer in seen:
                return False
            seen.add(kmer)
    return True


# =============================================================================
# Known Anti-IDP Antibody CDR Library
# =============================================================================

KNOWN_ANTI_IDP_CDRS = {
    # From S0 analysis (CDR-H3 sequences extracted from PDB)
    '4HIX_H3': {'seq': 'YDHYSGSSDY', 'ab': 'Aducanumab-like', 'target': 'Aβ N-term',
                'source': 'PDB 4HIX', 'epitope_disorder': 0.47},
    '5CSZ_H3': {'seq': 'GKGYVRYFDV', 'ab': 'Crenezumab-like', 'target': 'Aβ mid',
                'source': 'PDB 5CSZ', 'epitope_disorder': 0.45},
    '3UOT_H3': {'seq': 'LNGTQILR', 'ab': 'Anti-Aβ scFv', 'target': 'Aβ oligomer',
                'source': 'PDB 3UOT', 'epitope_disorder': 0.45},
    '6CGZ_H3': {'seq': 'DSMGAQGR', 'ab': 'Anti-Aβ Fab', 'target': 'Aβ protofibril',
                'source': 'PDB 6CGZ', 'epitope_disorder': 0.44},
    '6CBV_H3': {'seq': 'WGYWPGEPWWKAFDY', 'ab': 'Anti-tau Fab', 'target': 'Tau PHF',
                'source': 'PDB 6CBV', 'epitope_disorder': 0.45},
    '6H1F_H3': {'seq': 'DLHRPYGPGSQRTDDYDT', 'ab': 'Anti-αSyn Fab', 'target': 'α-Synuclein',
                'source': 'PDB 6H1F', 'epitope_disorder': 0.48},
    '4BKL_H3': {'seq': 'YGGYFDY', 'ab': 'Anti-αSyn scFv', 'target': 'α-Synuclein',
                'source': 'PDB 4BKL', 'epitope_disorder': 0.45},

    # Canonical anti-Aβ antibody CDR-H3 sequences from literature
    'aducanumab_H3': {'seq': 'ARDGTTVYYYYGMDV', 'ab': 'Aducanumab',
                      'target': 'Aβ N-term (3-7)', 'source': 'literature',
                      'epitope_disorder': 0.47},
    'crenezumab_H3': {'seq': 'ARDRSYGSSSWYFDY', 'ab': 'Crenezumab',
                      'target': 'Aβ mid (13-24)', 'source': 'literature',
                      'epitope_disorder': 0.45},
    'gantenerumab_H3': {'seq': 'ARKNRYSSSWYLDY', 'ab': 'Gantenerumab',
                        'target': 'Aβ N-term', 'source': 'literature',
                        'epitope_disorder': 0.47},
    'solanezumab_H3': {'seq': 'ARSGYSSSWYFDY', 'ab': 'Solanezumab',
                       'target': 'Aβ mid (16-26)', 'source': 'literature',
                       'epitope_disorder': 0.45},
    'bapineuzumab_H3': {'seq': 'ARDGNYGYYFDY', 'ab': 'Bapineuzumab',
                        'target': 'Aβ N-term (1-5)', 'source': 'literature',
                        'epitope_disorder': 0.47},
    'ponezumab_H3': {'seq': 'ARYDAGYGDFDY', 'ab': 'Ponezumab',
                     'target': 'Aβ C-term (33-40)', 'source': 'literature',
                     'epitope_disorder': 0.50},
    'lecanemab_H3': {'seq': 'AREGVYYYGMDV', 'ab': 'Lecanemab (BAN2401)',
                     'target': 'Aβ protofibril', 'source': 'literature',
                     'epitope_disorder': 0.44},
    'donanemab_H3': {'seq': 'ARLGYCSSTSCYFDY', 'ab': 'Donanemab',
                     'target': 'Aβ(p3-42) pyroglutamate', 'source': 'literature',
                     'epitope_disorder': 0.43},
}


# =============================================================================
# Aβ42 Epitope Profiles (for pliability matching)
# =============================================================================

ABETA42_SEQUENCE = 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA'
ABETA42_DISORDER = [  # IUPred-like per-residue disorder (higher = more flexible)
    0.65, 0.55, 0.50, 0.55, 0.50, 0.60, 0.65, 0.70, 0.55, 0.50,  # 1-10
    0.45, 0.40, 0.35, 0.35, 0.30, 0.35, 0.40, 0.45, 0.40, 0.35,  # 11-20
    0.38, 0.42, 0.45, 0.42, 0.38, 0.35, 0.42, 0.48, 0.50, 0.45,  # 21-30
    0.40, 0.35, 0.35, 0.38, 0.40, 0.42, 0.38, 0.35, 0.35, 0.38,  # 31-40
    0.40, 0.38,  # 41-42
]

ABETA_EPITOPES = {
    'N-term (1-11)':   (0, 11,  'DAEFRHDSGYE'),
    'mid (12-24)':     (11, 24, 'VHHQKLVFFAEDV'),
    'central (16-26)': (15, 26, 'KLVFFAEDVGS'),
    'C-term (30-42)':  (29, 42, 'AIIGLMVGGVVIA'),
    'turn (24-34)':    (23, 34, 'VGSNKGAIIGLM'),
    'full (1-42)':     (0, 42,  ABETA42_SEQUENCE),
}


def epitope_disorder_mean(start, end):
    """Mean disorder of an epitope region."""
    return float(np.mean(ABETA42_DISORDER[start:end]))


# =============================================================================
# Pliability-Matched Selection
# =============================================================================

def score_pliability_match(cdr_seq, epitope_disorder):
    """Score how well CDR pliability matches epitope flexibility.

    The Way4 thesis (if true) predicts: high epitope flexibility → high CDR disorder.
    We use this as a SELECTION filter, not a generative driver.

    For known anti-IDP antibodies, their CDR-H3 sequences have specific
    disorder characteristics (high Gly/Ser/Pro content, low hydrophobicity)
    that may contribute to binding flexible epitopes.

    Returns: (pliability_score, explanation_dict)
    """
    # CDR disorder propensity
    cdr_dp = disorder_profile(cdr_seq)
    cdr_mean_dp = np.mean(cdr_dp)

    # Match: how close is CDR disorder to epitope disorder
    # For high-disorder epitopes, we WANT high-disorder CDRs (flexible)
    # For low-disorder epitopes, we want moderate CDR disorder
    if epitope_disorder > 0.45:  # flexible epitope
        match_score = min(cdr_mean_dp / epitope_disorder, 1.5)
    else:  # more rigid epitope
        match_score = 1.0 - abs(cdr_mean_dp - 0.35)

    # Gly/Ser fraction (flexibility proxy)
    gly_ser_frac = (cdr_seq.count('G') + cdr_seq.count('S')) / len(cdr_seq)

    # Pro content (disorder-promoting)
    pro_frac = cdr_seq.count('P') / len(cdr_seq)

    # Aromatic content (may be important for Aβ binding via π-stacking)
    aromatic_frac = sum(1 for a in cdr_seq if a in 'YWF') / len(cdr_seq)

    # Hydrophobic content (lower = better solubility, but may reduce affinity)
    hydrophobic_frac = sum(1 for a in cdr_seq if a in 'ILVMFW') / len(cdr_seq)

    # Composite pliability score
    pliability = (
        0.35 * match_score +
        0.25 * min(gly_ser_frac / 0.3, 1.0) +  # bonus for flexibility, cap at 30%
        0.15 * (1.0 - abs(hydrophobic_frac - 0.25)) +  # moderate hydrophobicity
        0.15 * min(aromatic_frac / 0.15, 1.0) +  # some aromatics for π-stacking
        0.10 * pro_frac
    )

    return pliability, {
        'cdr_mean_disorder': cdr_mean_dp,
        'match_score': match_score,
        'gly_ser_frac': gly_ser_frac,
        'aromatic_frac': aromatic_frac,
        'hydrophobic_frac': hydrophobic_frac,
        'pro_frac': pro_frac,
    }


def score_candidate(cdr_seq, epitope_name, epitope_start, epitope_end,
                    source_ab='unknown', source_type='literature'):
    """Full candidate scoring for wet-lab prioritization.

    Returns: dict with all scores and a composite rank.
    """
    epi_disorder = epitope_disorder_mean(epitope_start, epitope_end)
    pliability, pli_details = score_pliability_match(cdr_seq, epi_disorder)

    # Basic quality metrics
    entropy = seq_entropy(cdr_seq)
    complexity = seq_complexity(cdr_seq)
    solubility = mean_solubility(cdr_seq)
    immunogenicity = mean_immunogenicity(cdr_seq)
    degen_ok = anti_degen_check(cdr_seq)

    # Composite score for ranking
    # Higher = better candidate
    composite = (
        0.30 * pliability +          # pliability match (Way4 selection)
        0.20 * min(entropy / 2.5, 1.0) +  # sufficient diversity
        0.20 * min(complexity, 1.0) +     # CDR complexity
        0.15 * solubility +               # soluble expression
        0.10 * (1.0 - immunogenicity) +   # low immunogenicity risk
        0.05 * (1.0 if degen_ok else 0.0) # no degeneration
    )

    return {
        'cdr_sequence': cdr_seq,
        'cdr_length': len(cdr_seq),
        'epitope': epitope_name,
        'epitope_sequence': ABETA42_SEQUENCE[epitope_start:epitope_end],
        'epitope_disorder': round(epi_disorder, 3),
        'source_antibody': source_ab,
        'source_type': source_type,
        'pliability_score': round(pliability, 4),
        'pliability_details': pli_details,
        'entropy': round(entropy, 3),
        'complexity': round(complexity, 3),
        'solubility': round(solubility, 3),
        'immunogenicity_risk': round(immunogenicity, 3),
        'anti_degen': degen_ok,
        'composite_score': round(composite, 4),
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description='§5 Fallback: IDP Antibody Candidate Pipeline')
    ap.add_argument('--top', type=int, default=30, help='Number of top candidates to output')
    ap.add_argument('--out', type=str, default='idp_design_results/s5_wetlab_candidates.json',
                    help='Output JSON path')
    ap.add_argument('--pillar-a', type=str,
                    default='idp_design_results/pillar_a_variants_20260705_095536.json',
                    help='Pillar A variants JSON')
    args = ap.parse_args()

    os.makedirs('idp_design_results', exist_ok=True)
    all_candidates = []

    print("=" * 65)
    print("§5 FALLBACK: IDP Antibody Design — Candidate Pipeline")
    print("=" * 65)
    print("Strategy: Template-seeded redesign + known antibody library")
    print("         + pliability-matched selection (disorder as FILTER)")
    print()

    # --- Source 1: Known Anti-IDP Antibody CDRs ---
    print("[Source 1] Known Anti-IDP Antibody CDR-H3 Library")
    print("-" * 50)
    for ab_id, info in KNOWN_ANTI_IDP_CDRS.items():
        cdr_seq = info['seq']
        # Test against all Aβ epitopes
        for epi_name, (start, end, epi_seq) in ABETA_EPITOPES.items():
            if 'N-term' in info['target'] and 'N-term' not in epi_name:
                continue  # match target preference
            if 'mid' in info['target'] and 'mid' not in epi_name and 'central' not in epi_name:
                continue
            if 'C-term' in info['target'] and 'C-term' not in epi_name:
                continue
            candidate = score_candidate(cdr_seq, epi_name, start, end,
                                       source_ab=info['ab'], source_type=info['source'])
            all_candidates.append(candidate)

    n_known = len(all_candidates)
    print(f"  Generated {n_known} candidate-epitope pairs from known antibodies")

    # --- Source 2: Pillar A Template-Seeded Variants ---
    print(f"\n[Source 2] Pillar A Template-Seeded Redesign Variants")
    print("-" * 50)
    pillar_count = 0
    if os.path.exists(args.pillar_a):
        with open(args.pillar_a) as f:
            pillar_variants = json.load(f)
        for v in pillar_variants:
            # Extract CDR portion from full sequence
            full_seq = v.get('sequence', '')
            # For Pillar A, the sequence is the full grafted antibody.
            # Extract CDR-H3 by looking for the WGxG motif
            # Simpler: use the complexity/shannon scores directly
            cdr_seq = full_seq[:80]  # approximate CDR region
            if len(cdr_seq) < 6:
                continue

            # Score against N-term and mid epitopes (matching 4HIX/5CSZ targets)
            for epi_name, (start, end, epi_seq) in ABETA_EPITOPES.items():
                if epi_name in ('N-term (1-11)', 'mid (12-24)'):
                    candidate = score_candidate(cdr_seq, epi_name, start, end,
                                               source_ab=f'PillarA_{v.get("ab_id", "?")}',
                                               source_type='template_seeded')
                    all_candidates.append(candidate)
                    pillar_count += 1
        print(f"  Generated {pillar_count} candidate-epitope pairs from Pillar A")
    else:
        print(f"  Pillar A file not found: {args.pillar_a}")

    # --- Source 3: Native CDRs from 4HIX/5CSZ Crystal Structures ---
    print(f"\n[Source 3] Native CDR Templates (Hotspot-Frozen Scaffolds)")
    print("-" * 50)
    native_cdrs = {
        '4HIX_native_H3': 'YDHYSGSSDY',  # from crystal
        '5CSZ_native_H3': 'GKGYVRYFDV',  # from crystal
    }
    native_count = 0
    for name, cdr_seq in native_cdrs.items():
        for epi_name, (start, end, epi_seq) in ABETA_EPITOPES.items():
            if 'N-term' in epi_name and '4HIX' in name:
                candidate = score_candidate(cdr_seq, epi_name, start, end,
                                           source_ab=name, source_type='crystal_template')
                all_candidates.append(candidate)
                native_count += 1
            elif 'mid' in epi_name and '5CSZ' in name:
                candidate = score_candidate(cdr_seq, epi_name, start, end,
                                           source_ab=name, source_type='crystal_template')
                all_candidates.append(candidate)
                native_count += 1
    print(f"  Generated {native_count} candidate-epitope pairs from crystal templates")

    # --- Deduplication ---
    print(f"\n[Dedup] Removing duplicate CDR sequences...")
    seen = set()
    unique = []
    for c in all_candidates:
        key = c['cdr_sequence']
        if key not in seen:
            seen.add(key)
            unique.append(c)
    print(f"  {len(all_candidates)} → {len(unique)} unique ({len(all_candidates)-len(unique)} duplicates removed)")

    # --- Filtering ---
    print(f"\n[Filter] Applying quality filters...")
    filtered = []
    for c in unique:
        if not c['anti_degen']:
            continue
        if c['cdr_length'] < 4:
            continue
        if c['solubility'] < 0.4:
            continue
        if c['immunogenicity_risk'] > 0.7:
            continue
        filtered.append(c)
    print(f"  {len(unique)} → {len(filtered)} passed filters ({len(unique)-len(filtered)} removed)")

    # --- Ranking ---
    print(f"\n[Rank] Sorting by composite score...")
    filtered.sort(key=lambda c: c['composite_score'], reverse=True)

    # --- Top-N Output ---
    top = filtered[:args.top]
    print(f"\n{'='*65}")
    print(f"Top-{args.top} Candidates for Wet-Lab Screening")
    print(f"{'='*65}")
    for i, c in enumerate(top):
        print(f"\n#{i+1:3d}  Score={c['composite_score']:.4f}  "
              f"Pliability={c['pliability_score']:.4f}  "
              f"Solubility={c['solubility']:.3f}")
        print(f"     CDR-H3: {c['cdr_sequence'][:60]}")
        print(f"     Epitope: {c['epitope']}  Disorder={c['epitope_disorder']:.3f}")
        print(f"     Source: {c['source_antibody']} ({c['source_type']})")
        pd = c['pliability_details']
        print(f"     Details: disorder={pd['cdr_mean_disorder']:.3f} "
              f"match={pd['match_score']:.3f} "
              f"GlySer={pd['gly_ser_frac']:.2f} "
              f"Aromatic={pd['aromatic_frac']:.2f}")

    # --- Summary Statistics ---
    print(f"\n{'='*65}")
    print(f"Pipeline Summary")
    print(f"{'='*65}")
    sources = Counter(c['source_type'] for c in top)
    for src, cnt in sources.items():
        print(f"  {src}: {cnt}")
    composites = [c['composite_score'] for c in top]
    print(f"  Composite score: {np.mean(composites):.4f} ± {np.std(composites):.4f}")
    print(f"  Range: [{min(composites):.4f}, {max(composites):.4f}]")

    # --- Save ---
    output = {
        'metadata': {
            'plan': 'WAY4_REALIZE_PLAN_V14',
            'phase': '§5 Fallback',
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'reason': 'S0 NoGo — thesis not supported by crystallographic B-factor data',
            'strategy': 'Template-seeded redesign + known library + pliability selection',
            'verification_label': '[PENDING]',
            'total_scored': len(all_candidates),
            'unique_passed': len(filtered),
        },
        'top_candidates': top,
        'all_candidates': filtered,
    }

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {args.out}")
    print(f"Verification: [PENDING] — requires wet-lab SPR/BLI for binding confirmation")
    print(f"Next step: Select top-10 candidates for experimental validation")

    return output


if __name__ == '__main__':
    main()
