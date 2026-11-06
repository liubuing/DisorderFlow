"""FLEX-BINDER Phase B, Gate J0 preregistration.

Freezes the judge-validation protocol BEFORE any scoring exists: cohort,
judges, shuffle protocol, endpoints, gates, seeds, and the honest power
statement from the real A1b cohort. This script performs no model inference,
no structure parsing beyond reading frozen artifacts, and no network access.

Gate J0 question: does the ensemble/pose-conditioned likelihood-shift judge
separate native CDR-H3 sequences from composition-matched shuffles on the
14-PDB development cohort? (M2 discipline: the judge is validated before it
is trusted with anything.)
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest

OUT = ROOT / 'results/flex_j0_v1'
COHORT_PDBS = ['2x6m', '5hm1', '5ivn', '6i2g', '6j7w', '6s0y', '6zwk',
               '8b8i', '8op0', '8opi', '8pii', '8skj', '9eiu', '9lds']

ALPHA_ONE_SIDED = 0.05
POWER_TARGET = 0.80


def binomial_sign_test(n_families, p_alt, alpha=ALPHA_ONE_SIDED):
    """Smallest critical count c with P(X >= c | p=0.5) <= alpha, then power."""
    def tail(count, p):
        return sum(math.comb(n_families, k) * p ** k * (1 - p) ** (n_families - k)
                   for k in range(count, n_families + 1))
    critical = None
    for c in range(n_families + 1):
        if tail(c, 0.5) <= alpha:
            critical = c
            break
    if critical is None:
        return None, None, None
    return critical, tail(critical, 0.5), tail(critical, p_alt)


def power_curve(n_families):
    curve = {}
    for p_alt in (0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        critical, alpha_real, power = binomial_sign_test(n_families, p_alt)
        curve[f'p_alt_{p_alt}'] = {'critical_count': critical,
                                   'realized_alpha': alpha_real,
                                   'power': power}
    # Minimum p_alt reaching the power target at the realized critical count.
    critical, alpha_real, _ = binomial_sign_test(n_families, 0.5)
    lo, hi = 0.5, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if tail_power(n_families, critical, mid) >= POWER_TARGET:
            hi = mid
        else:
            lo = mid
    return curve, critical, alpha_real, hi


def tail_power(n_families, critical, p_alt):
    return sum(math.comb(n_families, k) * p_alt ** k * (1 - p_alt) ** (n_families - k)
               for k in range(critical, n_families + 1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a1b = read(ROOT / 'results/flex_a1b_isolation_families_v2/summary.json')
    a1 = read(ROOT / 'results/flex_a1_scaffold_audit_v1/manifest.json')
    exposure = read(ROOT / 'results/flex_a1b_isolation_families_v1/exposure.json')

    assert sorted(a1b['same_pdb_list']) == sorted(set(a1b['same_pdb_list']))
    cohort_components = []
    for record in a1['records']:
        if not record['passes'] or record['pdb_id'] not in COHORT_PDBS:
            continue
        if not exposure.get(record['instance'], {}).get('exposure_clean'):
            continue
        cohort_components.append({
            'instance': record['instance'],
            'pdb_id': record['pdb_id'],
            'heavy_chain': record['chains']['heavy'],
            'peptide_chain': record['chains']['peptide'],
            'heavy_sequence': record['sequences']['heavy'],
            'peptide_sequence': record['sequences']['peptide'],
            'region_sequence': record['sequences']['region'],
        })
    cohort_components.sort(key=lambda c: c['instance'])
    cohort_pdbs_present = sorted({c['pdb_id'] for c in cohort_components})

    # Cohort antigen families under the declared A1b-v2 rule (0.50/0.80 merge).
    from scripts.flex_a1b_isolation_families_v1 import alignment_identity
    peptides = sorted({c['peptide_sequence'] for c in cohort_components})
    parent = {p: p for p in peptides}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(peptides):
        for b in peptides[i + 1:]:
            identity, cov_a, cov_b = alignment_identity(a, b)
            if (identity >= 0.50 and (cov_a >= 0.80 or cov_b >= 0.80)):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
    families = {}
    for p in peptides:
        families.setdefault(find(p), []).append(p)
    family_of = {p: find(p) for p in peptides}
    n_families = len(families)

    curve, critical, alpha_real, mde = power_curve(n_families)

    protocol = {
        'classification': 'judge_validation_gate_not_confirmatory_claim',
        'cohort': {
            'definition': 'A1 passing x A1b same-PDB-clean x >=1 exposure-clean component',
            'pdbs': cohort_pdbs_present,
            'n_pdbs': len(cohort_pdbs_present),
            'components': cohort_components,
            'n_components': len(cohort_components),
            'n_antigen_families': n_families,
            'antigen_families': {f'seq_{k}': v for k, v in enumerate(families.values())},
        },
        'judge_primary': {
            'name': 'bfn_ecls',
            'definition': 'ECLS = factual H3 NLL(complex context) - factual H3 NLL(stripped context); '
                          'lower ECLS = peptide coordinates help the sequence',
            'model': 'production checkpoint per app_config.yaml (CHECKPOINT_TRUTH s2032); '
                     'path and sha256 recorded at run time',
            'region': 'CDR-H3 Chothia 93-102 indices from the frozen A1 ANARCII numbering',
            'contexts': {'complex': "context_chains=['P']",
                         'stripped': "context_chains=[] (heavy chain only)"},
            'fixed_t': 0.5,
            'scoring_seeds': [20261029, 20261030, 20261031],
            'seed_note': 'seed-repeats averaged per context; if a determinism check on '
                         'the first component shows bit-identical repeats, collapse to '
                         'one pass and record that fact in the run output',
        },
        'judge_secondary_optional': {
            'name': 'proteinmpnn_nll',
            'definition': 'same two contexts, ProteinMPNN sequence NLL; comparison only',
            'status': 'run if environment permits; absence is disclosed, not hidden',
        },
        'shuffle_protocol': {
            'k_shuffles_per_component': 50,
            'method': 'composition-matched multiset permutation of the native region',
            'rng_seed': 20261029,
        },
        'endpoint': {
            'component_level': 'advantage = mean(ECLS over shuffles) - ECLS(native); '
                               'positive means native contrast more favorable',
            'family_level': 'mean advantage over components of each antigen family',
            'uncertainty': 'family-level cluster bootstrap, 10,000 draws, seed 20261029',
        },
        'gates': {
            'j0a_single_pose': {
                'pass': '>= 70% of antigen families positive AND family-level bootstrap '
                        '95% CI of mean advantage excludes zero',
                'meaning_on_pass': 'single-pose judge admissible for development '
                                   'calibration only; T1 already showed single-pose '
                                   'overestimates, so this alone cannot unlock Phase C',
            },
            'j0b_ensemble': {
                'status': 'defined now, executed after j0a',
                'protocol': 'T1-style restrained OpenMM ensemble per component '
                            '(4 seeds, peptide backbone 25 kJ/mol/nm^2), ECLS averaged '
                            'over conformers (ensemble-marginalized)',
                'pass': 'same thresholds as j0a AND ensemble mean advantage positive '
                        'with CI excluding zero',
                'meaning_on_pass': 'judge admissible for Phase C development work',
            },
            'on_fail': 'either gate failing returns to judge design; Phase C is forbidden',
        },
        'power_statement': {
            'n_families': n_families,
            'test': 'exact binomial sign test over antigen families, one-sided',
            'alpha_nominal': ALPHA_ONE_SIDED,
            'critical_count': critical,
            'realized_alpha': alpha_real,
            'power_curve': curve,
            'min_p_alt_for_80pct_power': round(mde, 4),
            'reading': f'with {n_families} families the gate can only detect a judge '
                       f'that is positive on >= {math.ceil(mde * n_families)} of '
                       f'{n_families} families; weaker judges are invisible here',
        },
        'status_declaration': 'Gate J0 is an internal go/no-go on the judge, run on a '
                              'development cohort. It is NOT a confirmatory endpoint '
                              '(M7): any confirmatory claim still requires universe '
                              'expansion per the campaign document section 6.',
        'provenance': {
            'a1_manifest_sha256': digest(ROOT / 'results/flex_a1_scaffold_audit_v1/manifest.json'),
            'a1b_summary_v2_sha256': digest(ROOT / 'results/flex_a1b_isolation_families_v2/summary.json'),
            'a1b_exposure_sha256': digest(ROOT / 'results/flex_a1b_isolation_families_v1/exposure.json'),
            'script_sha256': digest(Path(__file__)),
        },
    }
    freeze(OUT / 'preregistration.json', protocol)
    compact = {
        'n_pdbs': len(cohort_pdbs_present),
        'n_components': len(cohort_components),
        'n_antigen_families': n_families,
        'critical_count': critical,
        'realized_alpha': alpha_real,
        'min_p_alt_for_80pct_power': round(mde, 4),
    }
    print(json.dumps(compact, indent=1))


if __name__ == '__main__':
    main()
