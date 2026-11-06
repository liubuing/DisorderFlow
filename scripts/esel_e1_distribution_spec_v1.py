"""ESEL Phase E1 development distribution spec (mechanical derivation).

Derives, from frozen artifacts only, the library-distribution tables that the
E1 preregistration will quote when it freezes.  This is DEVELOPMENT evidence:
it freezes nothing, orders nothing, and produces no candidate sequences.
Per docs/ESEL_E1_LIBRARY_PREREG_DRAFT_V1.md section 1, the E1 deliverable is a
distribution spec (CDR-H3 length / charge / hydrophobic / aromatic position
statistics, bucketed by epitope family), and per DECISION_20261101 rule 3 the
spec must not involve any self-developed affinity score -- all inputs below
are cohort statistics and metadata counts.

Inputs (frozen, read-only):
  results/flex_j0_v1/preregistration.json + cohort.json   (28 components)
  results/esel_e0_v1/epitope_presentation_matrix.json     (29 rows)
  results/esel_e0b_v1/iedb_counts.json                    (metadata counts only)
  results/design_v1_20261101/package.json                 (filter-rule precedent)

Output (write-once): results/esel_e1_v1/distribution_spec_development.json
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze  # noqa: E402

J0 = ROOT / 'results/flex_j0_v1'
E0 = ROOT / 'results/esel_e0_v1/epitope_presentation_matrix.json'
E0B = ROOT / 'results/esel_e0b_v1/iedb_counts.json'
DESIGN = ROOT / 'results/design_v1_20261101/package.json'
OUT = ROOT / 'results/esel_e1_v1'


def net_charge(seq):
    """Integer-ish charge: K/R +1, D/E -1, H +0.5 (partial at neutral pH)."""
    return (seq.count('K') + seq.count('R') + 0.5 * seq.count('H')
            - seq.count('D') - seq.count('E'))


def thirds_stats(seq):
    """Aromatic (FWY) and Pro placement across N-terminal/middle/C-terminal thirds."""
    groups = {'n_terminal': seq[:len(seq) // 3],
              'middle': seq[len(seq) // 3: 2 * len(seq) // 3] or seq[len(seq) // 3:],
              'c_terminal': seq[2 * len(seq) // 3:]}
    return {k: {'aromatic': sum(1 for a in s if a in 'FWY'),
                'proline': s.count('P'),
                'length': len(s)}
            for k, s in groups.items() if s}


def main():
    cohort = read(J0 / 'cohort.json')
    prereg = read(J0 / 'preregistration.json')
    matrix = read(E0)
    iedb = read(E0B)
    design = read(DESIGN)

    family_of = {}
    for family_key, members in prereg['cohort']['antigen_families'].items():
        for peptide in members:
            family_of[peptide] = family_key

    components = cohort['components']
    length_hist = dict(sorted(Counter(len(e['region_sequence'])
                                      for e in components).items()))

    rows_by_family = defaultdict(list)
    for row in matrix['rows']:
        rows_by_family[row['family']].append(row)

    families = {}
    for family_key in sorted(rows_by_family):
        fam_rows = rows_by_family[family_key]
        fam_components = [e for e in components
                          if family_of.get(e['peptide_sequence']) == family_key]
        has_components = bool(fam_components)
        h3_charges = [net_charge(e['region_sequence']) for e in fam_components]
        peptides = sorted({row['parent_peptide'] for row in fam_rows})
        fam = {
            'n_matrix_rows': len(fam_rows),
            'n_cohort_components': len(fam_components),
            'parent_peptides': peptides,
            'peptide_stats': [
                {'peptide': p,
                 'length': len(p),
                 'net_charge': net_charge(p),
                 'aromatic_fraction': round(sum(1 for a in p if a in 'FWY')
                                            / max(len(p), 1), 3)}
                for p in peptides],
            'native_h3_net_charge': [round(c, 1) for c in h3_charges],
            'h3_aromatic_fraction_mean': round(sum(
                sum(1 for a in e['region_sequence'] if a in 'FWY')
                / len(e['region_sequence']) for e in fam_components)
                / max(len(fam_components), 1), 3) if has_components else None,
            'h3_proline_count_range': [
                min(e['region_sequence'].count('P') for e in fam_components),
                max(e['region_sequence'].count('P') for e in fam_components)]
                if has_components else None,
            'aromatic_thirds_pooled': {},
            'iedb_precedent': None,
        }
        thirds = Counter()
        for e in fam_components:
            for bucket, stats in thirds_stats(e['region_sequence']).items():
                thirds[bucket] += stats['aromatic']
        fam['aromatic_thirds_pooled'] = dict(thirds)
        families[family_key] = fam

    # Join IEDB precedent counts (metadata only) via the E0b accession map.
    accession_map = read(ROOT / 'results/esel_e0b_v1/accession_map.json')
    am_rows = accession_map.get('rows') if isinstance(accession_map, dict) \
        and 'rows' in accession_map else accession_map
    if isinstance(am_rows, dict):
        am_rows = am_rows.get('rows', [])
    for family_key, fam in families.items():
        counts = []
        for row in rows_by_family[family_key]:
            match = [r for r in am_rows
                     if isinstance(r, dict) and r.get('instance') == row['instance']]
            for m in match:
                accession = m.get('accession') or m.get('uniprot_accession')
                if accession and accession in iedb['counts_by_accession']:
                    counts.append({'instance': row['instance'],
                                   'accession': accession,
                                   **iedb['counts_by_accession'][accession]})
        if counts:
            fam['iedb_precedent'] = counts

    summary = {
        'schema_version': 'esel_e1_distribution_spec_development_v1',
        'classification': 'development_spec_not_frozen_prereg',
        'basis': 'cohort statistics and metadata counts only; no affinity '
                 'score involved (DECISION_20261101 rule 3)',
        'n_components': len(components),
        'h3_length_histogram': length_hist,
        'filter_rule_precedent': design['filter_rules'],
        'n_families': len(families),
        'families': families,
        'usage_note': 'tables to be quoted verbatim by the E1 preregistration '
                      'at freeze time; freeze remains blocked on partner '
                      'confirmation and biosafety review '
                      '(docs/ESEL_E1_LIBRARY_PREREG_DRAFT_V1.md section 4)',
    }
    OUT.mkdir(parents=True, exist_ok=True)
    freeze(OUT / 'distribution_spec_development.json', summary)
    print(json.dumps({'n_families': len(families),
                      'h3_length_histogram': length_hist,
                      'out': str(OUT / 'distribution_spec_development.json')},
                     indent=1))


if __name__ == '__main__':
    main()
