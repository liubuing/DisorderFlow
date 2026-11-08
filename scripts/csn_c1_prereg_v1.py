"""CSN Phase C1 (v1) preregistration freeze.

Freezes results/csn_c1_v1/preregistration.json AFTER the data-acquisition
artifacts exist and BEFORE any case/control window reads a flexibility label
(M4 discipline). The prereg records the pre-data design decisions: case/control
definitions, label ladder, endpoint, closed interpretation rules, stop rules,
red lines, and sha256 provenance of every input it depends on.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_aayl_original_endpoint_cohort import freeze  # noqa: E402
from scripts.benchmark_aayl_learnability import digest  # noqa: E402

OUT = ROOT / 'results/csn_c1_v1'
QUEUE = ROOT / 'results/flex_j0_v1/preregistration.json'
C0_SUMMARY = ROOT / 'results/csn_c0_v2/summary.json'
DATE = '2026-11-08'
CLASSIFICATION = 'development_descriptive_not_confirmatory_claim'

REQUIRED_INPUTS = ['accession_map_csn.json', 'iedb_counts_csn.json', 'parents.json']


def main():
    for name in REQUIRED_INPUTS:
        if not (OUT / name).exists():
            raise SystemExit(f'prerequisite missing: {name}; run csn_c1_data_v1 first')
    prereg = {
        'classification': CLASSIFICATION,
        'campaign': 'CSN v1',
        'phase': 'C1',
        'date': DATE,
        'question': ('Are antibody-epitope windows on neuronal/synaptic IDP parents '
                     'enriched for free-state transient structure (lower flexibility) '
                     'relative to exposure-matched, structurally-unprecedented windows '
                     'of the same parents?'),
        'case_windows': {
            'definition': ('epitope windows of C0 valid families mapped onto parent '
                           'sequences, plus ESEL E0b anchored rows'),
            'parent_anchoring_rules': {
                'anchored': 'parent peptide is a contiguous substring of the UniProt sequence',
                'anchored_with_mismatch': ('RCSB entity names the accession AND the peptide '
                                           'differs from the UniProt sequence by <=1 '
                                           'substitution in a <=20-residue peptide AND the '
                                           'entry title names the parent protein; every '
                                           'mismatch is recorded per window and the family '
                                           'is flagged'),
            },
            'pre_data_note': ('tau GSVQIVYKPK is expected to require anchored_with_mismatch '
                              '(canonical tau carries VQIVYKPVL at the homologous segment; '
                              'the deposited peptide reads GSVQIVYKPK per the VHH Z70 '
                              'entry) - this rule is frozen before any label read'),
        },
        'control_windows': {
            'definition': ('same parent, same window length, exposure-matched, and without '
                           'structural precedent within the local evidence universe'),
            'exposure_metric': ('mean per-residue CA SASA of the window computed on the '
                                'bound antigen chain of the deposited complex '
                                '(Bio.PDB.SASA ShrakeRupley); bound-state exposure is a '
                                'proxy - the free-state limitation is disclosed'),
            'precedent_universe': ('the local sabdab2 projected pool (1,146 CIFs) sweep '
                                   'plus the 28-component queue; NOT an exhaustive SAbDab '
                                   'search - disclosed limitation'),
            'iedb_role': ('protein-level IEDB metadata counts are recorded as covariates '
                          'only; window-level IEDB precedent is NOT obtainable at '
                          'metadata level and is never claimed'),
        },
        'label_ladder': {
            'primary': 'ATLAS MD RMSF (per-residue mean over replicates; cache-first)',
            'secondary': 'NMR ensemble RMSF v2 (results/flex_a2_nmr_rmsf_v2, corrected Kabsch)',
            'downgraded_fallback': ('AFDB pLDDT tracks (results/flex_a2_atlas_plddt_audit_v1) '
                                    '- disorder proxy, not flexibility; recorded as '
                                    'downgraded and never pooled with RMSF labels'),
        },
        'label_strata': {
            'primary_stratum': ('parents whose case window carries an ATLAS or NMR v2 '
                                'RMSF label (pre-data count: 3 - alpha-syn P37840 via '
                                '2kkw NMR v2; gp160 P04578 and spike P0DTC2 via ATLAS); '
                                'the primary endpoint is computed ON THIS STRATUM ONLY'),
            'downgraded_stratum': ('parents with AFDB pLDDT only (expected 6): '
                                   'descriptive-only, reported separately, never pooled '
                                   'with the primary stratum'),
            'stratum_note': ('different label types are different quantities; pooling '
                             'would fabricate coverage, so the strata are frozen here '
                             'before any label read'),
        },
        'environmental_notes': {
            'peptide_search_unavailable': ('UniProt peptide-search asyncrest submits '
                                           '(202 + Location) but the poll returned empty '
                                           'for >=90 s on 2026-11-08 for a known-positive '
                                           'control peptide (KEDAFSKLKEKFMNEL, contained '
                                           'in P21579); the fallback path is recorded '
                                           'unavailable, not zero-evidence'),
            'unresolved_rows': ('6i2g, 8skj, 6zwk remain unresolved and are excluded '
                                'from the C1 universe; tau 8op0/8opi/8pii anchor via '
                                'anchored_with_mismatch to P10636'),
        },
        'endpoint': {
            'metric': 'parent-level paired difference: mean window RMSF(case) - mean window RMSF(control)',
            'uncertainty': ('parent-level cluster bootstrap, 10,000 draws, seed 20261108'),
            'interpretation_closed_rules': {
                'CI excludes zero and negative': 'transient_structure_enrichment',
                'CI excludes zero and positive': 'inverse_enrichment_exposed_flexible',
                'CI crosses zero': 'no_evidence',
            },
        },
        'stop_rules': {
            'label_starvation': ('parents with case + control + label coverage < 3 -> '
                                 'freeze verdict label_starvation; downgrade to AFDB '
                                 'pLDDT or report and stop (E0 precedent)'),
            'iedb_unavailable': ('structural-only case definition with a recorded '
                                 'downgrade note'),
        },
        'red_lines': [
            'IEDB access stays metadata-level: counts and query strings only',
            'no model scores anywhere in C1 selection or scoring',
            'no cross-citation with frozen ECLS/PAE/J0 results',
            'enrichment describes window properties; no disease-mechanism or design claim',
        ],
        'claims_boundary': ('C1 tests a population-level association between antibody '
                            'precedent and free-state transient structure on neuronal IDP '
                            'parents. It does not establish why antibodies bind where they '
                            'bind, does not predict immunogenicity, and does not support '
                            'any design or clinical claim.'),
        'provenance': {
            'script_sha256': digest(Path(__file__)),
            'queue_sha256': digest(QUEUE),
            'c0_summary_sha256': digest(C0_SUMMARY),
            'data_artifacts_sha256': {name: digest(OUT / name)
                                      for name in REQUIRED_INPUTS},
        },
    }
    freeze(OUT / 'preregistration.json', prereg)
    print('C1 preregistration frozen:', digest(OUT / 'preregistration.json'))


if __name__ == '__main__':
    main()
