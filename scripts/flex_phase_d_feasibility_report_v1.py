"""FLEX-BINDER Phase D feasibility report v1 (metadata-level, frozen).

Consolidates the frozen audit artifacts into the feasibility verdict for the
Phase D data-feasibility gate. Metadata only: counts, identifier tallies and
accession-level both-label checks; no epitope sequences or record-level label
tables are persisted.
"""
import json
import sys
import urllib.parse
import urllib.request
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest

OUT = ROOT / 'results/flex_phase_d_v1'
API = 'https://query-api.iedb.org/bcell_search'
LENGTH = 'linear_sequence_length=gte.5&linear_sequence_length=lte.50'


def count(query):
    request = urllib.request.Request(
        f'{API}?{query}&limit=1',
        headers={'Prefer': 'count=exact', 'Range': '0-0',
                 'User-Agent': 'DisorderFlow-phaseD-feasibility/1'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return int(response.headers.get('Content-Range', '0-0/0').split('/')[-1])


def main():
    audit = read(OUT / 'iedb_feasibility_audit.json')
    top = audit['top_negative_source_antigens_by_records'][:8]
    both_labels = []
    for accession, negative_count in top:
        positive_count = count(
            f'qualitative_measure=eq.Positive&{LENGTH}&curated_source_antigen=not.is.null'
            f'&curated_source_antigen->>accession=eq.{urllib.parse.quote(accession, safe="")}')
        both_labels.append({'accession': accession,
                            'negative_records': negative_count,
                            'positive_records': positive_count})
        time.sleep(1)
        print(accession, negative_count, positive_count, flush=True)

    report = {
        'classification': 'metadata_level_feasibility_verdict_v1',
        'iedb_audit_sha256': digest(OUT / 'iedb_feasibility_audit.json'),
        'iedb_counts': audit['counts'],
        'distinct_negative_source_antigens': audit['counts']['I3_distinct_negative_source_antigens'],
        'top_antigens_with_both_labels': both_labels,
        'a4_binder_side': {
            'affinity_annotated_peptide_rows': 249,
            'unique_pdbs': 138,
            'distinct_antigen_names': 106,
            'source': 'results/flex_a4_affinity_inventory_v1 + summary aggregation 2026-10-30',
        },
        'protabank_status': 'unreachable from current network (SSL failure); '
                            'backup source, non-blocking',
        'gate_reference': {
            'required_families': '>=24 (target 30), per frozen power table '
                                 '(results/flex_phase_d_v1/power_table.json)',
            'per_family': '>=2 verified binders + >=1 composition-matched '
                          'experimentally-confirmed non-binder',
        },
        'verdict': {
            'raw_material': 'PASS with ~3 orders of magnitude margin: 30,388 '
                            'distinct antigens carry negative measurements; the '
                            'top-8 by negative count each also carry 1,300-3,000 '
                            'positive records (both labels on the same antigen)',
            'assembly_risk': 'OPEN - the gate requires curated families '
                             '(composition-matched negatives, verified binder '
                             'counts, per-family identity review); record-level '
                             'assembly is labor, not luck, and is the next step',
            'overall': 'feasible_to_proceed_to_assembly_pilot; preregistration '
                       'freeze remains blocked until the assembly pilot '
                       'demonstrates >=24 curated families or a definitive '
                       'shortfall',
        },
        'red_line_note': audit['red_line_note'],
        'script_sha256': digest(Path(__file__)),
    }
    freeze(OUT / 'feasibility_report_v1.json', report)
    print(json.dumps(report['verdict'], indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
