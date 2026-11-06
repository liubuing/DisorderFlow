"""FLEX-BINDER Phase A4: affinity-annotated peptide-antigen rows inventory.

Freezes the row-level list behind the scan's 249-affinity count and checks
overlap with the A1 qualified set. Read-only on the SAbDab summary snapshot.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest

REPO = ROOT
TSV = REPO / 'data/sabdab_summary_all.tsv'
SCAN = REPO / 'results/flex_campaign_v1_scaffold_scan.json'
A1 = REPO / 'results/flex_a1_scaffold_audit_v1/manifest.json'
OUT = REPO / 'results/flex_a4_affinity_inventory_v1'
PEPTIDE_TYPES = {'peptide', 'peptide-like'}


def split_field(value):
    return [p.strip() for p in (value or '').split('|') if p.strip() and p.strip() != 'NA']


def main():
    import csv
    with TSV.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    passing = set()
    if A1.exists():
        manifest = read(A1)
        passing = {r['pdb_id'] for r in manifest['records'] if r['passes']}

    inventory = []
    for row in rows:
        affinity = (row.get('affinity') or '').strip()
        if affinity in ('', 'None', 'NA'):
            continue
        ag_types = {t.lower() for t in split_field(row.get('antigen_type'))}
        if not (ag_types & PEPTIDE_TYPES):
            continue
        h = (row.get('Hchain') or '').strip()
        l = (row.get('Lchain') or '').strip()
        inventory.append({
            'pdb': row['pdb'],
            'has_light_annotation': bool(l) and l != 'NA',
            'affinity': affinity,
            'delta_g': (row.get('delta_g') or '').strip() or None,
            'affinity_method': (row.get('affinity_method') or '').strip() or None,
            'antigen_chain': row.get('antigen_chain'),
            'method': row.get('method'),
            'resolution': row.get('resolution'),
            'pmid': row.get('pmid'),
            'in_a1_qualified_set': row['pdb'].lower() in passing,
        })

    pdbs = sorted({r['pdb'] for r in inventory})
    summary = {
        'rows_with_affinity_and_peptide_antigen': len(inventory),
        'unique_pdbs': len(pdbs),
        'paired_HL_rows': sum(1 for r in inventory if r['has_light_annotation']),
        'no_light_rows': sum(1 for r in inventory if not r['has_light_annotation']),
        'rows_in_a1_qualified_set': sum(1 for r in inventory if r['in_a1_qualified_set']),
        'unique_pdbs_in_a1_qualified_set': len({r['pdb'] for r in inventory
                                                if r['in_a1_qualified_set']}),
        'affinity_methods': {},
    }
    for r in inventory:
        key = r['affinity_method'] or 'unspecified'
        summary['affinity_methods'][key] = summary['affinity_methods'].get(key, 0) + 1

    OUT.mkdir(parents=True, exist_ok=True)
    freeze(OUT / 'protocol.json', {
        'classification': 'affinity_metadata_inventory_not_binding_evidence',
        'tsv_sha256': digest(TSV),
        'boundary': 'SAbDab summary affinity annotations are unverified metadata; '
                    'no affinity claim may be derived without reading primary sources',
        'script_sha256': digest(Path(__file__))})
    freeze(OUT / 'summary.json', summary)
    freeze(OUT / 'inventory.json', {'rows': inventory})
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
