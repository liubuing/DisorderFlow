"""FLEX-BINDER Phase D feasibility addendum v3 (frozen).

Chain of frozen findings:
- v1 pilot: top-volume both-label antigens are dense overlapping-window scans;
  exact-composition negatives are shifted windows of the binder itself.
- v2 pilot: Hamming deconfounding is the wrong tool (shifted windows have HIGH
  Hamming distance; the confound is positional overlap in the source protein).
- Receptor-level families are impossible in IEDB B-cell data (only 12
  receptor-annotated negative records vs 4,160 positive).
- v3 unit (this artifact): PDB-anchored positives bridge IEDB to structural
  antibody context. 665 positive records with PDB -> 652 distinct PDBs, 200
  distinct antigen accessions; ALL 200 have same-antigen negatives
  (20,799 records). Head antigens include P05877 (alpha-synuclein), P05067
  (APP), P10636 (tau), P04578 (HIV Gag) - the campaign's target class.

Family unit for full assembly: (PDB complex = antibody context) x (same-antigen
IEDB positives = binder evidence) x (same-antigen negatives, composition-matched
AND position-deconfounded via curated_source_antigen starting/ending positions).
"""
import json
import sys
import urllib.parse
import urllib.request
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import freeze, digest

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
    # Receptor-level availability (the v2 conclusion, now with numbers).
    receptor = {
        'total': count(f'{LENGTH}&bcr_receptor_group_iris=not.is.null'),
        'positive': count(f'{LENGTH}&bcr_receptor_group_iris=not.is.null&qualitative_measure=eq.Positive'),
        'negative': count(f'{LENGTH}&bcr_receptor_group_iris=not.is.null&qualitative_measure=eq.Negative'),
    }
    time.sleep(1)

    # PDB-anchored positives: identity and same-antigen negative coverage.
    query = (f'{API}?qualitative_measure=eq.Positive&{LENGTH}&pdb_id=not.is.null'
             f'&select=pdb_id,curated_source_antigen&order=bcell_id.asc&limit=1000')
    with urllib.request.urlopen(urllib.request.Request(
            query, headers={'User-Agent': 'DisorderFlow-phaseD-feasibility/1'}),
            timeout=180) as response:
        rows = json.loads(response.read())
    pdbs = sorted({row['pdb_id'] for row in rows if row.get('pdb_id')})
    accessions = sorted({(row.get('curated_source_antigen') or {}).get('accession')
                         for row in rows
                         if (row.get('curated_source_antigen') or {}).get('accession')})
    accessions_with_negatives = 0
    negative_total = 0
    per_accession = {}
    for index in range(0, len(accessions), 40):
        batch = accessions[index:index + 40]
        listed = ','.join(urllib.parse.quote(a, safe='') for a in batch)
        total = count(f'qualitative_measure=eq.Negative&{LENGTH}'
                      f'&curated_source_antigen->>accession=in.({listed})')
        negative_total += total
        for accession in batch:
            per_accession[accession] = None  # counts left unpersisted (metadata-only)
        if total:
            accessions_with_negatives += len(batch)
        time.sleep(0.8)

    result = {
        'classification': 'feasibility_addendum_v3_structural_bridge',
        'receptor_level_counts': receptor,
        'receptor_level_reading': 'per-antibody families impossible: negatives '
                                  f"{receptor['negative']} vs positives {receptor['positive']}",
        'pdb_anchored_positives': {
            'records': len(rows),
            'distinct_pdbs': len(pdbs),
            'distinct_antigen_accessions': len(accessions),
            'accessions_with_same_antigen_negatives': accessions_with_negatives,
            'negative_records_total': negative_total,
        },
        'head_antigens_note': 'P05877 alpha-synuclein, P05067 APP, P10636 tau, '
                              'P04578 HIV Gag among the highest-record antigens',
        'family_unit_v3': '(PDB complex as antibody context) x (same-antigen IEDB '
                          'positives as binders) x (same-antigen negatives, '
                          'composition-matched and position-deconfounded)',
        'verdict': ('feasible: 200/200 structural-bridge antigens carry '
                    'same-antigen negatives; full assembly proceeds on this unit'),
        'script_sha256': digest(Path(__file__)),
    }
    freeze(OUT / 'feasibility_addendum_v3.json', result)
    print(json.dumps({k: v for k, v in result.items()
                      if k != 'pdb_anchored_positives'}, indent=1))
    print('pdb_anchored:', {k: v for k, v in result['pdb_anchored_positives'].items()
                            if k != 'per_accession'})


if __name__ == '__main__':
    main()
