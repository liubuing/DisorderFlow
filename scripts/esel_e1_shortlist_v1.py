"""ESEL Phase E1 epitope shortlist (development evidence, mechanical join).

Joins the frozen E0 presentation matrix (29 rows x 4 presentation variants)
with the E0b accession anchors and IEDB metadata counts, adds biosafety
flags, and proposes a 2-4 epitope RFQ shortlist plus numeric E1 thresholds
for the freeze-time decision.  Selection is metadata-driven only (anchor
status, exposure cleanliness, IEDB precedent counts, biosafety flags) -- no
affinity score and no model output participates (DECISION_20261101 rule 3).

Output (write-once): results/esel_e1_v1/epitope_shortlist_development.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze  # noqa: E402

E0 = ROOT / 'results/esel_e0_v1/epitope_presentation_matrix.json'
E0B = ROOT / 'results/esel_e0b_v1'
OUT = ROOT / 'results/esel_e1_v1'

# Accessions requiring the pending biosafety review before any order.
BIOSAFETY_REVIEW = {
    'P0DTC2': 'SARS-CoV-2 spike-derived epitope',
    'P04578': 'HIV gp160/gp41-derived epitope',
}


def main():
    matrix = read(E0)
    amap = read(E0B / 'accession_map.json')['mapping']
    counts = read(E0B / 'iedb_counts.json')['counts_by_accession']

    rows = []
    for row in matrix['rows']:
        inst = row['instance']
        anchor = amap.get(inst, {})
        accession = anchor.get('anchor_accession')
        iedb = counts.get(accession) if accession else None
        biosafety = BIOSAFETY_REVIEW.get(accession)
        rows.append({
            'row_id': row['row_id'],
            'instance': inst,
            'source_pdb': row['source_pdb'],
            'entry_title': anchor.get('entry_title'),
            'family': row['family'],
            'parent_peptide': row['parent_peptide'],
            'parent_length': row['parent_length'],
            'exposure_clean': row['exposure_clean'],
            'anchor_status': anchor.get('status', 'unresolved'),
            'anchor_accession': accession,
            'iedb_total_5_50': iedb['total_5_50'] if iedb else None,
            'iedb_positive_5_50': iedb['positive_5_50'] if iedb else None,
            'biosafety_flag': biosafety,
            'n_presentation_variants': len(row['variants']),
            'variants': row['variants'],
        })

    eligible = [r for r in rows
                if r['anchor_status'] == 'anchored' and r['exposure_clean']]
    # RFQ shortlist: precedent-rich first (positive IEDB records as an
    # immunological-prior proxy, not a prediction), capped at 4; biosafety-
    # flagged rows are included but marked pending review, never silently
    # dropped and never silently ordered.
    eligible_ranked = sorted(
        eligible,
        key=lambda r: (-(r['iedb_positive_5_50'] or 0), r['row_id']))
    shortlist = [dict(r, shortlist_rank=i + 1)
                 for i, r in enumerate(eligible_ranked[:4])]
    for r in shortlist:
        r['order_block'] = ('pending biosafety review' if r['biosafety_flag']
                            else 'pending E1 freeze')

    thresholds = {
        'capability_endpoint_T_star_nM': {
            'proposal': 100,
            'basis': 'lower edge of the published 3-100 nM band for de novo '
                     'disordered-target binders (Liu 2025, Nature '
                     '10.1038/s41586-025-09248-9); display-selected clones '
                     'routinely reach this band; 10 nM is the stretch goal, '
                     'not the gate',
            'status': 'proposal, written into the preregistration at E1 freeze',
        },
        'e2_enrichment_gate_fold': {
            'proposal': 10,
            'basis': 'round-over-round specific enrichment (R_n/R_1 on '
                     'target minus parallel negative-control matrix) at or '
                     'above 10x is the conventional display-panning '
                     'continuation signal; exact value frozen at E1',
            'status': 'proposal, written into the preregistration at E1 freeze',
        },
        'e2_control_matrix': {
            'proposal': ['irrelevant scFv negative', 'native positive control',
                         'composition-perturbed peptide per epitope'],
            'status': 'frozen in the E1 draft section 3; listed here for '
                      'the RFQ so the vendor quotes the parallel screens',
        },
    }

    summary = {
        'schema_version': 'esel_e1_epitope_shortlist_development_v1',
        'classification': 'development_evidence_not_frozen_prereg',
        'n_matrix_rows': len(rows),
        'n_anchored_exposure_clean': len(eligible),
        'n_shortlist': len(shortlist),
        'rows': rows,
        'shortlist': shortlist,
        'threshold_proposals': thresholds,
        'order_path_note': 'design_v1_20261101 package (4HIX Abeta-fab, 52 '
                           'constructs) is separately synthesis-ready via '
                           'order_list_v1.csv / synthesis_aa.fasta',
        'freeze_blockers': ['partner confirmation (three items, '
                            'docs/ESEL_PARTNER_BRIEF_V1.md)',
                            'biosafety review for flagged accessions'],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    freeze(OUT / 'epitope_shortlist_development.json', summary)
    print(json.dumps({k: summary[k] for k in
                      ['n_matrix_rows', 'n_anchored_exposure_clean',
                       'n_shortlist']}, indent=1))
    for r in shortlist:
        print(f"  #{r['shortlist_rank']} {r['instance']} "
              f"{r['parent_peptide'][:24]} len={r['parent_length']} "
              f"IEDB+={r['iedb_positive_5_50']} "
              f"bio={'REVIEW' if r['biosafety_flag'] else '-'} "
              f"{r['entry_title'][:50] if r['entry_title'] else ''}")


if __name__ == '__main__':
    main()
