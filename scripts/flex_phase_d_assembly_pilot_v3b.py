"""FLEX-BINDER Phase D pilot v3b: v3 matching against the FULL negative pool.

Identical frozen matching rule as v3 (composition-exact; same-antigen
position-deconfounded first, cross-antigen pool fallback), with the pool
scaled from the 200 bridge antigens (~15k sequences) to the full IEDB
negative universe (709,890 distinct sequences, frozen in
negative_pool_checkpoint.json). This is the A' fork test: rule unchanged,
pool scaled.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest
from scripts.flex_phase_d_assembly_pilot_v3 import (
    FREEZE_READINESS_FAMILIES, MIN_BINDERS, YIELD_GATE, collect)
from scripts.flex_phase_d_assembly_pilot_v1 import composition

OUT = ROOT / 'results/flex_phase_d_v1'


def main():
    checkpoint = read(OUT / 'negative_pool_checkpoint.json')
    pool_counts = checkpoint['pool']
    negative_pool = {sequence: {'regions': [], 'antigens': set()}
                     for sequence in pool_counts}
    # Precompute the composition index once: 700k sequences x per-binder
    # Counter recomputation would be billions of operations.
    composition_index = {}
    for sequence in negative_pool:
        composition_index.setdefault(composition(sequence), []).append(sequence)
    print('pool sequences:', len(negative_pool),
          'compositions:', len(composition_index), flush=True)

    positives, accessions, negatives_by_accession, _own = collect()
    print(f'positives {len(positives)}, accessions {len(accessions)}', flush=True)

    import scripts.flex_phase_d_assembly_pilot_v3 as v3

    by_accession = {}
    for row in positives:
        accession = (row.get('curated_source_antigen') or {}).get('accession')
        sequence = v3.standard_sequence(row.get('linear_sequence'))
        if not accession or not sequence:
            continue
        entry = by_accession.setdefault(accession, {})
        binder = entry.setdefault(sequence, {'regions': [], 'pdbs': set(),
                                             'n_records': 0})
        binder['regions'].extend(v3.parse_regions(row))
        if row.get('pdb_id'):
            binder['pdbs'].add(row['pdb_id'])
        binder['n_records'] += 1

    identity = {}
    for row in positives:
        accession = (row.get('curated_source_antigen') or {}).get('accession')
        if accession not in identity:
            antigen = row.get('curated_source_antigen') or {}
            identity[accession] = {'accession': accession,
                                   'name': antigen.get('name'),
                                   'organism': antigen.get('source_organism_name')}

    families = []
    for accession in sorted(by_accession):
        binders = by_accession[accession]
        binder_regions = [r for info in binders.values() for r in info['regions']]
        own_negatives = negatives_by_accession.get(accession, {})
        matched = []
        same_antigen_hits = 0
        for binder in sorted(binders):
            chosen = None
            source = None
            for sequence, info in sorted(own_negatives.items()):
                if sequence == binder or v3.composition(sequence) != v3.composition(binder):
                    continue
                if v3.regions_conflict(info['regions'], binder_regions):
                    continue
                chosen, source = sequence, 'same_antigen_positional'
                break
            if chosen is None:
                bucket = composition_index.get(v3.composition(binder), [])
                for sequence in bucket:
                    if sequence == binder:
                        continue
                    if sequence in own_negatives and v3.regions_conflict(
                            own_negatives[sequence]['regions'], binder_regions):
                        continue
                    chosen, source = sequence, 'cross_antigen_pool'
                    break
            if chosen:
                matched.append({'binder': binder, 'negative': chosen, 'source': source})
                if source == 'same_antigen_positional':
                    same_antigen_hits += 1
        yield_value = len(matched) / len(binders) if binders else 0.0
        families.append({
            'accession': accession,
            'identity': identity.get(accession),
            'n_distinct_binders': len(binders),
            'n_binder_pdbs': len({p for info in binders.values() for p in info['pdbs']}),
            'n_own_negative_sequences': len(own_negatives),
            'n_matched_binders': len(matched),
            'same_antigen_positional_matches': same_antigen_hits,
            'cross_antigen_pool_matches': len(matched) - same_antigen_hits,
            'yield': round(yield_value, 4),
            'matched_pairs_preview': matched[:50],
            'passes_pilot': len(binders) >= MIN_BINDERS and yield_value >= YIELD_GATE,
        })
    for family in families:
        freeze(OUT / 'pilot_families_v3b' / f"{family['accession']}.json", family)
    passing = [f for f in families if f['passes_pilot']]

    summary = {
        'pool_distinct_sequences': len(negative_pool),
        'families_attempted': len(families),
        'families_passed': len(passing),
        'freeze_readiness': len(passing) >= FREEZE_READINESS_FAMILIES,
        'mean_yield': round(sum(f['yield'] for f in families) / len(families), 4)
                      if families else None,
        'binders_matched_total': sum(f['n_matched_binders'] for f in families),
        'binders_total': sum(f['n_distinct_binders'] for f in families),
        'same_antigen_positional_total': sum(f['same_antigen_positional_matches']
                                             for f in families),
        'verdict': ('freeze_ready_proceed_to_phase_d_preregistration'
                    if len(passing) >= FREEZE_READINESS_FAMILIES
                    else 'exact_composition_infeasible_at_full_pool_scale'),
        'classification': 'pilot_v3b_summary_no_gate_claim',
    }
    freeze(OUT / 'pilot_v3b_summary.json', summary)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
