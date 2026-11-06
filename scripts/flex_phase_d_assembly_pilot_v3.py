"""FLEX-BINDER Phase D assembly pilot v3: structural-bridge families.

Assembles preregistration-grade family packages on the v3 unit frozen in
feasibility_addendum_v3.json:
  family = (PDB-anchored binders of one antigen)
         x (composition-exact negatives, position-deconfounded)

Matching order per binder: (1) same-antigen negative, composition-exact, ALL
of its epitope regions >= 3 residues from every binder region of the family;
(2) otherwise a cross-antigen negative from the pooled measured-negative
universe (composition-exact; no positional relation exists by construction).
Both yields are recorded separately for the freeze decision.

Pass rule (declared before running): >= 2 distinct binders AND pooled yield
>= 0.5. Freeze-readiness signal: >= 24 passing families. Boundary: assembly
only; packages are inert until the Phase D preregistration is frozen.
"""
import json
import sys
import time
import urllib.parse
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest
from scripts.flex_phase_d_assembly_pilot_v1 import (
    AA, assay_tally, composition, fetch_all, standard_sequence)

OUT = ROOT / 'results/flex_phase_d_v1'
API = 'https://query-api.iedb.org/bcell_search'
LENGTH = 'linear_sequence_length=gte.5&linear_sequence_length=lte.50'
POSITION_GAP = 3
MAX_PERSISTED_PAIRS = 50
YIELD_GATE = 0.5
MIN_BINDERS = 2
FREEZE_READINESS_FAMILIES = 24


def regions_conflict(regions_a, regions_b, gap=POSITION_GAP):
    """True if any region pair is closer than `gap` residues."""
    for start_a, end_a in regions_a:
        for start_b, end_b in regions_b:
            if not (end_a + gap < start_b or end_b + gap < start_a):
                return True
    return False


def parse_regions(row):
    antigen = row.get('curated_source_antigen') or {}
    start, end = antigen.get('starting_position'), antigen.get('ending_position')
    if isinstance(start, int) and isinstance(end, int) and end >= start:
        return [(start, end)]
    return []


def collect():
    """Stream PDB-anchored positives and per-antigen negatives."""
    positives = fetch_all(
        f'qualitative_measure=eq.Positive&{LENGTH}&pdb_id=not.is.null'
        f'&linear_sequence=not.is.null&select=bcell_id,linear_sequence,pdb_id,'
        f'curated_source_antigen,assay_description',
        1000, 2000)
    accessions = sorted({(row.get('curated_source_antigen') or {}).get('accession')
                         for row in positives
                         if (row.get('curated_source_antigen') or {}).get('accession')})
    negatives_by_accession = {}
    negative_pool = {}
    for index, accession in enumerate(accessions):
        rows = fetch_all(
            f'qualitative_measure=eq.Negative&{LENGTH}&linear_sequence=not.is.null'
            f'&curated_source_antigen->>accession=eq.{accession}'
            f'&select=bcell_id,linear_sequence,curated_source_antigen,assay_description',
            10000, 30000)
        sequences = {}
        for row in rows:
            sequence = standard_sequence(row.get('linear_sequence'))
            if not sequence:
                continue
            entry = sequences.setdefault(sequence, {'regions': [], 'n_records': 0})
            entry['regions'].extend(parse_regions(row))
            entry['n_records'] += 1
        negatives_by_accession[accession] = sequences
        for sequence, entry in sequences.items():
            pool_entry = negative_pool.setdefault(sequence, {'regions': [], 'antigens': set()})
            pool_entry['regions'].extend(entry['regions'])
            pool_entry['antigens'].add(accession)
        if (index + 1) % 40 == 0:
            print(f'  negatives fetched for {index + 1}/{len(accessions)} antigens',
                  flush=True)
        time.sleep(0.5)
    return positives, accessions, negatives_by_accession, negative_pool


def assemble(positives, negatives_by_accession, negative_pool):
    by_accession = {}
    for row in positives:
        accession = (row.get('curated_source_antigen') or {}).get('accession')
        sequence = standard_sequence(row.get('linear_sequence'))
        if not accession or not sequence:
            continue
        entry = by_accession.setdefault(accession, {})
        binder = entry.setdefault(sequence, {'regions': [], 'pdbs': set(),
                                             'n_records': 0})
        binder['regions'].extend(parse_regions(row))
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
        pool_hits = 0
        for binder in sorted(binders):
            chosen = None
            source = None
            for sequence, info in sorted(own_negatives.items()):
                if sequence == binder or composition(sequence) != composition(binder):
                    continue
                if regions_conflict(info['regions'], binder_regions):
                    continue
                chosen, source = sequence, 'same_antigen_positional'
                break
            if chosen is None:
                for sequence, info in sorted(negative_pool.items()):
                    if sequence == binder or composition(sequence) != composition(binder):
                        continue
                    if sequence in own_negatives and regions_conflict(
                            own_negatives[sequence]['regions'], binder_regions):
                        continue
                    chosen, source = sequence, 'cross_antigen_pool'
                    break
            if chosen:
                matched.append({'binder': binder, 'negative': chosen, 'source': source})
                if source == 'same_antigen_positional':
                    same_antigen_hits += 1
                else:
                    pool_hits += 1
        yield_value = len(matched) / len(binders) if binders else 0.0
        families.append({
            'accession': accession,
            'identity': identity.get(accession),
            'n_distinct_binders': len(binders),
            'n_binder_pdbs': len({p for info in binders.values() for p in info['pdbs']}),
            'n_own_negative_sequences': len(own_negatives),
            'n_matched_binders': len(matched),
            'same_antigen_positional_matches': same_antigen_hits,
            'cross_antigen_pool_matches': pool_hits,
            'yield': round(yield_value, 4),
            'matched_pairs_preview': matched[:MAX_PERSISTED_PAIRS],
            'passes_pilot': len(binders) >= MIN_BINDERS and yield_value >= YIELD_GATE,
        })
    return families


def main():
    freeze(OUT / 'pilot_v3_protocol.json', {
        'classification': 'assembly_pilot_v3_structural_bridge_no_model_scoring',
        'unit': 'PDB-anchored binders x composition-exact negatives '
                '(same-antigen positional first, cross-antigen pool fallback)',
        'position_rule': 'same-antigen negative admissible only if ALL its epitope '
                         'regions are >= 3 residues from every binder region',
        'pass_rule': {'min_distinct_binders': MIN_BINDERS,
                      'pooled_yield': YIELD_GATE},
        'freeze_readiness': f'>= {FREEZE_READINESS_FAMILIES} passing families',
        'negative_source_disclosure': 'antigen-level non-reactivity: measured '
                                      'non-reactive epitopes of the same antigen '
                                      '(any tested antibody/sera), not '
                                      'per-antibody negatives',
        'boundary': 'assembly only; packages inert until preregistration freeze',
        'script_sha256': digest(Path(__file__))})

    positives, accessions, negatives_by_accession, negative_pool = collect()
    print(f'positives {len(positives)}, accessions {len(accessions)}, '
          f'own-negative sets {len(negatives_by_accession)}, '
          f'pool distinct sequences {len(negative_pool)}', flush=True)

    families = assemble(positives, negatives_by_accession, negative_pool)
    for family in families:
        freeze(OUT / 'pilot_families_v3' / f"{family['accession']}.json", family)
        print(family['accession'], 'binders', family['n_distinct_binders'],
              'yield', family['yield'],
              'PASS' if family['passes_pilot'] else 'FAIL', flush=True)

    passing = [f for f in families if f['passes_pilot']]
    summary = {
        'families_attempted': len(families),
        'families_passed': len(passing),
        'freeze_readiness': len(passing) >= FREEZE_READINESS_FAMILIES,
        'mean_yield': round(sum(f['yield'] for f in families) / len(families), 4)
                      if families else None,
        'same_antigen_positional_total': sum(f['same_antigen_positional_matches']
                                             for f in families),
        'cross_antigen_pool_total': sum(f['cross_antigen_pool_matches']
                                        for f in families),
        'verdict': ('freeze_ready_proceed_to_phase_d_preregistration'
                    if len(passing) >= FREEZE_READINESS_FAMILIES
                    else 'shortfall_documented'),
        'classification': 'pilot_v3_summary_no_gate_claim',
    }
    freeze(OUT / 'pilot_v3_summary.json', summary)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
