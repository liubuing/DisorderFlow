"""Family clustering and family-disjoint train/eval split for the analog cohorts.

Population: components that passed every v26 coordinate gate, restricted to
entries with no v25 exposure flags (confirmatory population). Peptide families
are connected components under the v25 axis (ungapped identity >=0.3 at 0.8
both-coverage, 4-mer seeded). Split assigns whole families to eval or train;
no family spans both sides.

Frozen before preregistration freeze; no generation, no labels, no fitting.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest

OUT = ROOT / 'results/pae_analog_family_split_v27'
V25 = ROOT / 'results/pae_analog_overlap_v25'
V26 = ROOT / 'results/pae_analog_structures_v26'
AXES = {'identity': 0.3, 'coverage': 0.8}
SEED = 4
MIN_EVAL_COMPONENTS = 6
MIN_EVAL_FAMILIES = 3
MAX_EVAL_COMPONENTS = 12
MAX_TRAIN_COMPONENTS = 12
MIN_TRAIN_COMPONENTS = 10


def best_overlap(a, b):
    if not a or not b or min(len(a), len(b)) / max(len(a), len(b)) < AXES['coverage']:
        return None
    best = None
    for offset in range(-len(b) + 1, len(a)):
        start, stop = max(0, offset), min(len(a), offset + len(b))
        n = stop - start
        if n / len(a) < AXES['coverage'] or n / len(b) < AXES['coverage']:
            continue
        matches = sum(a[i] == b[i - offset] for i in range(start, stop))
        hit = {'identity': matches / n}
        if best is None or hit['identity'] > best['identity']:
            best = hit
    return best


class SeedIndex:
    def __init__(self, sequences):
        self.sequences = sequences
        self.index = {}
        for i, seq in enumerate(sequences):
            for j in range(len(seq) - SEED + 1):
                self.index.setdefault(seq[j:j + SEED], []).append(i)

    def candidates(self, query):
        found = set()
        for j in range(len(query) - SEED + 1):
            found.update(self.index.get(query[j:j + SEED], ()))
        return sorted(found)


def connected_components(nodes, edges):
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups = {}
    for n in nodes:
        groups.setdefault(find(n), []).append(n)
    return sorted(groups.values(), key=lambda g: min(g))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read(V26 / 'manifest.json')
    passed = [r for r in manifest['records'] if r['passes']]
    exposure = read(V25 / 'audit.json')
    flagged = {e['pdb'].lower() for e in exposure['entries']
               if e['pdb_exact_exposed'] or e['pdb_in_reference_union'] or e['receptor_axis_flags']}

    # Deduplicate: one representative instance per (pdb, receptor entity, peptide sequence).
    by_key = {}
    for r in passed:
        key = (r['pdb_id'].lower(), r['entity_ids']['receptor'], r['sequences']['peptide'])
        if key not in by_key or r['instance'] < by_key[key]['instance']:
            by_key[key] = r
    components = sorted(by_key.values(), key=lambda r: r['instance'])

    confirmatory = [r for r in components if r['pdb_id'].lower() not in flagged]
    # The VHH cohort collapses to one clean component after exposure review;
    # it is recorded as a descriptive pilot and excluded from confirmatory
    # splits. Confirmatory population is TCR-only.
    confirmatory_tcr = [r for r in confirmatory if r['cohort'] == 'tcr']
    vhh_pilot = [r for r in confirmatory if r['cohort'] == 'vhh']
    excluded = [r['instance'] for r in components if r['pdb_id'] in flagged]

    peptides = sorted({r['sequences']['peptide'] for r in confirmatory_tcr})
    index = SeedIndex(peptides)
    edges = []
    for i, a in enumerate(peptides):
        for j in index.candidates(a):
            if j <= i:
                continue
            b = peptides[j]
            hit = best_overlap(a, b)
            if hit and hit['identity'] >= AXES['identity']:
                edges.append((a, b))
    families = connected_components(peptides, edges)
    family_of = {}
    for k, group in enumerate(families):
        for seq in group:
            family_of[seq] = f'F{k:03d}'
    family_components = {}
    for r in confirmatory_tcr:
        family_components.setdefault(family_of[r['sequences']['peptide']], []).append(r['instance'])

    # Deterministic eval selection: smallest families first (singletons give
    # the most independent families per component), stopping once both
    # minimums are met; the cap guards against one large family flooding eval.
    ordered = sorted(family_components, key=lambda f: (len(family_components[f]), f))
    eval_families = []
    eval_components = []
    for family in ordered:
        if len(eval_components) >= MIN_EVAL_COMPONENTS and len(eval_families) >= MIN_EVAL_FAMILIES:
            break
        size = len(family_components[family])
        if eval_components and len(eval_components) + size > MAX_EVAL_COMPONENTS:
            continue
        eval_families.append(family)
        eval_components.extend(family_components[family])

    train_pool = [instance for f, members in sorted(family_components.items())
                  if f not in eval_families
                  for instance in members]
    train_components = sorted(train_pool)[:MAX_TRAIN_COMPONENTS]

    summary = {
        'population': {'v26_passed_components': len(passed),
                       'deduplicated': len(components),
                       'exposure_excluded': len(excluded),
                       'confirmatory': len(confirmatory)},
        'families': {'count': len(families),
                     'components_per_family': {f: len(m) for f, m in sorted(family_components.items())}},
        'split': {'eval_families': eval_families, 'eval_components': len(eval_components),
                  'train_families': sorted(set(family_of[r['sequences']['peptide']]
                                               for r in confirmatory_tcr
                                               if r['instance'] in train_components)),
                  'train_components': len(train_components),
                  'vhh_pilot_components': len(vhh_pilot),
                  'tcr_confirmatory_components': len(confirmatory_tcr)},
        'minimums': {'eval_components': MIN_EVAL_COMPONENTS, 'eval_families': MIN_EVAL_FAMILIES,
                     'met': len(eval_components) >= MIN_EVAL_COMPONENTS and len(eval_families) >= MIN_EVAL_FAMILIES},
    }
    protocol = {
        'classification': 'family_disjoint_split_freeze_not_model_validation',
        'axes': AXES, 'seed_method': '4-mer seeded best_overlap, connected components',
        'exposure_rule': 'exclude entries with any v25 flag from the confirmatory population',
        'cohort_rule': 'confirmatory population is TCR-only; VHH components recorded as descriptive pilot (one clean component, below every minimum)',
        'dedup_rule': 'one representative per (pdb, receptor entity, peptide sequence)',
        'caps': {'max_eval_components': MAX_EVAL_COMPONENTS, 'max_train_components': MAX_TRAIN_COMPONENTS,
                 'min_train_components': MIN_TRAIN_COMPONENTS},
        'deterministic_selection': 'families by (size asc, key); stop when minimums met; skip families that overflow eval cap',
        'source_sha256': {'v25_audit': digest(V25 / 'audit.json'), 'v26_manifest': digest(V26 / 'manifest.json')},
        'script_sha256': digest(Path(__file__)),
    }
    freeze(OUT / 'protocol.json', protocol)
    freeze(OUT / 'split.json', {'summary': summary,
                                'eval_families': eval_families,
                                'eval_components': sorted(eval_components),
                                'train_components': sorted(train_components),
                                'excluded_exposure': sorted(excluded),
                                'family_of': family_of,
                                'components': [{'instance': r['instance'], 'cohort': r['cohort'],
                                                'pdb_id': r['pdb_id'],
                                                'peptide': r['sequences']['peptide'],
                                                'family': family_of.get(r['sequences']['peptide'], 'unassigned'),
                                                'side': ('eval' if r['instance'] in eval_components
                                                         else 'train' if r['instance'] in train_components
                                                         else 'vhh_pilot' if r['cohort'] == 'vhh'
                                                         else 'train_unused' if r['pdb_id'].lower() not in flagged
                                                         else 'exposure_excluded')}
                                               for r in components]})
    freeze(OUT / 'summary.json', summary)
    freeze(OUT / 'completion.json', {'status': 'completed', 'generation_started': False,
                                     'af2_started': False, 'split_sha256': digest(OUT / 'split.json')})
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
