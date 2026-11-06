"""FLEX-BINDER Phase A1b-v2: corrected isolation interpretation and family split.

Deviation disclosure (recorded in protocol.json): v1 applied the v5
disorder-label homology gate (30% identity / 80% coverage) to immunoglobulin
domains, where Ig-frame conservation alone puts any two VHs above 30% - every
sequence chained into one cluster and 48/56 heavies flagged as "violations"
at median identity 0.796. v2 inspects the frozen v1 identity distributions and
applies domain-aware rules, declared before recomputation:

- hard violation: candidate PDB also present in the ECLS-line universe
  (v4 manifests + local post-2021 structure pool) - same structure, not
  independent data;
- heavy lineage violation/merge: VH identity >= 0.90 (clone level; identities
  0.7-0.85 represent frame conservation, not shared lineage);
- peptide family merge: identity >= 0.50 / coverage >= 0.80 (identity on
  5-15-mers is inflated; 30% of a 6-mer is 2 residues).

Pure post-processing of frozen artifacts (v1 isolation.json); no network.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest
from scripts.flex_a1b_isolation_families_v1 import alignment_identity

OUT = ROOT / 'results/flex_a1b_isolation_families_v1'
OUT2 = ROOT / 'results/flex_a1b_isolation_families_v2'
A1 = ROOT / 'results/flex_a1_scaffold_audit_v1/manifest.json'

HEAVY_LINEAGE_IDENTITY = 0.90
PEPTIDE_FAMILY_IDENTITY = 0.50
PEPTIDE_FAMILY_COVERAGE = 0.80


def single_linkage(keys, pairwise_ok):
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if pairwise_ok(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
    clusters = {}
    for key in keys:
        clusters.setdefault(find(key), []).append(key)
    return sorted(clusters.values(), key=len, reverse=True)


def main():
    OUT2.mkdir(parents=True, exist_ok=True)
    v1 = read(OUT / 'isolation.json')
    a1 = read(A1)
    passing = [r for r in a1['records'] if r['passes']]

    # Universe PDB set from the frozen v1 reference index.
    universe_pdbs = set()
    for bucket in ('heavy_like_unique_seqs', 'peptide_like_unique_seqs'):
        for entry in v1[bucket].values():
            for ref in entry['references']:
                universe_pdbs.add(ref['pdb'])
    my_pdbs = sorted({r['pdb_id'] for r in passing})
    same_pdb = sorted(set(my_pdbs) & universe_pdbs)
    clean_pdbs = sorted(set(my_pdbs) - universe_pdbs)

    freeze(OUT2 / 'protocol.json', {
        'classification': 'corrected_isolation_interpretation',
        'v1_isolation_sha256': digest(OUT / 'isolation.json'),
        'deviation_disclosure': 'v1 used the v5 disorder-label 30%/80% gate on Ig '
                                'domains where frame conservation alone exceeds it; '
                                'v2 thresholds were declared after inspecting the '
                                'frozen v1 identity distributions (median flagged '
                                'heavy identity 0.796) and before recomputation',
        'rules': {'same_pdb': 'hard exclusion',
                  'heavy_lineage_identity': HEAVY_LINEAGE_IDENTITY,
                  'peptide_family_identity': PEPTIDE_FAMILY_IDENTITY,
                  'peptide_family_coverage': PEPTIDE_FAMILY_COVERAGE},
        'script_sha256': digest(Path(__file__))})

    # Same-lineage heavy violations from the frozen best-match identities.
    heavy_best = {v['instances'][0]: v for v in v1['isolation']['heavy_violations']}
    heavy_lineage_hits = {}
    for record in passing:
        instance = record['instance']
        best = heavy_best.get(instance)
        if best and best['identity'] >= HEAVY_LINEAGE_IDENTITY:
            heavy_lineage_hits[record['pdb_id']] = {
                'identity': best['identity'],
                'reference_pdbs': [r['pdb'] for r in best['references']]}

    # Antigen-level violations (30%/80% gate is appropriate for short antigens
    # vs the general-protein universe, excluding same-PDB artifacts from the
    # count and listing them separately).
    antigen_best = {tuple(v['instances']): v
                    for v in v1['isolation']['antigen_violations']}
    antigen_hit_pdbs = set()
    for record in passing:
        best = antigen_best.get((record['instance'],)) \
            or antigen_best.get((record['instance'], record['instance']))
        if best:
            ref_pdbs = {r['pdb'] for r in best['references']}
            if not (ref_pdbs & {record['pdb_id']}):
                antigen_hit_pdbs.add(record['pdb_id'])

    # Corrected family split among clean PDBs.
    clean_records = [r for r in passing if r['pdb_id'] not in same_pdb]
    peptide_seqs = {}
    heavy_seqs = {}
    for record in clean_records:
        peptide_seqs.setdefault(record['sequences']['peptide'], []).append(record['pdb_id'])
        heavy_seqs.setdefault(record['sequences']['heavy'], []).append(record['pdb_id'])

    def peptide_pair(a, b):
        identity, cov_a, cov_b = alignment_identity(a, b)
        return (identity >= PEPTIDE_FAMILY_IDENTITY
                and (cov_a >= PEPTIDE_FAMILY_COVERAGE or cov_b >= PEPTIDE_FAMILY_COVERAGE))

    def heavy_pair(a, b):
        identity, cov_a, cov_b = alignment_identity(a, b)
        return identity >= HEAVY_LINEAGE_IDENTITY and cov_a >= 0.95 and cov_b >= 0.95

    peptide_keys = sorted(peptide_seqs)
    heavy_keys = sorted(heavy_seqs)
    peptide_clusters = single_linkage(peptide_keys, peptide_pair)
    heavy_clusters = single_linkage(heavy_keys, heavy_pair)

    summary = {
        'passing_pdbs_v1': len(my_pdbs),
        'same_pdb_excluded': len(same_pdb),
        'same_pdb_list': same_pdb,
        'clean_pdbs': len(clean_pdbs),
        'heavy_lineage_violation_pdbs': sorted(heavy_lineage_hits),
        'heavy_lineage_violation_count': len(heavy_lineage_hits),
        'antigen_violation_pdbs_excl_same_pdb': sorted(antigen_hit_pdbs),
        'clean_unique_peptide_seqs': len(peptide_keys),
        'clean_unique_heavy_seqs': len(heavy_keys),
        'peptide_families': len(peptide_clusters),
        'peptide_family_sizes': [len(c) for c in peptide_clusters],
        'heavy_lineages': len(heavy_clusters),
        'heavy_lineage_sizes': [len(c) for c in heavy_clusters],
        'boundary': 'family counts are for power statements only; not independence '
                    'proofs and not binding evidence',
    }
    freeze(OUT2 / 'families.json', {
        'peptide_clusters': [sorted(c) for c in peptide_clusters],
        'heavy_clusters': [sorted(c) for c in heavy_clusters]})
    freeze(OUT2 / 'summary.json', summary)
    print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
