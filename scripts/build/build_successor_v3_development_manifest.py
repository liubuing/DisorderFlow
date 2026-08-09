#!/usr/bin/env python
"""Freeze exposed peptide/H3 records as successor-v3 development-only folds."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_ids(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return {row['id'] for row in csv.DictReader(handle)}


def component_fold(component, count=5):
    digest = hashlib.sha256(component.encode()).digest()
    return int.from_bytes(digest[:8], 'big') % count


def connected_component_labels(cluster_sets):
    parent = {}

    def find(item):
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for clusters in cluster_sets:
        for cluster in clusters:
            find(cluster)
        for cluster in clusters[1:]:
            union(clusters[0], cluster)
    return {cluster: find(cluster) for cluster in parent}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output', type=Path,
        default=ROOT / 'data/successor_v3_development/manifest.json')
    args = parser.parse_args()

    source_dir = ROOT / 'data/peptide_h3_publication_split_v4'
    audit_path = source_dir / 'audit.json'
    adaptation_path = source_dir / 'adaptation_manifest.csv'
    development_path = source_dir / 'development_manifest.csv'
    sealed_path = source_dir / 'sealed_final_manifest.csv'
    v2_path = ROOT / 'data/multiscaffold_confirmatory_v2/holdout_manifest.json'
    v2_analysis_path = ROOT / 'results/multiscaffold_confirmatory_v2/analysis.json'
    protocol_path = ROOT / 'configs/benchmarks/successor_v3_development.yml'
    training_path = ROOT / 'configs/train/bfn_successor_v3_h3_specificity.yml'

    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    retained = {
        'adaptation': read_ids(adaptation_path),
        'development': read_ids(development_path),
    }
    sealed_ids = read_ids(sealed_path)
    v2 = json.loads(v2_path.read_text(encoding='utf-8'))
    prohibited_ids = sealed_ids | set(v2['representative_ids'])

    source_records = []
    for source_class in ('adaptation', 'development'):
        for record in audit['records'][source_class]:
            if record['id'] not in retained[source_class]:
                continue
            clusters = record['axis_values']['official_antigen_cluster']
            if not clusters:
                raise RuntimeError(f"{record['id']} has no antigen component")
            if record['id'] in prohibited_ids:
                raise RuntimeError(f"Prohibited record leaked into development: {record['id']}")
            source_records.append((source_class, record, clusters))

    component_labels = connected_component_labels(
        [clusters for _, _, clusters in source_records])
    records = []
    for source_class, record, clusters in source_records:
        component = component_labels[clusters[0]]
        records.append({
            'id': record['id'],
            'source_class': source_class,
            'component': component,
            'fold': component_fold(component),
            'antigen_length': record['antigen_length'],
            'h3_length': record['h3_length'],
            'n_contacting_h3_positions': record['n_contacting_h3_positions'],
            'n_h3_antigen_residue_contacts': record['n_h3_antigen_residue_contacts'],
        })

    components = sorted({record['component'] for record in records})
    representatives = []
    for component in components:
        members = [record for record in records if record['component'] == component]
        members.sort(key=lambda item: (
            -item['n_contacting_h3_positions'],
            -item['n_h3_antigen_residue_contacts'],
            item['id'],
        ))
        representatives.append(members[0]['id'])

    payload = {
        'schema_version': 1,
        'status': 'frozen_development_only; not confirmatory',
        'created_by': 'scripts/build/build_successor_v3_development_manifest.py',
        'source_hashes': {
            str(path.relative_to(ROOT)).replace('\\', '/'): sha256(path)
            for path in (
                audit_path, adaptation_path, development_path, sealed_path,
                v2_path, v2_analysis_path, protocol_path, training_path,
            )
        },
        'counts': {
            'records': len(records),
            'components': len(components),
            'representatives': len(representatives),
            'fold_components': {
                str(fold): sum(component_fold(item) == fold for item in components)
                for fold in range(5)
            },
        },
        'components_sha256': hashlib.sha256(
            '\n'.join(components).encode()).hexdigest(),
        'representative_ids_sha256': hashlib.sha256(
            '\n'.join(representatives).encode()).hexdigest(),
        'prohibited_overlap': {
            'sealed_final_ids': 0,
            'v2_representative_ids': 0,
        },
        'components': components,
        'representative_ids': representatives,
        'records': sorted(records, key=lambda item: (item['component'], item['id'])),
    }
    if payload['counts']['components'] < 12:
        raise RuntimeError('Successor development panel has fewer than 12 components')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload['counts'], indent=2))


if __name__ == '__main__':
    main()
