#!/usr/bin/env python3
"""Select a mutation-pattern-diverse N52H H3 panel for absolute structure gates."""

import argparse
import csv
import json
from pathlib import Path


def mutation_positions(text):
    return tuple(sorted(int(token.split(':')[1]) for token in text.split(';') if token))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--developability', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--max-candidates', type=int, default=6)
    args = parser.parse_args()
    with args.developability.open(newline='', encoding='utf-8') as handle:
        rows = [row for row in csv.DictReader(handle)
                if row['developability_status'] == 'developability_pass'
                and row['nas_status'] == 'rescued_N52H']
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    sequences = {row['construct_id']: row for row in manifest['records']}
    selected, patterns = [], set()
    for row in rows:
        pattern = mutation_positions(row['h3_mutations'])
        if pattern in patterns:
            continue
        selected.append(row)
        patterns.add(pattern)
        if len(selected) == args.max_candidates:
            break
    if len(selected) < args.max_candidates:
        for row in rows:
            if row not in selected:
                selected.append(row)
            if len(selected) == args.max_candidates:
                break
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'structure_panel.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=selected[0].keys())
        writer.writeheader()
        writer.writerows(selected)
    with (args.output_dir / 'fab.fasta').open('w', encoding='ascii') as fab, \
            (args.output_dir / 'fab_abeta11.fasta').open('w', encoding='ascii') as short, \
            (args.output_dir / 'fab_abeta42.fasta').open('w', encoding='ascii') as full:
        for row in selected:
            record = sequences[row['construct_id']]
            chains = f"{record['heavy_sequence']}:{record['light_sequence']}"
            fab.write(f">{row['construct_id']}\n{chains}\n")
            short.write(f">{row['construct_id']}\n{chains}:DAEFRHDSGYE\n")
            full.write(f">{row['construct_id']}\n{chains}:DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA\n")
    report = {
        'selection_contract': 'developability pass, fixed N52H background, mutation-position diversity',
        'ranking_contract': 'multimer ipTM is an absolute rejection gate only and never reorders passing candidates',
        'records': [dict(row) for row in selected],
    }
    (args.output_dir / 'structure_panel.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'selected': len(selected), 'patterns': [list(p) for p in patterns]}, indent=2))


if __name__ == '__main__':
    main()
