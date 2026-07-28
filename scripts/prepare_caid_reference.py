#!/usr/bin/env python3
"""Convert an official CAID three-line reference FASTA for benchmark_caid.py."""

import argparse
import hashlib
import json
from pathlib import Path


def load_reference(path):
    lines = [line.strip() for line in Path(path).read_text(encoding='utf-8').splitlines()
             if line.strip()]
    records = []
    index = 0
    while index < len(lines):
        if not lines[index].startswith('>') or index + 2 >= len(lines):
            raise ValueError(f'Invalid CAID record near line {index + 1}')
        target_id = lines[index][1:].split()[0]
        sequence = lines[index + 1].upper()
        labels = lines[index + 2]
        if len(sequence) != len(labels):
            raise ValueError(
                f'{target_id}: sequence length {len(sequence)} != labels {len(labels)}')
        records.append((target_id, sequence, labels))
        index += 3
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference')
    parser.add_argument('output_dir')
    parser.add_argument('--source-url', required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()

    reference = Path(args.reference)
    output = Path(args.output_dir)
    labels_dir = output / 'labels'
    labels_dir.mkdir(parents=True, exist_ok=False)
    records = load_reference(reference)

    with open(output / 'sequences.fasta', 'w', encoding='ascii') as fasta:
        for target_id, sequence, labels in records:
            fasta.write(f'>{target_id}\n{sequence}\n')
            normalized = ['nan' if label == '-' else label for label in labels]
            with open(labels_dir / f'{target_id}.label', 'w', encoding='ascii') as handle:
                handle.write(' '.join(normalized) + '\n')

    digest = hashlib.sha256(reference.read_bytes()).hexdigest()
    metadata = {
        'name': args.name,
        'source_url': args.source_url,
        'source_file': str(reference),
        'sha256': digest,
        'n_sequences': len(records),
        'n_residues': sum(len(sequence) for _, sequence, _ in records),
        'n_positive': sum(labels.count('1') for _, _, labels in records),
        'n_negative': sum(labels.count('0') for _, _, labels in records),
        'n_undefined': sum(labels.count('-') for _, _, labels in records),
        'usage': 'blind external evaluation only; never use for training',
    }
    with open(output / 'metadata.json', 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
