#!/usr/bin/env python3
"""Build split-scoped, non-circular sequence disorder profile artifacts.

The current clustered Phase3 records do not contain experimental B-factors,
UniProt mappings, or conformational ensembles. This builder therefore uses a
versioned charge-hydropathy heuristic and records that source explicitly. It
must not be described as IUPred output or experimental ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import lmdb
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
AA = "ACDEFGHIKLMNPQRSTVWY"
KD = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}
CHARGE = {'R': 1.0, 'K': 1.0, 'D': -1.0, 'E': -1.0, 'H': 0.1}
SOURCE = "charge_hydropathy_heuristic_v1"


def sequence_profile(aa_indices, window=15):
    letters = [AA[int(index)] if 0 <= int(index) < 20 else 'G' for index in aa_indices]
    hydropathy = np.asarray([(KD[aa] + 4.5) / 9.0 for aa in letters], dtype=np.float64)
    charge = np.asarray([CHARGE.get(aa, 0.0) for aa in letters], dtype=np.float64)
    output = np.zeros(len(letters), dtype=np.float64)
    half = window // 2
    for index in range(len(letters)):
        start, end = max(0, index - half), min(len(letters), index + half + 1)
        raw = 6.0 * (np.abs(charge[start:end]).mean() - hydropathy[start:end].mean()) + 0.5
        output[index] = 1.0 / (1.0 + np.exp(-raw))
    edge = min(5, len(output) // 4)
    if edge:
        ramp = np.linspace(0.3, 1.0, edge)
        output[:edge] *= ramp
        output[-edge:] *= ramp[::-1]
    return output.astype(np.float32)


def ids_digest(ids):
    return hashlib.sha256('\n'.join(map(str, ids)).encode('utf-8')).hexdigest()


def build_artifact(lmdb_path, split, output_path):
    ids_path = Path(f"{lmdb_path}-ids")
    ids = pickle.loads(ids_path.read_bytes())
    env = lmdb.open(str(lmdb_path), subdir=False, readonly=True, lock=False, readahead=False)
    profiles, provenance = {}, {}
    lengths = []
    with env.begin() as transaction:
        for sample_id in ids:
            raw = transaction.get(str(sample_id).encode())
            if raw is None:
                raise RuntimeError(f"LMDB key missing for {sample_id}")
            record = pickle.loads(raw)
            antigen = record.get('antigen')
            if antigen is None or 'aa' not in antigen:
                raise RuntimeError(f"Antigen sequence missing for {sample_id}")
            aa = antigen['aa'].numpy() if hasattr(antigen['aa'], 'numpy') else np.asarray(antigen['aa'])
            profile = sequence_profile(aa)
            profiles[str(sample_id)] = profile
            lengths.append(len(profile))
            provenance[str(sample_id)] = {
                'source': SOURCE,
                'pdb_id': str(record.get('pdb_id', record.get('id', sample_id))),
                'length': len(profile),
                'window': 15,
                'experimental_ground_truth': False,
                'model_generated': False,
            }
    env.close()
    means = np.asarray([profile.mean() for profile in profiles.values()])
    artifact = {
        'schema_version': 1,
        'split': split,
        'lmdb_path': str(Path(lmdb_path).resolve()),
        'ids_path': str(ids_path.resolve()),
        'ids_sha256': ids_digest(ids),
        'source_contract': {
            'primary_source': SOURCE,
            'claim_scope': 'non-circular sequence-derived weak profile',
            'unsupported_claims': ['experimental disorder label', 'IUPred prediction'],
        },
        'profiles': profiles,
        'provenance': provenance,
        'stats': {
            'n_ids': len(ids),
            'n_profiles': len(profiles),
            'length_min': min(lengths),
            'length_max': max(lengths),
            'mean_profile_quantiles': {
                str(q): float(np.quantile(means, q)) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
            },
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL))
    output_path.with_suffix('.audit.json').write_text(
        json.dumps({key: value for key, value in artifact.items() if key not in ('profiles', 'provenance')}, indent=2),
        encoding='utf-8',
    )
    print(json.dumps({'split': split, **artifact['stats'], 'output': str(output_path)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lmdb', required=True, type=Path)
    parser.add_argument('--split', required=True, choices=('train', 'val'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    build_artifact(args.lmdb, args.split, args.output)


if __name__ == '__main__':
    main()
