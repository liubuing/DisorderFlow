"""FLEX-BINDER Phase A2 (NMR side): per-residue RMSF from solution-NMR ensembles.

Computes C-alpha RMSF over models for each NMR ensemble in the local
calibration set, after per-model global Kabsch alignment to the first model
(same alignment convention as the v5 AFDB conformation build). Output is the
calibration reference for auditing ATLAS MD RMSF as a replacement label source
for the AF2-seed RMSF proxy.

No model training, no AF2, no claims: label infrastructure only.
"""
import json
import sys
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import freeze, digest

SRC = ROOT / 'data/calibration_nmr_pdb'
OUT = ROOT / 'results/flex_a2_nmr_rmsf_v1'
MIN_MODELS = 5
MIN_RESIDUES = 20
AA = set('ACDEFGHIKLMNPQRSTVWY')


def kabsch(mobile, reference):
    """Rotation+translation aligning mobile onto reference (both (N,3))."""
    mc = mobile - mobile.mean(axis=0)
    rc = reference - reference.mean(axis=0)
    cov = mc.T @ rc
    u, _s, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return mc @ rotation + reference.mean(axis=0)


def parse_entry(path):
    structure = PDBParser(QUIET=True).get_structure(path.stem, str(path))
    models = list(structure.get_models())
    if len(models) < MIN_MODELS:
        return None
    per_model = []
    for model in models:
        cas = {}
        for chain in model:
            for residue in chain:
                if residue.id[0] != ' ' or 'CA' not in residue:
                    continue
                aa = seq1(residue.resname, custom_map={'MSE': 'M'})
                if aa not in AA:
                    continue
                key = (chain.id, residue.id[1], residue.id[2].strip())
                cas[key] = (aa, np.asarray(residue['CA'].coord, dtype=float))
        per_model.append(cas)
    common = set(per_model[0])
    for cas in per_model[1:]:
        common &= set(cas)
    common = sorted(common)
    if len(common) < MIN_RESIDUES:
        return None
    reference = np.stack([per_model[0][k][1] for k in common])
    coords = np.stack([np.stack([cas[k][1] for k in common]) for cas in per_model])
    aligned = np.stack([kabsch(frame, reference) for frame in coords])
    rmsf = np.linalg.norm(aligned.std(axis=0, ddof=0), axis=-1)
    sequence = ''.join(per_model[0][k][0] for k in common)
    return {
        'pdb': path.stem.lower(),
        'n_models': len(models),
        'n_common_ca': len(common),
        'sequence': sequence,
        'rmsf': [round(float(v), 4) for v in rmsf],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    skipped = {}
    for path in sorted(SRC.glob('*.pdb')):
        try:
            entry = parse_entry(path)
        except Exception as error:
            entry = None
            skipped[path.stem] = f'parse_failed: {error}'
        if entry is None:
            skipped.setdefault(path.stem, 'below_model_or_residue_minimum')
            continue
        entries.append(entry)

    rmsf_all = np.concatenate([np.array(e['rmsf']) for e in entries])
    summary = {
        'source_dir': str(SRC),
        'source_note': 'NMR calibration set from the v5.1 physical-label gate',
        'entries': len(entries),
        'skipped': skipped,
        'models_per_entry_median': float(np.median([e['n_models'] for e in entries])),
        'residues_per_entry_median': float(np.median([e['n_common_ca'] for e in entries])),
        'rmsf_angstrom_mean': round(float(rmsf_all.mean()), 4),
        'rmsf_angstrom_p90': round(float(np.percentile(rmsf_all, 90)), 4),
        'classification': 'label_infrastructure_not_model_result',
        'alignment': 'per-model global Kabsch to first model, C-alpha only',
        'script_sha256': digest(Path(__file__)),
    }
    freeze(OUT / 'protocol.json', {
        'classification': summary['classification'],
        'alignment': summary['alignment'],
        'min_models': MIN_MODELS, 'min_residues': MIN_RESIDUES,
        'script_sha256': summary['script_sha256']})
    freeze(OUT / 'rmsf.json', {'entries': entries})
    freeze(OUT / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'skipped'}, indent=1))
    print('skipped:', len(skipped))


if __name__ == '__main__':
    main()
