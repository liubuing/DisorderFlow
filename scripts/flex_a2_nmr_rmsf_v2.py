"""FLEX A2 (NMR side) v2: per-residue RMSF from solution-NMR ensembles, REPAIR.

Recomputes results/flex_a2_nmr_rmsf_v1 with a corrected Kabsch superposition.
The legacy function applied the transposed rotation (a perfect rigid motion
yields 5-9 A RMSD instead of 0), inflating every RMSF value with misalignment
noise; finding disclosed in docs/CSN_CAMPAIGN_V1.md section 6 and pinned by the
canary test in tests/test_csn_c0_contracts.py.

This script is self-contained (does not modify or import the frozen v1 script)
and freezes a disclosure comparing v1 vs v2 per entry. No model training, no
AF2, no claims: label infrastructure repair only.
"""
import json
import sys
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_aayl_original_endpoint_cohort import freeze  # noqa: E402
from scripts.benchmark_aayl_learnability import digest  # noqa: E402

SRC = ROOT / 'data/calibration_nmr_pdb'
V1 = ROOT / 'results/flex_a2_nmr_rmsf_v1/rmsf.json'
OUT = ROOT / 'results/flex_a2_nmr_rmsf_v2'
MIN_MODELS = 5
MIN_RESIDUES = 20
AA = set('ACDEFGHIKLMNPQRSTVWY')
DATE = '2026-11-08'
CLASSIFICATION = 'label_infrastructure_repair_not_model_result'


def kabsch_row(mobile, reference):
    """Row-convention Kabsch: aligned = centered_mobile @ R + ref_mean, with
    R = U diag(1,1,d) V^T from SVD of centered_mobile^T centered_reference.
    Unit-tested against rigid roundtrips (tests/test_csn_c0_contracts.py)."""
    mc = mobile - mobile.mean(axis=0)
    rc = reference - reference.mean(axis=0)
    u, _s, vt = np.linalg.svd(mc.T @ rc)
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, d]) @ vt
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
    aligned = np.stack([kabsch_row(frame, reference) for frame in coords])
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
        except Exception as error:  # noqa: BLE001 - repair rerun, every failure recorded
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
        'classification': CLASSIFICATION,
        'alignment': ('per-model global Kabsch (corrected row convention) to first '
                      'model, C-alpha only; unit-tested rigid roundtrips'),
        'script_sha256': digest(Path(__file__)),
        'repair_note': ('v2 of results/flex_a2_nmr_rmsf_v1: legacy kabsch applied the '
                        'transposed rotation, inflating RMSF with misalignment noise; '
                        'v1 artifacts remain write-once preserved and are superseded '
                        'for interpretive use'),
    }
    freeze(OUT / 'protocol.json', {
        'classification': CLASSIFICATION, 'alignment': summary['alignment'],
        'min_models': MIN_MODELS, 'min_residues': MIN_RESIDUES,
        'script_sha256': summary['script_sha256']})
    freeze(OUT / 'rmsf.json', {'entries': entries})
    freeze(OUT / 'summary.json', summary)

    # v1-vs-v2 disclosure: per-entry Pearson r and residue-level ratio.
    v1 = json.loads(V1.read_text(encoding='utf-8'))
    v1_by_pdb = {e['pdb']: e for e in v1['entries']}
    comparisons = []
    for e in entries:
        old = v1_by_pdb.get(e['pdb'])
        if not old or len(old['rmsf']) != len(e['rmsf']):
            continue
        a = np.asarray(old['rmsf'], dtype=float)
        b = np.asarray(e['rmsf'], dtype=float)
        comparisons.append({
            'pdb': e['pdb'],
            'v1_mean': round(float(a.mean()), 4),
            'v2_mean': round(float(b.mean()), 4),
            'median_ratio_v1_over_v2': round(float(np.median(a / np.maximum(b, 1e-6))), 4),
            'pearson_r': round(float(np.corrcoef(a, b)[0, 1]), 4),
        })
    ratios = np.asarray([c['median_ratio_v1_over_v2'] for c in comparisons])
    rs = np.asarray([c['pearson_r'] for c in comparisons])
    freeze(OUT / 'v1_disclosure.json', {
        'classification': CLASSIFICATION, 'date': DATE,
        'n_compared': len(comparisons),
        'median_ratio_v1_over_v2': round(float(np.median(ratios)), 4) if len(ratios) else None,
        'p10_ratio': round(float(np.percentile(ratios, 10)), 4) if len(ratios) else None,
        'p90_ratio': round(float(np.percentile(ratios, 90)), 4) if len(ratios) else None,
        'pearson_r_median': round(float(np.median(rs)), 4) if len(rs) else None,
        'reading': ('v1 RMSF should exceed v2 systematically (misalignment noise); '
                    'per-residue rank correlation quantifies how much residue-level '
                    'order information survived the bug'),
        'comparisons': comparisons,
        'downstream_flags': [
            'results/flex_a2_atlas_nmr_audit_v1 (Spearman vs ATLAS RMSF) used v1 as '
            'the NMR side; verdict requires re-examination against v2',
            'IDP ensemble pose-diversity audits used the same legacy function; '
            'diversity numbers are inflated by misalignment noise',
        ],
    })
    print(json.dumps({k: v for k, v in summary.items() if k != 'skipped'}, indent=1))
    print('skipped:', len(skipped), '| v1 comparison entries:', len(comparisons))


if __name__ == '__main__':
    main()
