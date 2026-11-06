"""FLEX-BINDER Phase A2 leg-2 v2: recompute the ATLAS MD-RMSF vs pLDDT audit
on RAW track values.

Deviation disclosure: v1 computed Spearman on the tracks' normalised ``value``
fields. The ATLAS website documents pLDDT as "inverted for visualisation", and
the v1 median of -0.71 (RMSF positively tracking the inverted-normalised
pLDDT) is exactly the signature of that display convention - a physically
impossible direction if taken at face value. The ``featureId`` payload carries
raw values (RMSF in Angstrom, pLDDT 0-100); v2 recomputes from those, per
protein, over cached tracks only (no network). v1 artifacts are kept as the
record of the direction mis-specification.

Per-protein min-max normalisation is rank-preserving, so RMSF raw vs value is
equivalent within a protein; the pLDDT inversion is not, and is the point of
this correction.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import freeze, digest
from scripts.flex_a2_atlas_nmr_audit_v1 import average_ranks

OUT = ROOT / 'results/flex_a2_atlas_plddt_audit_v1'
OUT2 = ROOT / 'results/flex_a2_atlas_plddt_audit_v2'
RAW = re.compile(r'^value:\s*([0-9.]+)')


def raw_series(track):
    values = []
    positions = []
    for entry in track:
        match = RAW.match(entry.get('featureId') or '')
        if not match:
            return None, None
        values.append(float(match.group(1)))
        positions.append(entry['begin'])
    return values, positions


def spearman(a, b):
    if len(a) < 10 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(average_ranks(np.asarray(a, dtype=float)),
                             average_ranks(np.asarray(b, dtype=float)))[0, 1])


def main():
    OUT2.mkdir(parents=True, exist_ok=True)
    track_files = sorted((OUT / 'tracks').glob('*.js'))
    freeze(OUT2 / 'protocol.json', {
        'classification': 'label_relevance_audit_corrected_direction',
        'v1_summary_sha256': digest(OUT / 'summary.json'),
        'deviation_disclosure': 'v1 used normalised display values; plddtTrack is '
                                'inverted for visualisation per the ATLAS site. '
                                'v2 recomputes on raw featureId values from the '
                                'same cached tracks; v1 artifacts kept.',
        'metric': 'per-protein Spearman(raw MD RMSF, -raw pLDDT)',
        'expected_direction': 'positive (low pLDDT at high flexibility)',
        'n_tracks': len(track_files),
        'script_sha256': digest(Path(__file__))})

    results = []
    for path in track_files:
        pdb_id = path.stem
        try:
            data = json.loads(path.read_text(encoding='utf-8', errors='replace')
                              .split('=', 1)[1].strip().rstrip(';'))
            rmsf, rmsf_pos = raw_series(data.get('rmsfTrack') or [])
            plddt, plddt_pos = raw_series(data.get('plddtTrack') or [])
            if not rmsf or not plddt:
                results.append({'pdb_id': pdb_id, 'status': 'unparsable_featureId'})
                continue
            if rmsf_pos != plddt_pos:
                results.append({'pdb_id': pdb_id, 'status': 'position_mismatch'})
                continue
            rho = spearman(rmsf, [-p for p in plddt])
            results.append({'pdb_id': pdb_id,
                            'status': 'audited' if rho is not None else 'insufficient_variation',
                            'n_residues': len(rmsf),
                            'spearman_rmsf_neg_plddt_raw': rho,
                            'rmsf_mean': round(float(np.mean(rmsf)), 4),
                            'plddt_mean': round(float(np.mean(plddt)), 2)})
        except Exception as error:
            results.append({'pdb_id': pdb_id, 'status': f'error: {error}'})

    rhos = np.array([r['spearman_rmsf_neg_plddt_raw'] for r in results
                     if r.get('spearman_rmsf_neg_plddt_raw') is not None])
    summary = {
        'tracks': len(track_files),
        'audited': len(rhos),
        'rho_median': float(np.median(rhos)) if len(rhos) else None,
        'rho_iqr': ([float(np.percentile(rhos, 25)), float(np.percentile(rhos, 75))]
                    if len(rhos) else None),
        'rho_percentiles_10_90': ([float(np.percentile(rhos, 10)),
                                   float(np.percentile(rhos, 90))]
                                  if len(rhos) else None),
        'fraction_rho_ge_0.30': float(np.mean(rhos >= 0.30)) if len(rhos) else None,
        'fraction_rho_ge_0.50': float(np.mean(rhos >= 0.50)) if len(rhos) else None,
        'classification': 'descriptive_distribution_not_gate_claim',
    }
    freeze(OUT2 / 'per_protein.json', {'results': results})
    freeze(OUT2 / 'summary.json', summary)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
