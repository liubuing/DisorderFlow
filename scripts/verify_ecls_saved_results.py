"""Verify ECLS statistics from saved scores only; never invoke a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import csv
from collections import Counter
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_scores(payload, seed, assignments=None):
    rows = payload["results"]
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate structure IDs")
    for row in rows:
        controls = row["composition_shuffles"]
        native = row["native"]
        possible = math.factorial(len(native["sequence"]))
        for count in Counter(native["sequence"]).values():
            possible //= math.factorial(count)
        if len(controls) != min(200, possible - 1):
            raise ValueError("Composition-control count differs from the frozen capped target")
        sequences = [r["sequence"] for r in controls]
        if len(set(sequences)) != len(sequences) or native["sequence"] in sequences:
            raise ValueError("Controls must be unique non-native permutations")
        for scored in [native, *controls]:
            if sorted(scored["sequence"]) != sorted(native["sequence"]):
                raise ValueError("Control composition mismatch")
            delta = scored["complex_h3_nll"] - scored["apo_h3_nll"]
            if abs(delta - scored["epitope_delta_nll"]) > 3e-6:
                raise ValueError("Paired score arithmetic mismatch")
        effect = np.mean([r["epitope_delta_nll"] for r in controls]) - native["epitope_delta_nll"]
        if abs(effect - row["summary"]["ecls_advantage"]) > 3e-6:
            raise ValueError("Record advantage mismatch")
    if "inference_units" in payload:
        units = payload["inference_units"]
    else:
        if assignments is None or set(assignments) != set(by_id):
            raise ValueError("Adaptation cluster assignments missing or incomplete")
        grouped = {}
        # The historical adaptation calculation resampled its stored record order.
        # Each selected record is one cluster; sorting cluster labels would change
        # finite Monte Carlo bootstrap draws despite using the same seed.
        for record_id in by_id:
            unit = assignments[record_id]
            grouped.setdefault(unit, []).append(record_id)
        units = [{"unit": unit, "record_ids": ids} for unit, ids in grouped.items()]
    memberships = [rid for unit in units for rid in unit["record_ids"]]
    if len(memberships) != len(set(memberships)) or set(memberships) != set(by_id):
        raise ValueError("Cluster memberships do not partition records")
    values = []
    for unit in units:
        value = float(np.mean([by_id[rid]["summary"]["ecls_advantage"] for rid in unit["record_ids"]]))
        if "mean_ecls_advantage" in unit and abs(value - unit["mean_ecls_advantage"]) > 1e-9:
            raise ValueError("Stored cluster mean mismatch")
        values.append(value)
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    observed = {
        "n_complexes": len(rows), "n_inference_units": len(values),
        "mean_ecls_advantage": float(values.mean()),
        "median_ecls_advantage": float(np.median(values)),
        "ecls_advantage_mean_ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "fraction_positive_ecls_advantage": float(np.mean(values > 0)),
    }
    for key, value in observed.items():
        if not np.allclose(value, payload["aggregate"][key], atol=1e-6, rtol=0):
            raise ValueError(f"Frozen aggregate mismatch: {key}")
    return observed


def verify_manifest(root):
    manifest = read(root / "MANIFEST.json")
    expected = {entry["path"] for entry in manifest["files"]} | {"MANIFEST.json"}
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
              and "__pycache__" not in p.parts}
    if expected != actual:
        raise ValueError("Archive has missing or unexpected files")
    for entry in manifest["files"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Manifest path escapes archive")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {entry['path']}")
    return len(manifest["files"])


def close(actual, expected, label):
    if not np.allclose(actual, expected, atol=1e-9, rtol=0):
        raise ValueError(f"Saved ranking mismatch: {label}")


def bootstrap(values, seed):
    values = np.asarray(values)
    indices = np.random.default_rng(seed).integers(0, len(values), size=(10000, len(values)))
    return np.quantile(values[indices].mean(axis=1), [0.025, 0.975]).tolist()


def native_rank(native, candidates):
    if not candidates:
        raise ValueError('Empty candidate pool')
    return 1.0 - (sum(x < native for x in candidates) + .5 * sum(x == native for x in candidates)) / len(candidates)


def verify_ranking(payload, calibration):
    """Recompute ranks and the three plotted pooled intervals from saved scores."""
    grouped = {}
    seen = set()
    for row in payload['results']:
        key = row['generator'], row['unit']
        if (*key, row['seed']) in seen:
            raise ValueError('Duplicate generator/cluster/seed')
        seen.add((*key, row['seed']))
        scores = row['native_scores']
        candidates = row['candidates']
        sequence_set = {r['sequence'] for r in candidates}
        if len(sequence_set) != len(candidates) or row['native_sequence'] in sequence_set:
            raise ValueError('Candidate pool contains duplicates or native sequence')
        for field in ('ecls', 'complex_nll', 'apo_nll'):
            rank = native_rank(scores[field], [r[field] for r in candidates])
            close(rank, row['metrics'][field + '_nnr'], field + ' native rank')
        coefficient = calibration['coefficients'][row['generator']]
        calibrated = native_rank(scores['complex_nll'] - coefficient * scores['apo_nll'],
                                 [r['complex_nll'] - coefficient * r['apo_nll'] for r in candidates])
        grouped.setdefault(key, []).append([row['metrics']['ecls_nnr'], row['metrics']['complex_nll_nnr'], calibrated])
    if len(grouped) != 21 or any(len(rows) != 3 for rows in grouped.values()):
        raise ValueError('Expected three generators, seven clusters and three seeds')
    means = {key: np.mean(rows, axis=0) for key, rows in grouped.items()}
    development = calibration['development']
    calibrated_units = {r['unit']: r for r in development['units']}
    for generator, aggregate in payload['aggregate']['generators'].items():
        for unit in aggregate['units']:
            ecls, baseline, calibrated = means[generator, unit['unit']]
            close([ecls, baseline, ecls-baseline], [unit['ecls_nnr'], unit['complex_nll_nnr'], unit['gain']], 'generator cluster means')
            close(calibrated, calibrated_units[unit['unit']][generator], 'calibrated cluster mean')
    units = payload['aggregate']['units']
    pooled = []
    for unit in units:
        values = np.mean([means[g, unit['unit']] for g in ('bfn','esm_if','proteinmpnn')], axis=0)
        ecls, baseline, calibrated = values
        close([ecls, baseline, ecls-baseline], [unit['ecls_nnr'],unit['complex_nll_nnr'],unit['gain']], 'pooled unit')
        close(calibrated, calibrated_units[unit['unit']]['generator_aware_nnr'], 'pooled calibrated unit')
        pooled.append(values)
    pooled = np.asarray(pooled)
    gains = pooled[:,0] - pooled[:,1]
    calibrated_gains = pooled[:,2] - pooled[:,1]
    over_random = pooled[:,2] - .5
    for actual, expected, label in [
        (gains.mean(), payload['aggregate']['overall_mean_gain'], 'universal mean'),
        (bootstrap(gains,2467), payload['aggregate']['overall_gain_ci95'], 'universal interval'),
        (pooled[:,2].mean(), development['mean_generator_aware_nnr'], 'calibrated mean'),
        (calibrated_gains.mean(), development['mean_gain_over_complex_nll'], 'calibrated gain'),
        (bootstrap(calibrated_gains,2523), development['gain_over_complex_nll_ci95'], 'calibrated baseline interval'),
        (bootstrap(over_random,2522), development['generator_aware_over_random_ci95'], 'calibrated random interval'),
    ]:
        close(actual, expected, label)
    return {'n_clusters':len(units),'n_pools':len(seen),'universal_gain':float(gains.mean()),
            'calibrated_gain_over_random':float(over_random.mean()),
            'calibrated_gain_over_complex':float(calibrated_gains.mean())}


def verify_figure_data(root, temporal, candidate, calibration):
    figure = read(root/'figure_source_data.json')
    for actual, expected in [(figure['temporal_units'],temporal['inference_units']),
                             (figure['temporal_aggregate'],temporal['aggregate']),
                             (figure['ranking'],candidate['aggregate']),
                             (figure['calibration'],calibration['development'])]:
        if actual != expected:
            raise ValueError('Figure source differs from saved evidence')
    if figure['illustration']['record_id'] != min(r['id'] for r in temporal['results']):
        raise ValueError('Structural illustration was not selected by identifier')
    with (root/'primary_results.csv').open(encoding='utf-8',newline='') as handle:
        rows = list(csv.DictReader(handle))
    adaptation = read(root/'evidence/adaptation_results_portable.json')['aggregate']
    final = temporal['aggregate'];rank = candidate['aggregate'];cal = calibration['development']
    expected = [(46,adaptation['mean_ecls_advantage'],*adaptation['ecls_advantage_mean_ci95']),
                (15,final['mean_ecls_advantage'],*final['ecls_advantage_mean_ci95']),
                (7,rank['overall_mean_gain'],*rank['overall_gain_ci95']),
                (7,cal['mean_generator_aware_nnr']-.5,*cal['generator_aware_over_random_ci95']),
                (7,cal['mean_gain_over_complex_nll'],*cal['gain_over_complex_nll_ci95'])]
    close([[float(row[k]) for k in ('clusters','mean','ci95_low','ci95_high')] for row in rows], expected, 'endpoint table')


def verify(root):
    root = Path(root)
    checked = verify_manifest(root)
    evidence = root / "evidence"
    adaptation = verify_scores(read(evidence / "adaptation_results_portable.json"), 2281,
                               read(evidence / "adaptation_cluster_assignments.json"))
    temporal = verify_scores(read(evidence / "temporal_results_portable.json"), 2309)
    candidate = read(evidence / 'candidate_results.json')
    calibration = read(evidence / 'calibration_analysis.json')
    ranking = verify_ranking(candidate, calibration)
    verify_figure_data(root, read(evidence/'temporal_results_portable.json'), candidate, calibration)
    decision = read(evidence / "final_decision_original.json")
    if decision["rerun_permitted"] is not False or decision["attempt"] != 1 or not decision["terminal"]:
        raise ValueError("Final decision is not the terminal first attempt")
    ledger = read(root / "SOURCE_PROVENANCE.json")
    original = ledger["sources"]["results/publication/h3_ecls_temporal_final_v1/results.json"]
    if original["original_sha256"] != decision["source_result_sha256"]:
        raise ValueError("Original result provenance differs from frozen decision")
    return {"status": "valid", "files_checked": checked, "adaptation": adaptation,
            "temporal_final": temporal, "development_ranking": ranking, "model_forward_performed": False,
            "scope": "saved-score numerical reanalysis; not end-to-end inference"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))


if __name__ == "__main__":
    main()
