"""Analog prescreening evaluation: teacher labels, recipe retrain, screening.

Mirrors the frozen v4/v7 recipe exactly: teacher = mean normalized directional
candidate-region-to-peptide PAE over six AF2-Multimer v3 matrices (/31);
Ridge(alpha=10) on StandardScaler with composition + length + design-backbone
CB-distance features; component-equal recall of the teacher top-20% at the 20%
budget under uniform random tie breaking; paired component bootstrap. Control
entities are reported descriptively. No binding, speedup or superiority claim.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest

OUT = ROOT / 'results/pae_analog_evaluation_v30'
V28 = ROOT / 'results/pae_analog_generation_v28'
V29 = ROOT / 'results/pae_analog_teacher_v29'
AA = 'ACDEFGHIKLMNPQRSTVWY'
BOOTSTRAP_SEED = 20260918
BOOTSTRAP_REPS = 10000


def directional_target(pae, start, region_length, design_length, partner_length, peptide_length):
    """v4 teacher path with mapped names; /31 normalization is part of the frozen estimand.

    The region start index comes from the frozen v26 candidate_region indices
    (design chain sequence preserves the same index layout), never from a
    sequence search.
    """
    agstart = design_length + partner_length
    if pae.shape != (agstart + peptide_length,) * 2:
        raise ValueError(f"Teacher array shape mismatch: {pae.shape}")
    return float(pae[start:start + region_length, agstart:].mean() / 31)


def cb_residues(pdb_path):
    from Bio.PDB import PDBParser
    structure = PDBParser(QUIET=True).get_structure('x', str(pdb_path))
    model = next(structure.get_models())
    result = {}
    for chain in model:
        result[chain.id] = np.asarray([
            (r['CB'] if 'CB' in r else r['CA']).coord for r in chain.get_residues()
            if r.id[0] == ' ' and 'CA' in r], dtype=float)
    return result


def geometry(design_pdb, region_indices, peptide_len, design_chain, peptide_chain):
    coords = cb_residues(design_pdb)
    region = coords[design_chain][region_indices]
    peptide = coords[peptide_chain]
    d = np.linalg.norm(region[:, None, :] - peptide[None, :, :], axis=-1).min(1)
    return [len(region_indices), peptide_len, float(d.mean()), float(d.std()),
            float(d.min()), float((d < 8).mean())]


def composition(sequence):
    return [sequence.count(aa) / len(sequence) for aa in AA]


def screening(pred, target, fraction, top_fraction=0.2):
    """v4 recall under uniform random tie breaking; component-level."""
    n = len(target)
    k, m = max(1, math.ceil(n * fraction)), max(1, math.ceil(n * top_fraction))

    def inclusion(values, slots):
        values = np.asarray(values)
        cutoff = np.sort(values)[slots - 1]
        below, equal = values < cutoff, values == cutoff
        return below.astype(float) + equal * ((slots - below.sum()) / equal.sum())

    selected = inclusion(pred, k)
    optimal = inclusion(target, m)
    return float((selected * optimal).sum() / optimal.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-teacher-check', action='store_true',
                        help='dev use only; require verified v29 result otherwise')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    v29 = read(V29 / 'result.json')
    if not args.skip_teacher_check and not (v29.get('verified') and v29['completed'] == v29['total']):
        raise RuntimeError('v29 teacher run is not verified complete; refusing to evaluate')

    protocol = {
        'classification': 'analog_transfer_screening_evaluation_not_binding_validation',
        'v28_generation_sha256': digest(V28 / 'generation.json'),
        'v29_result_sha256': digest(V29 / 'result.json'),
        'estimand': 'mean normalized directional candidate-region-to-peptide PAE /31 over six replicates',
        'features': 'region composition 20 + peptide composition 20 + geometry 6 (CB distances, CA fallback for GLY)',
        'model': 'StandardScaler + Ridge(alpha=10) fit on train components only',
        'screening': 'recall of teacher top-20% at 20% budget (3/14), random tie breaking, component-equal',
        'bootstrap': {'reps': BOOTSTRAP_REPS, 'seed': BOOTSTRAP_SEED, 'unit': 'component, paired'},
        'sign_rule': 'MPNN NLL sign fixed on train side only',
        'boundary': 'analog transfer; no binding, affinity, speedup or architecture-superiority claim',
        'script_sha256': digest(Path(__file__)),
    }
    freeze(OUT / 'protocol.json', protocol)

    generation = read(V28 / 'generation.json')['components']
    by_instance = {c['instance']: c for c in generation}

    # Teacher labels per entity: scan verified attempts on disk (resumed runs
    # skip earlier batches without appending to result.json's attempt list).
    rows = []
    job_index = {}
    verified_attempts = {}
    for status_path in sorted(V29.glob('model*_batch_*/attempt*/status.json')):
        if read(status_path).get('verified'):
            folder = status_path.parent
            verified_attempts[folder.parent.name] = folder
    for batch_name, folder in sorted(verified_attempts.items()):
        for i, line in enumerate((folder / 'jobs.jsonl').read_text().splitlines()):
            if not line.strip():
                continue
            job = json.loads(line)
            npz = folder / 'pae' / f'{i:02d}.npz'
            if npz.exists():
                job_index[job['id']] = npz
    if len(job_index) != v29['total']:
        raise ValueError(f'PAE index has {len(job_index)} of {v29["total"]} jobs')

    # Frozen v26 region indices per component; the geometry step reuses them.
    v26 = {r['instance']: r for r in read(ROOT / 'results/pae_analog_structures_v26/manifest.json')['records']
           if r['passes']}
    for instance, component in sorted(by_instance.items()):
        record = v26[instance]
        start = int(record['candidate_region']['indices_zero_based'][0])
        for entity in component['entities']:
            grid = []
            for job_id, npz in sorted(job_index.items()):
                if not job_id.startswith(entity['entity_id'] + '|'):
                    continue
                model = int(job_id.split('|model')[1].split('|')[0])
                seed = int(job_id.rsplit('seed', 1)[1])
                with np.load(npz) as z:
                    pae = z['pae']
                partner = component['partner_sequence'] or ''
                region = entity['region_sequence']
                if entity['design_chain_sequence'][start:start + len(region)] != region:
                    raise ValueError(f"{entity['entity_id']}: region mismatch at frozen indices")
                value = directional_target(pae, start, len(region),
                                           len(entity['design_chain_sequence']), len(partner),
                                           len(component['peptide']))
                grid.append({'model': model, 'seed': seed, 'value': value, 'npz': npz.as_posix()})
            if len(grid) != 6:
                raise ValueError(f"{entity['entity_id']}: {len(grid)} of 6 replicates")
            values = np.asarray([g['value'] for g in grid])
            sd = float(np.std(values, ddof=1))
            rows.append({'entity_id': entity['entity_id'], 'instance': instance,
                         'cohort': component['cohort'], 'side': component['side'],
                         'arm': entity['arm'], 'region_sequence': entity['region_sequence'],
                         'peptide': component['peptide'],
                         'target': float(values.mean()), 'replicate_sd': sd,
                         'weight': float(np.clip(1 / (1 + (sd / 0.1) ** 2), 0.1, 1)),
                         'generator_score': entity['generator_score']})
        print(f'{instance}: labeled', flush=True)

    # Geometry from frozen design PDBs; constant within a component by construction.
    features = {}
    for instance, component in sorted(by_instance.items()):
        record = v26[instance]
        design_pdb = ROOT / component['design_pdb']
        design_chain = 'B' if component['cohort'] == 'tcr' else 'V'
        geom = geometry(design_pdb, record['candidate_region']['indices_zero_based'],
                        len(component['peptide']), design_chain, 'P')
        features[instance] = geom
    freeze(OUT / 'geometry.json', {'instances': features,
                                   'boundary': 'constant within component by construction'})

    # Assemble design matrices.
    train_rows = [r for r in rows if r['side'] == 'train' and r['arm'] == 'candidate']
    eval_rows = [r for r in rows if r['side'] == 'eval' and r['arm'] == 'candidate']
    control_rows = [r for r in rows if r['side'] == 'eval' and r['arm'] != 'candidate']

    def design_vector(row):
        component = by_instance[row['instance']]
        geom = features[row['instance']]
        return composition(row['region_sequence']) + composition(row['peptide']) + geom

    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x_train = np.asarray([design_vector(r) for r in train_rows])
    y_train = np.asarray([r['target'] for r in train_rows])
    x_eval = np.asarray([design_vector(r) for r in eval_rows])
    model = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(x_train, y_train)
    eval_pred = model.predict(x_eval)
    comp_only_model = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(x_train[:, :40], y_train)
    comp_only_pred = comp_only_model.predict(x_eval[:, :40])

    # MPNN NLL sign fixed on the train side.
    train_nll = np.asarray([r['generator_score'] for r in train_rows])
    nll_sign = 1.0 if np.corrcoef(train_nll, y_train)[0, 1] < 0 else -1.0
    nll_pred = -nll_sign * np.asarray([r['generator_score'] for r in eval_rows])

    # Component-equal screening.
    instances = sorted({r['instance'] for r in eval_rows})
    per_component = {}
    for instance in instances:
        members = [r for r in eval_rows if r['instance'] == instance]
        index = [eval_rows.index(r) for r in members]
        target = [r['target'] for r in members]
        entry = {'n': len(members),
                 'ridge': screening(eval_pred[index], target, 0.2),
                 'nll': screening(nll_pred[index], target, 0.2),
                 'random': screening(np.zeros(len(members)), target, 0.2),
                 'train_mean': screening(np.full(len(members), float(y_train.mean())), target, 0.2)}
        entry['composition_only'] = screening(comp_only_pred[index], target, 0.2)
        per_component[instance] = entry

    def component_equal(key):
        values = [entry[key] for entry in per_component.values()]
        return {'mean': float(np.mean(values)), 'values': values}

    report = {'per_component': per_component,
              'summary': {key: component_equal(key) for key in
                          ('ridge', 'nll', 'random', 'train_mean', 'composition_only')}}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ridge_values = np.asarray([e['ridge'] for e in per_component.values()])
    random_values = np.asarray([e['random'] for e in per_component.values()])
    differences = ridge_values - random_values
    means = rng.choice(differences, (BOOTSTRAP_REPS, len(differences)), replace=True).mean(1)
    report['ridge_vs_random'] = {'mean_difference': float(differences.mean()),
                                 'percentile_95_ci': np.quantile(means, [0.025, 0.975]).tolist(),
                                 'units': len(differences), 'unit': 'component; descriptive'}
    positives = int(sum(1 for e in per_component.values() if e['ridge'] > e['random']))
    report['ridge_positive_components'] = {'count': positives, 'of': len(per_component)}
    report['controls'] = [
        {'entity_id': r['entity_id'], 'arm': r['arm'], 'target': r['target'],
         'instance': r['instance']} for r in control_rows]
    report['counts'] = {'train_rows': len(train_rows), 'eval_rows': len(eval_rows),
                        'control_rows': len(control_rows)}

    freeze(OUT / 'evaluation.json', report)
    freeze(OUT / 'predictions.json', {
        'train': [{'entity_id': r['entity_id'], 'target': r['target']} for r in train_rows],
        'eval': [{'entity_id': r['entity_id'], 'target': r['target'],
                  'ridge': float(p)} for r, p in zip(eval_rows, eval_pred)]})
    freeze(OUT / 'summary.json', {'ridge_recall': report['summary']['ridge'],
                                  'random_recall': report['summary']['random'],
                                  'ridge_vs_random': report['ridge_vs_random'],
                                  'positive_components': report['ridge_positive_components'],
                                  'nll': report['summary']['nll'],
                                  'composition_only': report['summary']['composition_only']})
    freeze(OUT / 'completion.json', {'status': 'completed',
                                     'evaluation_sha256': digest(OUT / 'evaluation.json')})
    print(json.dumps(report['summary'], indent=1), flush=True)


if __name__ == '__main__':
    main()
