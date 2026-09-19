"""Freeze the existing Ridge screen, measure design-input latency, refresh new IDs.

No fitting to v6 labels and no new teacher runs. Cost replay is retrospective.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest

OUT = ROOT / 'results/pae_screening_v7'
OLD = ROOT / 'results/pae_surrogate_revision_v4'
DENSE = ROOT / 'results/pae_dense_v6'


def rank_ids(rows, scores, fraction=.2):
    """Stable candidate-only selection within each component."""
    selected = []
    for component in sorted({r['component_id'] for r in rows if r['entity_type'] == 'candidate'}):
        indices = [i for i, r in enumerate(rows) if r['component_id'] == component and r['entity_type'] == 'candidate']
        indices.sort(key=lambda i: (float(scores[i]), rows[i]['entity_id']))
        selected.extend(rows[i]['entity_id'] for i in indices[:int(np.ceil(fraction * len(indices)))] )
    return selected


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    initial_code_hash = digest(Path(__file__))
    if (OUT / 'protocol.json').exists():
        initial_code_hash = read(OUT / 'protocol.json')['script_sha256']
        if initial_code_hash != digest(Path(__file__)):
            amendment = read(OUT / 'implementation_amendment_01.json')
            assert amendment['original_sha256'] == initial_code_hash
            assert amendment['corrected_sha256'] == digest(Path(__file__))
    data = torch.load(OLD / 'features.pt', map_location='cpu', weights_only=False)
    train = np.array([r['split'] == 'train' for r in data['rows']])
    y = np.array([r['target'] for r in data['rows']], dtype=np.float32)
    model = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(data['simple'][train], y[train])
    params = {'mean': model[0].mean_, 'scale': model[0].scale_, 'coef': model[1].coef_, 'intercept': model[1].intercept_}
    path = OUT / 'ridge_fixed10.npz'
    if path.exists():
        with np.load(path) as saved:
            for key, value in params.items():
                np.testing.assert_array_equal(saved[key], value)
    else:
        np.savez(path, **params)
    freeze(OUT / 'protocol.json', {
        'classification': 'prospective_screen_preparation_and_retrospective_cost_replay',
        'model': 'v4 Ridge alpha10, original train split only; no v6 refitting',
        'model_sha256': digest(path), 'training_cache_sha256': digest(OLD / 'features.pt'),
        'development_exposure': 'v4, v5 and v6 performance already inspected; choice is development-informed',
        'selection': 'per component lowest predicted PAE 20 percent, ceil, entity_id tie break; controls excluded',
        'teacher': 'AF2-Multimer v3 models1/2 seeds7103/7111/7121 recycles3, six complete replicates',
        'future_primary': 'component-equal recall of true best20 percent at20 percent budget; within-generator secondary',
        'uncertainty': 'paired component bootstrap10000 seed20260918; compare random20 percent; candidate rows are not independent samples',
        'minimum_components': 6, 'minimum_antigen_families': 3,
        'feasibility_threshold_not_power_analysis': True,
        'isolation': {'vh': .9, 'vl': .9, 'paired_cdr': .7, 'h3': .5, 'antigen': .3, 'both_coverage': .8},
        'short_sequence_audit': 'MMseqs plus exhaustive ungapped H3/antigen sensitivity; pretraining overlap separately audited',
        'independent_gate': 'all structurally eligible new IDs, no score selection; stop before labels if isolation/family/power review incomplete',
        'timing': 'CPU design-PDB to features to ranks, model loading reported separately, one warmup and five repeats',
        'cost_boundary': 'recorded AF2 elapsed replay only; excludes worker warmup, generation, failures and queueing; not measured end-to-end speedup',
        'script_sha256': initial_code_hash})
    return data


def benchmark():
    from scripts.harden_pae_surrogate_v4 import chain_residues, locate_h3, build_region_batch, Fragment
    data = prepare()
    torch.set_num_threads(2)
    teacher = read(DENSE / 'teacher_labels.json')
    assert not teacher['incomplete_entities']
    entities = teacher['rows']
    checked = set()
    for row in entities:
        assert {(o['model'], o['seed']) for o in row['observations']} == {(m,s) for m in (1,2) for s in (7103,7111,7121)}
        assert len(row['observations']) == 6
        for o in row['observations']:
            if o['pae_path'] not in checked:
                assert digest(ROOT / o['pae_path']) == o['pae_sha256']
                checked.add(o['pae_path'])
    meta = {c['component_id']: c for c in read(ROOT / 'data/candidate_interface_external_calibration_v1/extension_manifest_v1.json')['components']}
    start = time.perf_counter()
    with np.load(OUT / 'ridge_fixed10.npz') as z:
        weights = {k: z[k].copy() for k in z.files}
    load_seconds = time.perf_counter() - start
    aa = 'ACDEFGHIKLMNPQRSTVWY'
    predictions = np.zeros(len(entities))
    timings = []
    for component in sorted(meta):
        ids = [i for i, r in enumerate(entities) if r['component_id'] == component]
        if not ids:
            continue
        m = meta[component]
        provenance = next(p for p in data['provenance'] if p['scaffold'] == component)
        pdb = ROOT / provenance['pdb']
        assert digest(pdb) == provenance['sha256']
        def score():
            residues = {c: chain_residues(pdb, c) for c in provenance['chains']}
            hc, lc, ac = provenance['chains']
            indices = locate_h3(residues[hc], m['cdr_h3_sequence'], m.get('h3_heavy_indices_zero_based'))
            region = hc + ':' + ','.join(str(residues[hc][i][0]) for i in indices)
            batch = build_region_batch(str(pdb), region, context_chains=[lc, ac], antigen_chains=[ac],
                                       antigen_context_cap=50, preserve_context_chain_order=True, device='cpu')
            cm = batch['generate_flag'][0].bool() & batch['mask'][0].bool()
            am = (batch['fragment_type'][0] == int(Fragment.Antigen)) & batch['mask'][0].bool()
            assert int(am.sum()) == len(m['antigen_sequence'])
            d = torch.cdist(batch['pos_heavyatom'][0][cm, 1], batch['pos_heavyatom'][0][am, 1]).min(1).values
            geom = [len(indices), int(am.sum()), float(d.mean()), float(d.std(unbiased=False)), float(d.min()), float((d < 8).float().mean())]
            composition = lambda seq: [seq.count(a) / len(seq) for a in aa]
            x = np.array([composition(entities[i]['h3_sequence']) + composition(m['antigen_sequence']) + geom for i in ids])
            p = ((x - weights['mean']) / weights['scale']) @ weights['coef'] + weights['intercept']
            ranked = rank_ids([entities[i] for i in ids], p)
            return x, p, ranked
        score()
        elapsed = []
        for _ in range(5):
            start = time.perf_counter()
            x, p, ranked = score()
            elapsed.append(time.perf_counter() - start)
        expected = data['simple'][next(i for i,r in enumerate(data['rows']) if r['scaffold'] == component)]
        np.testing.assert_allclose(x[:, 20:], np.broadcast_to(expected[20:], x[:, 20:].shape), atol=2e-5)
        predictions[ids] = p
        timings.append({'component': component, 'entities': len(ids), 'runs_seconds': elapsed, 'median_seconds': float(np.median(elapsed))})
    previous = read(DENSE / 'predictions.json')
    expected = dict(zip([r['id'] for r in previous['rows']], previous['models']['v4_ridge_fixed10']))
    np.testing.assert_allclose(predictions, [expected[r['entity_id']] for r in entities], atol=2e-5)
    selected = set(rank_ids(entities, predictions))
    assert selected == set(rank_ids(entities, [expected[r['entity_id']] for r in entities]))
    # Deduplicate actual AF2 work across generator arms, including selected overlaps.
    def slots(rows):
        result = {}
        for row in rows:
            for o in row['observations']:
                if not o.get('success', o.get('reused_historical', False)):
                    raise ValueError('Incomplete teacher cost')
                key = (row['component_id'], row['heavy_sequence'], row['light_sequence'], row['antigen_sequence'], o['model'], o['seed'])
                result[key] = o
        return result
    candidates = [r for r in entities if r['entity_type'] == 'candidate']
    full = slots(candidates)
    screened = slots([r for r in candidates if r['entity_id'] in selected])
    full_seconds = sum(o['elapsed'] for o in full.values())
    screen_seconds = sum(o['elapsed'] for o in screened.values())
    freeze(OUT / 'cost_replay.json', {
        'classification': 'retrospective_exposed_panel_cost_replay_not_end_to_end_speedup',
        'source_teacher_sha256': digest(DENSE / 'teacher_labels.json'),
        'candidate_count': len(candidates), 'selected_count': len(selected), 'selected_ids': sorted(selected),
        'full_unique_teacher_slots': len(full), 'selected_unique_teacher_slots': len(screened),
        'avoided_unique_slot_fraction': 1-len(screened)/len(full),
        'recorded_full_af2_seconds': full_seconds, 'recorded_selected_af2_seconds': screen_seconds,
        'recorded_elapsed_reduction_fraction': 1-screen_seconds/full_seconds,
        'model_load_seconds_excluding_imports': load_seconds, 'timing_scaffolds': timings,
        'prediction_equivalence_atol': 2e-5, 'selected_ids_exact_match_v6': True,
        'limitations': 'Same exposed6 scaffolds; mixed cached historical and current AF2 elapsed; no rerun, no hardware normalization, no generation/worker startup/retry costs. Timings include controls.'})
    print(json.dumps(read(OUT / 'cost_replay.json'), indent=2))


def discover():
    from scripts.build.acquire_rcsb_candidate_interface_extension import acquire
    from scripts.build.discover_rcsb_candidate_interface_extension import discover as discover_metadata
    prepare()
    out = ROOT / 'data/pae_independent_v7'
    query = read(ROOT / 'data/pae_external_refresh_v6/query.json')
    for node in query['query']['nodes']:
        if node.get('parameters', {}).get('attribute') == 'rcsb_accession_info.initial_release_date':
            node['parameters']['value']['to'] = '2026-09-18'
    freeze(out / 'query.json', query)
    ref = read(ROOT / 'data/ecls_independent_feasibility_v1/reference_union.json')
    exposed = set(ref['exact_exposed_pdb_ids'])
    prior_paths = ['data/ecls_independent_feasibility_v1/snapshot/entries.json', 'data/pae_external_refresh_v6/snapshot/entries.json']
    for path in prior_paths:
        exposed.update(r['rcsb_id'].lower() for r in read(ROOT / path)['entries'])
    ref['exact_exposed_pdb_ids'] = sorted(exposed)
    freeze(out / 'reference_union.json', ref)
    freeze(out / 'protocol.json', {'classification': 'metadata_only_feasibility',
        'discovery_contract': {'query_config_sha256': digest(out / 'query.json')},
        'screen_protocol_sha256': digest(OUT / 'protocol.json'),
        'reference_sha256': digest(out / 'reference_union.json'),
        'prior_snapshots': {p: digest(ROOT / p) for p in prior_paths},
        'selection': 'all candidate IDs absent prior metadata; unchanged structural and five-axis criteria; no labels'})
    if not (out / 'snapshot/acquisition.json').exists():
        acquire(out / 'query.json', out / 'protocol.json', out / 'snapshot')
    if not (out / 'discovery.json').exists():
        discover_metadata(out / 'snapshot', out / 'discovery.json', include_viral=True)
    discovery = read(out / 'discovery.json')
    filtered = copy.deepcopy(discovery)
    for e in filtered['entries']:
        if e['status'] == 'candidate' and e['pdb_id'].lower() in exposed:
            e['status'] = 'excluded_prior_metadata_exposure'
    freeze(out / 'discovery_unexposed.json', filtered)
    ids = [e['pdb_id'] for e in filtered['entries'] if e['status'] == 'candidate']
    freeze(out / 'feasibility.json', {'status': 'requires_structure_and_isolation' if ids else 'no_unexposed_metadata_candidates',
        'discovery_counts': discovery['counts'], 'unexposed_candidate_ids': ids,
        'independent_components_established': 0, 'teacher_scoring_started': False,
        'scope_limitation': 'Original paired-HL peptide5-50aa query, incremental through2026-09-18; not global absence of independent data'})
    print(json.dumps(read(out / 'feasibility.json'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'benchmark', 'discover'])
    args = parser.parse_args()
    {'prepare': prepare, 'benchmark': benchmark, 'discover': discover}[args.stage]()
