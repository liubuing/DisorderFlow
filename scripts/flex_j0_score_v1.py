"""FLEX-BINDER Gate J0 scoring run.

Executes the frozen preregistration (results/flex_j0_v1/preregistration.json):
for each cohort component, score the native CDR-H3 sequence and its 50
composition-matched shuffles in two contexts (complex: peptide present;
stripped: heavy chain only) with the production BFN checkpoint, and compute
the ECLS contrast.

NLL field disclosure: the run prefers ``losses['factual_nll_per_sample']``;
when the checkpoint carries no antigen-mismatch loss weight that field is
absent, and the single-sample ``losses['seq']`` is used instead - the same
masked cross-entropy over the generate region of the injected sequence. Which
field was used is recorded in the frozen output.

Usage: python scripts/flex_j0_score_v1.py [--smoke]
"""
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest

OUT = ROOT / 'results/flex_j0_v1'
SCORING_SEEDS = [20261029, 20261030, 20261031]
FIXED_T = 0.5
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261029


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_nll(model, batch, candidate, seed, device):
    from disorderflow.utils.train import recursive_to
    from disorderflow.utils.misc import seed_all
    from modules.bfn_loader import inject_candidate_sequence

    seed_all(seed)
    run_batch = recursive_to(copy.deepcopy(batch), device)
    run_batch['fixed_t'] = FIXED_T
    run_batch['return_per_sample_metrics'] = True
    inject_candidate_sequence(run_batch, candidate)
    with torch.inference_mode():
        losses = model(run_batch)
    if 'factual_nll_per_sample' in losses:
        return float(losses['factual_nll_per_sample'][0]), 'factual_nll_per_sample'
    return float(losses['seq']), 'seq'


def score_component(model, entry, device, seeds, smoke=False):
    from modules.bfn_loader import build_region_batch
    pdb_path = str(ROOT / entry['component_pdb'])
    candidates = [entry['region_sequence']] + entry['shuffles']
    if smoke:
        candidates = candidates[:4]
    batches = {
        'complex': build_region_batch(pdb_path, entry['region_spec'],
                                      context_chains=['P'], device=device),
        'stripped': build_region_batch(pdb_path, entry['region_spec'],
                                       context_chains=[], device=device),
    }
    per_candidate = {context: {} for context in batches}
    field_used = None
    determinism_collapsed = False
    native = entry['region_sequence']
    for context, batch in batches.items():
        native_repeats = [candidate_nll(model, batch, native, seed, device)
                          for seed in seeds]
        field_used = native_repeats[0][1]
        determinism_collapsed = (len({n for n, _ in native_repeats}) == 1
                                 and len(seeds) > 1)
        for candidate in candidates:
            if determinism_collapsed and candidate == native:
                per_candidate[context][candidate] = float(
                    np.mean([n for n, _ in native_repeats]))
                continue
            if determinism_collapsed:
                nll, field = candidate_nll(model, batch, candidate, seeds[0], device)
                repeats = [nll]
            else:
                repeats = [candidate_nll(model, batch, candidate, seed, device)[0]
                           for seed in seeds]
            per_candidate[context][candidate] = float(np.mean(repeats))
    return {
        'instance': entry['instance'],
        'nll_field': field_used,
        'seeds_used': [seeds[0]] if determinism_collapsed else seeds,
        'determinism_collapsed': determinism_collapsed,
        'nll': per_candidate,
    }


def main():
    smoke = '--smoke' in sys.argv
    cohort = read(OUT / 'cohort.json')
    prereg_sha = read(OUT / 'preregistration_sha256.txt') if (OUT / 'preregistration_sha256.txt').exists() else None
    if cohort['preregistration_sha256'] != prereg_sha and prereg_sha is not None:
        raise RuntimeError('preregistration sha changed after cohort build')
    entries = cohort['components'][:1] if smoke else cohort['components']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from modules.bfn_loader import load_bfn, _get_checkpoint_path
    model, config = load_bfn(device)
    checkpoint_path = _get_checkpoint_path()

    out_dir = OUT / 'scoring'
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for entry in entries:
        cache_path = out_dir / f"{entry['instance']}.json"
        if cache_path.exists():
            row = read(cache_path)
        else:
            row = score_component(model, entry, device, SCORING_SEEDS, smoke=smoke)
            freeze(cache_path, row)
        all_rows.append(row)
        print(row['instance'], row['nll_field'], 'collapsed' if row['determinism_collapsed'] else '',
              flush=True)
        if smoke:
            break

    if smoke:
        print('smoke run complete; no gate computation', flush=True)
        return

    # ---- Endpoint computation on the frozen per-component NLLs. ----
    prereg = read(OUT / 'preregistration.json')
    family_of = {}
    for family_key, members in prereg['cohort']['antigen_families'].items():
        for peptide in members:
            family_of[peptide] = family_key

    advantages = []
    for row, entry in zip(all_rows, cohort['components']):
        native = entry['region_sequence']
        ecls_native = row['nll']['complex'][native] - row['nll']['stripped'][native]
        shuffle_ecls = [row['nll']['complex'][s] - row['nll']['stripped'][s]
                        for s in entry['shuffles']]
        advantage = float(np.mean(shuffle_ecls)) - ecls_native
        advantages.append({'instance': entry['instance'],
                           'pdb': entry['pdb_id'],
                           'peptide': entry['peptide_sequence'],
                           'family': family_of[entry['peptide_sequence']],
                           'ecls_native': ecls_native,
                           'advantage': advantage})

    family_means = {}
    for row in advantages:
        family_means.setdefault(row['family'], []).append(row['advantage'])
    family_values = {k: float(np.mean(v)) for k, v in family_means.items()}
    positive = sum(1 for v in family_values.values() if v > 0)
    positive_fraction = positive / len(family_values)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    keys = sorted(family_values)
    values = np.array([family_values[k] for k in keys])
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[indices].mean(axis=1)
    ci = [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]

    gate_pass = positive_fraction >= 0.70 and (ci[0] > 0)
    summary = {
        'gate': 'j0a_single_pose',
        'n_components': len(advantages),
        'n_families': len(family_values),
        'positive_families': positive,
        'positive_fraction': positive_fraction,
        'family_positive_fraction_gate': 0.70,
        'mean_advantage': float(values.mean()),
        'bootstrap_ci_family_mean': ci,
        'per_family': {k: {'mean_advantage': family_values[k],
                           'n_components': len(family_means[k])} for k in keys},
        'verdict': 'pass' if gate_pass else 'fail',
        'nll_field': all_rows[0]['nll_field'],
        'any_determinism_collapsed': any(r['determinism_collapsed'] for r in all_rows),
        'checkpoint_path': checkpoint_path,
        'checkpoint_sha256': sha256(checkpoint_path),
        'classification': 'development_gate_not_confirmatory_claim',
    }
    freeze(OUT / 'j0a_results.json', {'components': advantages})
    freeze(OUT / 'j0a_summary.json', summary)
    freeze(OUT / 'completion.json', {
        'status': 'completed', 'phase_c_unlocked': False,
        'summary_sha256': digest(OUT / 'j0a_summary.json')})
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
