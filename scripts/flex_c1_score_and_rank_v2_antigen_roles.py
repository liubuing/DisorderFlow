"""FLEX C1 ranker re-evaluation with corrected chain-role encoding (v2).

Re-runs the judge side of Phase C1 (ensemble-marginalized ECLS) with the
intended Fragment encoding (H -> Heavy(1), P -> Antigen(3)) and re-uses the
frozen, role-independent baselines (ProteinMPNN conditional NLL, 20-dim
composition logistic).  Frozen C1 artifacts are untouched; everything lands in
results/flex_c1_v2_antigen_roles/.

Also fixes the reporting bug of the original script: its ``top1_rate`` /
``top5_rate`` counted ``percentile == 0`` (never last) instead of the native
being ranked first / top-5 under the 1 = best convention.

Usage:
  python scripts/flex_c1_score_and_rank_v2_antigen_roles.py --instances 2x6m_A_B --out results/flex_c1_v2_smoke
  python scripts/flex_c1_score_and_rank_v2_antigen_roles.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_pae_dense_v6 import read, freeze  # noqa: E402
from scripts.flex_j0_score_v1 import SCORING_SEEDS, candidate_nll  # noqa: E402

C1 = ROOT / 'results/flex_c1_v1'
J0 = ROOT / 'results/flex_j0_v1'
J0B = ROOT / 'results/flex_j0b_v1'
DEFAULT_OUT = ROOT / 'results/flex_c1_v2_antigen_roles'
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261029


def judge_scores_for_component(model, entry, designs, device, out_dir):
    cache = out_dir / 'scoring' / f"{entry['instance']}.json"
    if cache.exists():
        return read(cache)

    from modules.bfn_loader import build_region_batch

    audit = read(J0B / 'ensembles' / entry['instance'] / 'ensemble_audit.json')
    accepted_seeds = [c['seed'] for c in audit['conformers']
                      if c['status'] == 'accepted']
    conformer_files = [
        J0B / 'conformer_component_pdbs' /
        f"{entry['instance']}_{entry['instance']}_seed_{seed}.pdb"
        for seed in accepted_seeds]
    missing = [str(p) for p in conformer_files if not p.exists()]
    if missing:
        raise RuntimeError(f"{entry['instance']}: missing conformers {missing}")

    native = entry['region_sequence']
    pool = [native] + list(designs) + list(entry['shuffles'])

    stripped_batch = build_region_batch(
        str(J0 / 'pdb' / f"{entry['instance']}.pdb"), entry['region_spec'],
        context_chains=[], antigen_chains=[], device=device)
    stripped = {}
    for candidate in pool:
        stripped[candidate] = float(np.mean([
            candidate_nll(model, stripped_batch, candidate, seed, device)[0]
            for seed in SCORING_SEEDS]))

    complex_per_conformer = []
    for conformer_file in conformer_files:
        batch = build_region_batch(
            str(conformer_file), entry['region_spec'],
            context_chains=['P'], antigen_chains=['P'], device=device)
        scores = {}
        for candidate in pool:
            scores[candidate] = float(np.mean([
                candidate_nll(model, batch, candidate, seed, device)[0]
                for seed in SCORING_SEEDS]))
        complex_per_conformer.append(scores)
        print(f"  {entry['instance']} conformer {len(complex_per_conformer)}/"
              f"{len(conformer_files)} scored {len(pool)}", flush=True)

    judge = {}
    for candidate in pool:
        complex_marginal = float(np.mean(
            [d[candidate] for d in complex_per_conformer]))
        judge[candidate] = complex_marginal - stripped[candidate]
    result = {
        'instance': entry['instance'],
        'designs': list(designs),
        'judge_scores': judge,
        'n_conformers': len(conformer_files),
        'accepted_seeds': accepted_seeds,
        'role_encoding': 'H=Heavy(1), P=Antigen(3)',
    }
    freeze(cache, result)
    return result


def percentile_of_native(native, scores, better='lower'):
    others = [s for cand, s in scores.items() if cand != native]
    native_score = scores[native]
    if better == 'lower':
        worse = sum(1 for s in others if s > native_score)
    else:
        worse = sum(1 for s in others if s < native_score)
    tied = sum(1 for s in others if s == native_score)
    return (worse + 0.5 * tied) / len(others)


def rank_of_native(native, scores, better='lower'):
    native_score = scores[native]
    if better == 'lower':
        better_count = sum(1 for c, s in scores.items()
                           if c != native and s < native_score)
    else:
        better_count = sum(1 for c, s in scores.items()
                           if c != native and s > native_score)
    return 1 + better_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instances', default='')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prereg = read(C1 / 'preregistration.json')
    entries = list(prereg['components'])
    if not args.all:
        wanted = [x for x in args.instances.split(',') if x]
        if not wanted:
            raise SystemExit('pass --instances a,b or --all')
        entries = [e for e in entries if e['instance'] in wanted]
        missing = set(wanted) - {e['instance'] for e in entries}
        if missing:
            raise SystemExit(f'unknown instances: {sorted(missing)}')

    mpnn_cache = read(C1 / 'mpnn_nll_cache.json')
    composition_cache = read(C1 / 'composition_prob_cache.json')

    import torch
    from modules.bfn_loader import load_bfn
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _cfg = load_bfn(device)

    rows = []
    for entry in entries:
        generation = read(C1 / 'generation' / entry['instance'] / 'candidates.json')
        designs = generation['designs']
        scored = judge_scores_for_component(model, entry, designs, device, out_dir)
        judge = scored['judge_scores']
        native = entry['region_sequence']
        mpnn = mpnn_cache[entry['instance']]
        comp = composition_cache[entry['instance']]
        rows.append({
            'instance': entry['instance'],
            'family': entry['family'],
            'native': native,
            'pool_size': len(judge),
            'judge': {
                'percentile': percentile_of_native(native, judge),
                'rank': rank_of_native(native, judge),
            },
            'proteinmpnn_nll': {
                'percentile': percentile_of_native(native, mpnn),
                'rank': rank_of_native(native, mpnn),
            },
            'composition_logistic': {
                'percentile': percentile_of_native(native, comp, better='higher'),
                'rank': rank_of_native(native, comp, better='higher'),
            },
        })
        print(entry['instance'], 'judge pct',
              round(rows[-1]['judge']['percentile'], 3),
              'rank', rows[-1]['judge']['rank'], flush=True)

    # ---- pooled statistics (family-clustered bootstrap) ----
    def pooled(values_by_component):
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        by_family = {}
        for item in values_by_component:
            by_family.setdefault(item['family'], []).append(item['percentile'])
        family_list = [by_family[k] for k in sorted(by_family)]
        means = []
        for _ in range(BOOTSTRAP_DRAWS):
            sampled = [family[rng.integers(0, len(family))] for family in family_list]
            means.append(float(np.mean(sampled)))
        point = float(np.mean([x['percentile'] for x in values_by_component]))
        return point, [float(np.quantile(means, 0.025)),
                       float(np.quantile(means, 0.975))]

    stats = {}
    for method in ('judge', 'proteinmpnn_nll', 'composition_logistic'):
        values = [{'family': r['family'], 'percentile': r[method]['percentile']}
                  for r in rows]
        point, ci = pooled(values)
        ranks = [r[method]['rank'] for r in rows]
        stats[method] = {
            'mean_percentile': point,
            'bootstrap_ci_family_bootstrap': ci,
            'mean_percentile_component_equal': point,
            'n_families': len({r['family'] for r in rows}),
            'top1_rate': float(np.mean([1.0 if r == 1 else 0.0 for r in ranks])),
            'top5_rate': float(np.mean([1.0 if r <= 5 else 0.0 for r in ranks])),
            'median_rank': float(np.median(ranks)),
            'min_rank': int(min(ranks)),
        }

    # ---- gate (same preregistered rule, corrected sign conventions) ----
    judge_by_instance = {r['instance']: r['judge']['percentile'] for r in rows}
    baseline_stats = {}
    for baseline in ('composition_logistic', 'proteinmpnn_nll'):
        base = {r['instance']: r[baseline]['percentile'] for r in rows}
        paired = {r['instance']: judge_by_instance[r['instance']] - base[r['instance']]
                  for r in rows}
        by_family = {}
        for r in rows:
            by_family.setdefault(r['family'], []).append(paired[r['instance']])
        family_means = {k: float(np.mean(v)) for k, v in by_family.items()}
        baseline_stats[baseline] = {
            'family_mean_paired_diff': family_means,
            'families_judge_better': sum(1 for v in family_means.values() if v > 0),
            'n_families': len(family_means),
        }

    gate_parts = {
        'beats_composition_70pct': (baseline_stats['composition_logistic']
                                    ['families_judge_better']
                                    / baseline_stats['composition_logistic']['n_families']) >= 0.70,
        'beats_proteinmpnn_70pct': (baseline_stats['proteinmpnn_nll']
                                    ['families_judge_better']
                                    / baseline_stats['proteinmpnn_nll']['n_families']) >= 0.70,
        'ci_beats_random': stats['judge']['bootstrap_ci_family_bootstrap'][0] > 0.5,
    }
    verdict = 'pass' if all(gate_parts.values()) else 'fail'
    summary = {
        'gate': 'c1_v2_corrected_roles',
        'n_components': len(rows),
        'n_families': len({r['family'] for r in rows}),
        'stats': stats,
        'baseline_comparisons': baseline_stats,
        'gate_parts': gate_parts,
        'verdict': verdict,
        'role_encoding': 'H=Heavy(1), P=Antigen(3)',
        'baselines_reused': ['mpnn_nll_cache.json', 'composition_prob_cache.json'],
        'classification': 'development_gate_not_confirmatory_claim',
        'provenance': 're-scores frozen results/flex_c1_v1/preregistration.json; '
                      'frozen C1 artifacts untouched',
    }
    freeze(out_dir / 'c1_v2_summary.json', summary)
    freeze(out_dir / 'c1_v2_results.json', {'components': rows})
    print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
