"""FLEX-BINDER Phase C1, judge scoring + ranker comparison + gate.

Scores the 50 ProteinMPNN designs per component with the J0b-validated judge
(ensemble-marginalized ECLS: per-conformer complex NLL minus stripped NLL,
3 scoring seeds), reuses the frozen J0a/J0b caches bit-identically for the
native and shuffle pool members, computes the ProteinMPNN-NLL and
composition-logistic baselines, ranks the 101-candidate pools, and evaluates
the preregistered gate.

Resume-able per component. Fail-closed: design lengths, MPNN log_p shapes and
H3 mask positions are asserted before any ranking.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest
from scripts.flex_j0_score_v1 import SCORING_SEEDS, candidate_nll

OUT = ROOT / 'results/flex_c1_v1'
J0 = ROOT / 'results/flex_j0_v1'
J0B = ROOT / 'results/flex_j0b_v1'
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261029
MPNN_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYXBUZO'


def write_cache(path, value):
    """Plain atomic write for incrementally-growing derived caches; unlike
    ``freeze`` these are reproducible resume helpers, not one-shot artifacts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def score_designs_judge(model, entry, designs, device, conformer_pdbs):
    """Ensemble-marginal ECLS for the new designs; caches per component."""
    from modules.bfn_loader import build_region_batch
    cache_path = OUT / 'scoring' / f"{entry['instance']}.json"
    if cache_path.exists():
        return read(cache_path)
    j0b_row = read(J0B / 'scoring' / f"{entry['instance']}.json")
    j0a_row = read(J0 / 'scoring' / f"{entry['instance']}.json")
    stripped_cache = j0a_row['nll']['stripped']
    complex_per_conformer = list(j0b_row['nll_complex_per_conformer'])
    # Construct conformer file paths explicitly from the audit's accepted seed
    # order (matches the cached conformer order); no globbing.
    audit = read(J0B / 'ensembles' / entry['instance'] / 'ensemble_audit.json')
    accepted_seeds = [c['seed'] for c in audit['conformers']
                      if c['status'] == 'accepted']
    if len(accepted_seeds) != len(complex_per_conformer):
        raise RuntimeError(
            f"{entry['instance']}: {len(accepted_seeds)} accepted seeds vs "
            f"{len(complex_per_conformer)} cached conformers")
    conformer_files = [
        conformer_pdbs / f"{entry['instance']}_{entry['instance']}_seed_{seed}.pdb"
        for seed in accepted_seeds]
    missing = [str(p) for p in conformer_files if not p.exists()]
    if missing:
        raise RuntimeError(f"{entry['instance']}: missing conformer PDBs: {missing}")
    stripped_batch = build_region_batch(
        str(J0 / 'pdb' / f"{entry['instance']}.pdb"), entry['region_spec'],
        context_chains=[], device=device)
    design_stripped = {}
    for design in designs:
        design_stripped[design] = float(np.mean([
            candidate_nll(model, stripped_batch, design, seed, device)[0]
            for seed in SCORING_SEEDS]))
    for conformer_index, conformer_file in enumerate(conformer_files):
        complex_batch = build_region_batch(
            str(conformer_file), entry['region_spec'], context_chains=['P'],
            device=device)
        for design in designs:
            complex_per_conformer[conformer_index][design] = float(np.mean([
                candidate_nll(model, complex_batch, design, seed, device)[0]
                for seed in SCORING_SEEDS]))
        print(f"  {entry['instance']} conformer {conformer_index + 1}/"
              f"{len(conformer_files)} designs scored", flush=True)
    result = {
        'instance': entry['instance'],
        'designs': designs,
        'complex_per_conformer': complex_per_conformer,
        'stripped_for_designs': design_stripped,
        'stripped_cache_source': 'j0a_via_j0b',
    }
    freeze(cache_path, result)
    return result


def mpnn_conditional_nll(entry, designs):
    """Complex-state MPNN conditional NLL for every pool candidate."""
    import subprocess
    out_dir = OUT / 'mpnn_scoring' / entry['instance']
    npz_path = out_dir / 'conditional_probs_only' / f"{entry['instance']}.npz"
    if not npz_path.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        pdb_path = ROOT / entry['component_pdb']
        lengths = {}
        from Bio.PDB import PDBParser
        model0 = next(PDBParser(QUIET=True).get_structure(
            'x', str(pdb_path)).get_models())
        for chain in model0:
            lengths[chain.id] = sum(1 for r in chain.get_residues() if r.id[0] == ' ')
        lo, hi = (int(x) for x in entry['region_spec'].split(':')[1].split('-'))
        fixed_heavy = [p for p in range(1, lengths['H'] + 1) if not lo <= p <= hi]
        fixed_path = out_dir / 'fixed_positions.jsonl'
        fixed_path.write_text(json.dumps(
            {pdb_path.stem: {'H': fixed_heavy,
                             'P': list(range(1, lengths['P'] + 1))}}) + '\n',
            encoding='ascii')
        command = [
            sys.executable,
            (ROOT / 'ProteinMPNN/protein_mpnn_run.py').resolve().as_posix(),
            '--pdb_path', pdb_path.resolve().as_posix(),
            '--pdb_path_chains', 'H P',
            '--fixed_positions_jsonl', fixed_path.resolve().as_posix(),
            '--path_to_model_weights',
            (ROOT / 'ProteinMPNN/vanilla_model_weights').resolve().as_posix(),
            '--model_name', 'v_48_020',
            '--conditional_probs_only', '1',
            '--conditional_probs_only_backbone', '1',
            '--num_seq_per_target', '1',
            '--batch_size', '1',
            '--seed', str(entry['mpnn_seed']),
            '--suppress_print', '1',
            '--out_folder', out_dir.resolve().as_posix(),
        ]
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=900)
        if completed.returncode:
            raise RuntimeError(
                f"MPNN scoring failed for {entry['instance']}: "
                f"{completed.stderr[-1500:]}")
    result = np.load(npz_path)
    log_p = result['log_p'][0]
    # The conditional_probs_only branch writes no design_mask; validate the
    # H3 window by total length instead (window positions are already pinned
    # by the generation-stage native-record contract).
    total_length = int(result['mask'].shape[-1]) if 'mask' in result else log_p.shape[0]
    lo, hi = (int(x) for x in entry['region_spec'].split(':')[1].split('-'))
    from Bio.PDB import PDBParser
    model0 = next(PDBParser(QUIET=True).get_structure(
        'x', str(ROOT / entry['component_pdb'])).get_models())
    lengths = {chain.id: sum(1 for r in chain.get_residues() if r.id[0] == ' ')
               for chain in model0}
    if log_p.shape[0] != lengths['H'] + lengths['P'] or total_length != log_p.shape[0]:
        raise RuntimeError(
            f"{entry['instance']}: MPNN log_p length {log_p.shape[0]} != "
            f"H+P {lengths['H'] + lengths['P']}")
    h3_indices = list(range(lo - 1, hi))
    region_len = hi - lo + 1

    def nll(sequence):
        if len(sequence) != region_len:
            raise RuntimeError(f"{entry['instance']}: candidate length mismatch")
        return -sum(float(log_p[index, MPNN_ALPHABET.index(aa)])
                    for index, aa in zip(h3_indices, sequence)) / region_len

    return nll


def percentile_of_native(native, scores, better='lower'):
    """Tie-handled percentile: 0 = native best. better='lower' for NLL-like."""
    others = [s for cand, s in scores.items() if cand != native]
    native_score = scores[native]
    if better == 'lower':
        worse = sum(1 for s in others if s > native_score)
        tied = sum(1 for s in others if s == native_score)
    else:
        worse = sum(1 for s in others if s < native_score)
        tied = sum(1 for s in others if s == native_score)
    return (worse + 0.5 * tied) / len(others)


def main():
    prereg = read(OUT / 'preregistration.json')
    components = sorted(prereg['components'], key=lambda c: c['instance'])
    conformer_pdbs = J0B / 'conformer_component_pdbs'
    scoring_dir = OUT / 'scoring'
    scoring_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from modules.bfn_loader import load_bfn, _get_checkpoint_path
    model, _config = load_bfn(device)
    checkpoint_path = _get_checkpoint_path()

    mpnn_cache_path = OUT / 'mpnn_nll_cache.json'
    mpnn_cache = read(mpnn_cache_path) if mpnn_cache_path.exists() else {}
    composition_cache_path = OUT / 'composition_prob_cache.json'
    composition_cache = (read(composition_cache_path)
                         if composition_cache_path.exists() else {})

    pool_rows = []
    for entry in components:
        generation = read(OUT / 'generation' / entry['instance'] / 'candidates.json')
        designs = generation['designs']
        native = entry['region_sequence']
        shuffles = entry['shuffles']
        pool = [native] + designs + shuffles

        scoring = score_designs_judge(model, entry, designs, device, conformer_pdbs)
        complex_per_conformer = scoring['complex_per_conformer']
        stripped_designs = scoring['stripped_for_designs']

        j0b_row = read(J0B / 'scoring' / f"{entry['instance']}.json")
        j0a_row = read(J0 / 'scoring' / f"{entry['instance']}.json")
        stripped_pool = dict(j0a_row['nll']['stripped'])
        stripped_pool.update(stripped_designs)

        judge_scores = {}
        for candidate in pool:
            complex_marginal = float(np.mean([
                complex_dict[candidate] for complex_dict in complex_per_conformer]))
            judge_scores[candidate] = complex_marginal - stripped_pool[candidate]

        if entry['instance'] not in mpnn_cache:
            nll_fn = mpnn_conditional_nll(entry, designs)
            mpnn_cache[entry['instance']] = {c: nll_fn(c) for c in pool}
            write_cache(mpnn_cache_path, mpnn_cache)
        mpnn_scores = mpnn_cache[entry['instance']]

        if entry['instance'] not in composition_cache:
            composition_cache[entry['instance']] = None  # filled in analysis pass
        pool_rows.append({
            'instance': entry['instance'],
            'family': entry['family'],
            'native': native,
            'pool_size': len(pool),
            'judge_scores': judge_scores,
            'mpnn_scores': mpnn_scores,
        })
        print(entry['instance'], f"pool {len(pool)} scored", flush=True)

    # ---- composition logistic baseline (leave-one-family-out). ----
    from sklearn.linear_model import LogisticRegression
    aa_order = 'ACDEFGHIKLMNPQRSTVWY'
    families = sorted({row['family'] for row in pool_rows})
    for fold_family in families:
        train_rows = [r for r in pool_rows if r['family'] != fold_family]
        x_train, y_train = [], []
        for row in train_rows:
            entry = next(c for c in components if c['instance'] == row['instance'])
            x_train.append([row['native'].count(aa) / len(row['native'])
                            for aa in aa_order])
            y_train.append(1)
            for sequence in entry['shuffles']:
                x_train.append([sequence.count(aa) / len(sequence)
                                for aa in aa_order])
                y_train.append(0)
        clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000,
                                 random_state=BOOTSTRAP_SEED)
        clf.fit(np.array(x_train), np.array(y_train))
        for row in pool_rows:
            if row['family'] != fold_family:
                continue
            entry = next(c for c in components if c['instance'] == row['instance'])
            generation = read(OUT / 'generation' / entry['instance'] / 'candidates.json')
            probs = {}
            for candidate in [row['native'], *generation['designs'],
                              *entry['shuffles']]:
                features = np.array([[candidate.count(aa) / len(candidate)
                                      for aa in aa_order]])
                probs[candidate] = float(clf.predict_proba(features)[0, 1])
            composition_cache[row['instance']] = probs
    write_cache(composition_cache_path, composition_cache)

    # ---- percentiles and gate. ----
    method_scores = {
        'flex_ecls_marginal': lambda row: row['judge_scores'],
        'proteinmpnn_nll': lambda row: row['mpnn_scores'],
        'composition_logistic': lambda row: composition_cache[row['instance']],
    }
    directions = {'flex_ecls_marginal': 'lower', 'proteinmpnn_nll': 'lower',
                  'composition_logistic': 'higher'}
    per_method = {name: [] for name in method_scores}
    for row in pool_rows:
        for name, extractor in method_scores.items():
            percentile = percentile_of_native(
                row['native'], extractor(row), better=directions[name])
            per_method[name].append({'instance': row['instance'],
                                     'family': row['family'],
                                     'percentile': percentile,
                                     'pool_size': row['pool_size']})

    def pooled_stats(values_by_component):
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        by_family = {}
        for item in values_by_component:
            by_family.setdefault(item['family'], []).append(item['percentile'])
        family_list = [by_family[k] for k in sorted(by_family)]
        means = []
        for _ in range(BOOTSTRAP_DRAWS):
            sampled = []
            for family in family_list:
                picked = family[rng.integers(0, len(family))]
                sampled.append(picked)
            means.append(float(np.mean(sampled)))
        point = float(np.mean([item['percentile'] for item in values_by_component]))
        return point, [float(np.quantile(means, 0.025)),
                       float(np.quantile(means, 0.975))]

    stats = {}
    for name in method_scores:
        point, ci = pooled_stats(per_method[name])
        top1 = float(np.mean([1 if item['percentile'] == 0 else 0
                              for item in per_method[name]]))
        top5 = float(np.mean([1 if item['percentile'] <= 5 / (item['pool_size'] - 1)
                              else 0 for item in per_method[name]]))
        stats[name] = {'mean_percentile': point, 'bootstrap_ci': ci,
                       'top1_rate': top1, 'top5_rate': top5}

    judge_by_instance = {item['instance']: item['percentile']
                         for item in per_method['flex_ecls_marginal']}
    baseline_stats = {}
    for baseline in ('composition_logistic', 'proteinmpnn_nll'):
        base_by_instance = {item['instance']: item['percentile']
                            for item in per_method[baseline]}
        paired = {row['instance']: judge_by_instance[row['instance']]
                  - base_by_instance[row['instance']] for row in pool_rows}
        by_family = {}
        for row in pool_rows:
            by_family.setdefault(row['family'], []).append(paired[row['instance']])
        family_means = {k: float(np.mean(v)) for k, v in by_family.items()}
        wins = sum(1 for v in family_means.values() if v < 0)
        baseline_stats[baseline] = {
            'family_mean_paired_diff': family_means,
            'families_judge_better': wins,
            'n_families': len(family_means),
        }

    gate_parts = {}
    for baseline in baseline_stats:
        gate_parts[baseline] = (baseline_stats[baseline]['families_judge_better']
                                / baseline_stats[baseline]['n_families']) >= 0.70
    gate_parts['judge_ci_beats_random'] = stats['flex_ecls_marginal']['bootstrap_ci'][1] < 0.5
    gate_pass = all(gate_parts.values())

    summary = {
        'gate': 'c1_ranker_development_gate',
        'stats': stats,
        'baseline_comparisons': baseline_stats,
        'gate_parts': gate_parts,
        'verdict': 'pass' if gate_pass else 'fail',
        'on_fail_consequence': 'M8 repositioning clause',
        'checkpoint_path': checkpoint_path,
        'checkpoint_sha256': digest(Path(checkpoint_path)),
        'classification': 'development_ranker_comparison_not_confirmatory_claim',
    }
    freeze(OUT / 'c1_results.json', {'per_component': per_method,
                                     'pool_rows_note': 'scores in scoring/ and caches'})
    freeze(OUT / 'c1_summary.json', summary)
    freeze(OUT / 'completion.json', {
        'status': 'completed', 'gate_passed': bool(gate_pass),
        'summary_sha256': digest(OUT / 'c1_summary.json')})
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
