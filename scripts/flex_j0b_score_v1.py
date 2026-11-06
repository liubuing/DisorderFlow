"""FLEX-BINDER Gate J0b, ensemble-marginalized scoring.

For every component with a VALID ensemble (declared rule in the J0b build
protocol): for each accepted conformer, build a scoring PDB with the DEPOSITED
heavy chain and the conformer's peptide coordinates, then score the native H3
and its 50 shuffles in two contexts. The stripped context file is identical
across conformers of a component, so the contrast isolates the effect of
peptide conformational uncertainty on the judge.

Marginalization (frozen in the preregistration wording): per candidate,
ECLS_marginal = mean over conformers of (NLL_complex - NLL_stripped);
advantage = mean over shuffles of ECLS_marginal - native ECLS_marginal.
The alternative order (mean of per-conformer advantages) is recorded
descriptively as well.

Gate: same thresholds as J0a over the valid components' antigen families
(>= 70% families positive AND family bootstrap 95% CI excludes zero).
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest
from scripts.flex_j0_score_v1 import (
    BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, FIXED_T, SCORING_SEEDS, candidate_nll)

OUT = ROOT / 'results/flex_j0b_v1'
J0 = ROOT / 'results/flex_j0_v1'


def conformer_component_pdb(entry, conformer_pdb_path, out_path,
                            heavy_rows_cache):
    """Deposited heavy rows + conformer peptide rows, written in J0a layout."""
    from scripts.materialize_pae_analog_structures_v26 import (
        strict_residues, write_component_pdb)

    key = entry['instance']
    if key not in heavy_rows_cache:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(entry['instance'],
                                         str(J0 / 'pdb' / f"{entry['instance']}.pdb"))
        model = next(structure.get_models())
        heavy_rows_cache[key] = strict_residues(model['H'], allow_mse=True)
    conformer = PDBParser(QUIET=True).get_structure(
        entry['instance'], str(conformer_pdb_path))
    conformer_model = next(conformer.get_models())
    peptide_rows = strict_residues(conformer_model['P'], allow_mse=True)
    peptide_seq = ''.join(r['aa'] for r in peptide_rows)
    if peptide_seq != entry['peptide_sequence']:
        raise RuntimeError(
            f"{entry['instance']}: conformer peptide sequence mismatch "
            f"{peptide_seq} != {entry['peptide_sequence']}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    write_component_pdb(out_path, [('H', heavy_rows_cache[key]),
                                   ('P', peptide_rows)])
    return out_path


def marginal_ecls(per_conformer_complex, per_conformer_stripped, candidate):
    diffs = [complex_nll[candidate] - stripped_nll[candidate]
             for complex_nll, stripped_nll in
             zip(per_conformer_complex, per_conformer_stripped)]
    return float(np.mean(diffs)), diffs


def main():
    build = read(OUT / 'build_summary.json')
    cohort = read(J0 / 'cohort.json')
    entries = {e['instance']: e for e in cohort['components']}
    valid = [row for row in build['rows'] if row['valid']]
    print(f'valid ensembles: {len(valid)}/{len(build["rows"])}', flush=True)

    scoring_dir = OUT / 'scoring'
    scoring_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from modules.bfn_loader import load_bfn, _get_checkpoint_path
    model, _config = load_bfn(device)
    checkpoint_path = _get_checkpoint_path()

    from scripts.materialize_pae_analog_structures_v26 import strict_residues
    heavy_rows_cache = {}
    conformer_pdbs_dir = OUT / 'conformer_component_pdbs'
    conformer_pdbs_dir.mkdir(parents=True, exist_ok=True)

    j0a_stripped = {}
    for instance_cached in sorted((J0 / 'scoring').glob('*.json')):
        cached_row = read(instance_cached)
        j0a_stripped[cached_row['instance']] = cached_row['nll']['stripped']

    verification = None
    rows = []
    for row in valid:
        entry = entries[row['instance']]
        audit = read(OUT / 'ensembles' / entry['instance'] / 'ensemble_audit.json')
        accepted = [c for c in audit['conformers'] if c['status'] == 'accepted']
        cache_path = scoring_dir / f"{entry['instance']}.json"
        if cache_path.exists():
            rows.append(read(cache_path))
            print(row['instance'], 'cached', flush=True)
            continue
        if entry['instance'] not in j0a_stripped:
            raise RuntimeError(f"{entry['instance']}: no J0a stripped cache to reuse")
        from modules.bfn_loader import build_region_batch
        per_conformer_complex = []
        candidates = [entry['region_sequence']] + entry['shuffles']
        for conformer_index, conformer in enumerate(accepted):
            source = ROOT / conformer['pdb']
            target = conformer_pdbs_dir / f"{entry['instance']}_{Path(source).stem}.pdb"
            conformer_component_pdb(entry, source, target, heavy_rows_cache)
            complex_batch = build_region_batch(
                str(target), entry['region_spec'], context_chains=['P'],
                device=device)
            complex_nll = {}
            for candidate in candidates:
                complex_nll[candidate] = float(np.mean([
                    candidate_nll(model, complex_batch, candidate, seed, device)[0]
                    for seed in SCORING_SEEDS]))
            per_conformer_complex.append(complex_nll)
            # One-time equivalence check: the stripped context (heavy chain
            # only) is identical by construction to the J0a stripped batch,
            # so cached J0a values must reproduce bit-near-exactly.
            if verification is None:
                stripped_batch = build_region_batch(
                    str(target), entry['region_spec'], context_chains=[],
                    device=device)
                max_diff = 0.0
                for candidate in candidates[:3]:
                    fresh = float(np.mean([
                        candidate_nll(model, stripped_batch, candidate, seed, device)[0]
                        for seed in SCORING_SEEDS]))
                    max_diff = max(max_diff, abs(fresh - j0a_stripped[entry['instance']][candidate]))
                if max_diff > 1e-9:
                    raise RuntimeError(
                        f"stripped equivalence check failed (max diff {max_diff}); "
                        'refusing to reuse J0a stripped cache')
                verification = {'instance': entry['instance'],
                                'conformer_seed': conformer['seed'],
                                'max_abs_diff': max_diff,
                                'candidates_checked': 3}
            print(f"  {entry['instance']} conformer {conformer['seed']} scored",
                  flush=True)
        result = {
            'instance': entry['instance'],
            'n_conformers': len(accepted),
            'nll_complex_per_conformer': per_conformer_complex,
            'stripped_source': 'j0a_cache',
            'nll_stripped_per_conformer': [j0a_stripped[entry['instance']]]
                                          * len(accepted),
            'candidates_order_note': 'dicts keyed by candidate sequence',
        }
        freeze(cache_path, result)
        rows.append(result)
        print(row['instance'], f"scored ({len(accepted)} conformers)", flush=True)

    # ---- Endpoint over valid components. ----
    prereg = read(J0 / 'preregistration.json')
    family_of = {}
    for family_key, members in prereg['cohort']['antigen_families'].items():
        for peptide in members:
            family_of[peptide] = family_key

    advantages = []
    for result in rows:
        entry = entries[result['instance']]
        native = entry['region_sequence']
        marginal_native, _ = marginal_ecls(
            result['nll_complex_per_conformer'],
            result['nll_stripped_per_conformer'], native)
        shuffle_marginal = [marginal_ecls(
            result['nll_complex_per_conformer'],
            result['nll_stripped_per_conformer'], shuffle)[0]
            for shuffle in entry['shuffles']]
        advantage = float(np.mean(shuffle_marginal)) - marginal_native
        # Descriptive alternative order: mean of per-conformer advantages.
        per_conformer_advantages = [
            float(np.mean([complex_nll[s] - stripped_nll[s]
                           for s in entry['shuffles']]))
            - (complex_nll[native] - stripped_nll[native])
            for complex_nll, stripped_nll in
            zip(result['nll_complex_per_conformer'],
                result['nll_stripped_per_conformer'])]
        advantages.append({
            'instance': entry['instance'],
            'pdb': entry['pdb_id'],
            'peptide': entry['peptide_sequence'],
            'family': family_of[entry['peptide_sequence']],
            'ecls_native_marginal': marginal_native,
            'advantage': advantage,
            'advantage_mean_of_per_conformer': float(np.mean(per_conformer_advantages)),
        })

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
    gate_pass = positive_fraction >= 0.70 and ci[0] > 0

    j0a = read(J0 / 'j0a_summary.json')
    summary = {
        'gate': 'j0b_ensemble',
        'n_components_valid': len(advantages),
        'n_families': len(family_values),
        'positive_families': positive,
        'positive_fraction': positive_fraction,
        'mean_advantage': float(values.mean()),
        'bootstrap_ci_family_mean': ci,
        'mean_advantage_alt_order': float(np.mean([r['advantage_mean_of_per_conformer']
                                                   for r in advantages])),
        'j0a_mean_advantage_reference': j0a['mean_advantage'],
        'attenuation_vs_j0a': float(j0a['mean_advantage'] - values.mean()),
        'nll_field': 'seq',
        'stripped_source': 'j0a_cache_with_equivalence_verification',
        'stripped_equivalence_verification': verification,
        'scoring_seeds': SCORING_SEEDS,
        'fixed_t': FIXED_T,
        'checkpoint_path': checkpoint_path,
        'checkpoint_sha256': digest(Path(checkpoint_path)),
        'verdict': 'pass' if gate_pass else 'fail',
        'classification': 'development_gate_not_confirmatory_claim',
    }
    freeze(OUT / 'j0b_results.json', {'components': advantages})
    freeze(OUT / 'j0b_summary.json', summary)
    freeze(OUT / 'completion.json', {
        'status': 'completed',
        'phase_c_unlocked': bool(gate_pass),
        'summary_sha256': digest(OUT / 'j0b_summary.json')})
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
