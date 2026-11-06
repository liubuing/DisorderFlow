"""FLEX-BINDER Gate J0b re-scoring with corrected chain-role encoding (v2).

Re-runs the ensemble-marginalized J0b gate on the SAME frozen ensembles with
the intended Fragment encoding (H -> Heavy, P -> Antigen, other -> Light),
mirroring scripts/flex_j0b_score_v1.py. Differences from v1:

* complex batches are built with ``antigen_chains=['P']``.  The frozen J0b v1
  numbers were produced by a batch where the heavy chain was encoded as
  padding (index 0), the peptide as Heavy (index 1) and ``mask_antigen`` was
  empty, silently disabling the receiver's antigen branch (see
  .dsh-research/flex-distance/code-fix-report.md);
* stripped NLLs are reused from the CORRECTED J0a v2 cache
  (results/flex_j0_v2_antigen_roles/scoring), with the same one-time
  bit-exactness verification v1 applied against its J0a cache.

Frozen J0b v1 artifacts are never touched; output goes to
results/flex_j0b_v2_antigen_roles/ (write-once; corrections open a new
directory).

Usage:
  python scripts/flex_j0b_score_v2_antigen_roles.py --instances 2x6m_A_B --limit-conformers 1  # smoke
  python scripts/flex_j0b_score_v2_antigen_roles.py --all
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest  # noqa: E402
from scripts.flex_j0_score_v1 import (  # noqa: E402
    BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, FIXED_T, SCORING_SEEDS, candidate_nll)
from scripts.flex_j0b_score_v1 import conformer_component_pdb, marginal_ecls  # noqa: E402

J0B_V1 = ROOT / 'results/flex_j0b_v1'
J0 = ROOT / 'results/flex_j0_v1'
J0A_V2 = ROOT / 'results/flex_j0_v2_antigen_roles'
OUT = ROOT / 'results/flex_j0b_v2_antigen_roles'


def build_batches(pdb_path, region_spec, device):
    """Same two contexts as J0b v1, but with the intended Fragment encoding."""
    from modules.bfn_loader import build_region_batch

    complex_batch = build_region_batch(
        pdb_path, region_spec, context_chains=['P'],
        antigen_chains=['P'], device=device)
    stripped_batch = build_region_batch(
        pdb_path, region_spec, context_chains=[],
        antigen_chains=[], device=device)
    return complex_batch, stripped_batch


def role_report(batch):
    """Record the fragment_type layout actually fed to the model."""
    from disorderflow.utils.protein.constants import Fragment
    ft = batch['fragment_type'][0]
    chains = [c[0] if isinstance(c, (list, tuple)) else c for c in batch['chain_id']]
    per_chain: dict[str, list[int]] = {}
    for index, chain in enumerate(chains):
        per_chain.setdefault(chain, [])
        value = int(ft[index])
        if value not in per_chain[chain]:
            per_chain[chain].append(value)
    return {
        'length': int(ft.shape[0]),
        'fragment_type_by_chain': {k: sorted(v) for k, v in per_chain.items()},
        'n_heavy': int((ft == int(Fragment.Heavy)).sum()),
        'n_light': int((ft == int(Fragment.Light)).sum()),
        'n_antigen': int((ft == int(Fragment.Antigen)).sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instances', default='')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--limit-conformers', type=int, default=0,
                        help='debug only: first N accepted conformers (0 = all)')
    parser.add_argument('--out', default=str(OUT),
                        help='debug only: redirect output (smoke tests)')
    args = parser.parse_args()
    if not args.all and not args.instances:
        raise SystemExit('pass --instances a,b or --all')
    out_dir = Path(args.out)

    build = read(J0B_V1 / 'build_summary.json')
    cohort = read(J0 / 'cohort.json')
    entries = {e['instance']: e for e in cohort['components']}
    valid = [row for row in build['rows'] if row['valid']]
    if args.instances:
        wanted = [x for x in args.instances.split(',') if x]
        missing = set(wanted) - {row['instance'] for row in valid}
        if missing:
            raise SystemExit(f'not valid ensembles: {sorted(missing)}')
        valid = [row for row in valid if row['instance'] in wanted]
    print(f'valid ensembles: {len(valid)}/{len(build["rows"])}', flush=True)

    scoring_dir = out_dir / 'scoring'
    scoring_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from modules.bfn_loader import load_bfn, _get_checkpoint_path
    model, _config = load_bfn(device)
    checkpoint_path = _get_checkpoint_path()

    from scripts.materialize_pae_analog_structures_v26 import strict_residues
    heavy_rows_cache = {}
    conformer_pdbs_dir = J0B_V1 / 'conformer_component_pdbs'
    conformer_pdbs_dir.mkdir(parents=True, exist_ok=True)

    # Corrected-encoding stripped NLLs, per component: {candidate: nll}.
    j0a_v2_stripped = {}
    for instance_cached in sorted(J0A_V2.glob('scoring/*.json')):
        cached_row = read(instance_cached)
        j0a_v2_stripped[cached_row['instance']] = cached_row['nll']['stripped']

    verification = None
    rows = []
    for row in valid:
        entry = entries[row['instance']]
        audit = read(J0B_V1 / 'ensembles' / entry['instance'] / 'ensemble_audit.json')
        accepted = [c for c in audit['conformers'] if c['status'] == 'accepted']
        if args.limit_conformers:
            accepted = accepted[:args.limit_conformers]
        cache_path = scoring_dir / f"{entry['instance']}.json"
        if cache_path.exists():
            rows.append(read(cache_path))
            print(row['instance'], 'cached', flush=True)
            continue
        if entry['instance'] not in j0a_v2_stripped:
            raise RuntimeError(f"{entry['instance']}: no corrected J0a v2 stripped cache")
        per_conformer_complex = []
        candidates = [entry['region_sequence']] + entry['shuffles']
        role_encoding = None
        for conformer_index, conformer in enumerate(accepted):
            source = ROOT / conformer['pdb']
            target = conformer_pdbs_dir / f"{entry['instance']}_{Path(source).stem}.pdb"
            conformer_component_pdb(entry, source, target, heavy_rows_cache)
            complex_batch, stripped_batch = build_batches(
                str(target), entry['region_spec'], device=device)
            if role_encoding is None:
                role_encoding = {'complex': role_report(complex_batch),
                                 'stripped': role_report(stripped_batch)}
            complex_nll = {}
            for candidate in candidates:
                complex_nll[candidate] = float(np.mean([
                    candidate_nll(model, complex_batch, candidate, seed, device)[0]
                    for seed in SCORING_SEEDS]))
            per_conformer_complex.append(complex_nll)
            # One-time equivalence check: the stripped context (heavy chain
            # only, antigen_chains=[]) must reproduce the corrected J0a v2
            # stripped values bit-near-exactly, else the reuse is invalid.
            if verification is None:
                max_diff = 0.0
                for candidate in candidates[:3]:
                    fresh = float(np.mean([
                        candidate_nll(model, stripped_batch, candidate, seed, device)[0]
                        for seed in SCORING_SEEDS]))
                    max_diff = max(max_diff, abs(
                        fresh - j0a_v2_stripped[entry['instance']][candidate]))
                if max_diff > 1e-9:
                    raise RuntimeError(
                        f"stripped equivalence check failed (max diff {max_diff}); "
                        'refusing to reuse corrected J0a v2 stripped cache')
                verification = {'instance': entry['instance'],
                                'conformer_seed': conformer['seed'],
                                'max_abs_diff': max_diff,
                                'candidates_checked': 3}
            print(f"  {entry['instance']} conformer {conformer['seed']} scored",
                  flush=True)
        result = {
            'instance': entry['instance'],
            'n_conformers': len(accepted),
            'role_encoding': role_encoding,
            'nll_complex_per_conformer': per_conformer_complex,
            'stripped_source': 'j0a_v2_cache_corrected_roles',
            'nll_stripped_per_conformer': [j0a_v2_stripped[entry['instance']]]
                                          * len(accepted),
            'candidates_order_note': 'dicts keyed by candidate sequence',
        }
        freeze(cache_path, result)
        rows.append(result)
        print(row['instance'], f"scored ({len(accepted)} conformers)", flush=True)

    # ---- Endpoint over scored components. ----
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

    j0a_v2 = read(J0A_V2 / 'j0a_v2_summary.json')
    summary = {
        'gate': 'j0b_ensemble_corrected_roles',
        'n_components_valid': len(advantages),
        'n_families': len(family_values),
        'positive_families': positive,
        'positive_fraction': positive_fraction,
        'mean_advantage': float(values.mean()),
        'bootstrap_ci_family_mean': ci,
        'mean_advantage_alt_order': float(np.mean([r['advantage_mean_of_per_conformer']
                                                   for r in advantages])),
        'j0a_v2_mean_advantage_reference': j0a_v2['mean_advantage'],
        'attenuation_vs_j0a_v2': float(j0a_v2['mean_advantage'] - values.mean()),
        'frozen_j0b_v1_mean_advantage_reference': 0.141,
        'role_encoding_note': (
            'H -> Fragment.Heavy(1), P -> Fragment.Antigen(3), '
            'other context -> Fragment.Light(2)'),
        'nll_field': 'seq',
        'stripped_source': 'j0a_v2_cache_corrected_roles_with_equivalence_verification',
        'stripped_equivalence_verification': verification,
        'scoring_seeds': SCORING_SEEDS,
        'fixed_t': FIXED_T,
        'checkpoint_path': checkpoint_path,
        'checkpoint_sha256': digest(Path(checkpoint_path)),
        'verdict': 'pass' if gate_pass else 'fail',
        'classification': 'development_gate_not_confirmatory_claim',
        'provenance': 're-scores frozen results/flex_j0b_v1 ensembles; '
                      'frozen J0b v1 artifacts untouched',
    }
    freeze(OUT / 'j0b_v2_results.json', {'components': advantages})
    freeze(OUT / 'j0b_v2_summary.json', summary)
    freeze(OUT / 'completion.json', {
        'status': 'completed',
        'summary_sha256': digest(OUT / 'j0b_v2_summary.json')})
    print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
