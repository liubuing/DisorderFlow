#!/usr/bin/env python
"""P2: Multi-conformation Ensemble Scoring for IDP antibody design.

Dual-line scoring:
  Line 1 (BFN):   BFN confidence scores across epitope conformations
  Line 2 (Phys):  Distance-based interface metrics (no freesasa dependency)

Usage:
  scorer = EnsembleScorer()
  enriched = scorer.score_designs_bfn(designs, epitope_sequence, model, device)
  enriched = scorer.score_designs_physics(designs, epitope_sequence)
  # Combine both for ensemble ranking
  final = ensemble_rank_dual(enriched)
"""
import os, sys, tempfile, shutil, pickle, glob, time
import numpy as np
import lmdb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFORMATION_DB = os.path.join(
    PROJECT_ROOT, 'data', 'confidence_conformation_v5', 'confidence_train.lmdb')
DEFAULT_SCAFFOLD_PDB = os.path.join(
    PROJECT_ROOT, 'data', 'misfolding_targets', '5IMK.pdb')
DEFAULT_EPITOPE_PDB = os.path.join(
    PROJECT_ROOT, 'data', 'misfolding_targets', '2NAO_model1_A_1-42.pdb')


class EnsembleScorer:
    """Score CDR designs against multiple epitope conformations."""

    def __init__(self, conformation_db_path=None, scaffold_pdb=None,
                 epitope_pdb=None):
        self.conformation_db_path = conformation_db_path or DEFAULT_CONFORMATION_DB
        self.scaffold_pdb = scaffold_pdb or DEFAULT_SCAFFOLD_PDB
        self.epitope_pdb = epitope_pdb or DEFAULT_EPITOPE_PDB
        self._env = None
        self._cache = {}

    def _get_env(self):
        if self._env is None:
            self._env = lmdb.open(self.conformation_db_path, readonly=True)
        return self._env

    def find_conformations(self, epitope_sequence):
        """Find multi-conformation data. Returns dict or None."""
        key = epitope_sequence[:30]
        if key in self._cache:
            return self._cache[key]
        # Try LMDB
        env = self._get_env()
        with env.begin() as txn:
            cursor = txn.cursor()
            for _, val in cursor:
                try:
                    entry = pickle.loads(val)
                except Exception:
                    continue
                if isinstance(entry, dict) and entry.get('sequence') == epitope_sequence:
                    self._cache[key] = entry
                    return entry
            cursor.close()
        # Try pickle files
        for pkl_path in glob.glob(os.path.join(
                PROJECT_ROOT, 'data', 'abeta_conformations', '*.pkl')):
            try:
                with open(pkl_path, 'rb') as f:
                    entry = pickle.load(f)
                if entry.get('sequence') == epitope_sequence:
                    self._cache[key] = entry
                    return entry
            except Exception:
                continue
        return None

    def build_conformation_pdbs(self, ca_positions_list, epitope_chain='A'):
        """Build PDB files for each epitope conformation."""
        with open(self.epitope_pdb) as f:
            epi_lines = f.readlines()
        epi_atoms = {}
        for line in epi_lines:
            if line.startswith('ATOM') and line[21] == epitope_chain:
                rn = int(line[22:26].strip())
                an = line[12:16].strip()
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                if rn not in epi_atoms:
                    epi_atoms[rn] = []
                epi_atoms[rn].append((line, x, y, z, an == 'CA', an))
        sorted_rn = sorted(epi_atoms.keys())
        n_res = len(sorted_rn)
        tmpdir = tempfile.mkdtemp(prefix='ensemble_')
        paths = []
        for conf_idx, ca_pos in enumerate(ca_positions_list):
            if ca_pos.shape[0] < n_res:
                continue
            out = os.path.join(tmpdir, f'conf_{conf_idx}.pdb')
            with open(self.scaffold_pdb) as f:
                sc_lines = f.readlines()
            with open(out, 'w') as f:
                for line in sc_lines:
                    if line.startswith('ATOM') or line.startswith('HETATM'):
                        f.write(line)
                for i, rn in enumerate(sorted_rn):
                    if i >= ca_pos.shape[0]:
                        break
                    new_ca = ca_pos[i]
                    atoms = epi_atoms[rn]
                    old_ca = None
                    for _, x, y, z, is_ca, _ in atoms:
                        if is_ca:
                            old_ca = np.array([x, y, z]); break
                    offset = (new_ca - old_ca) if old_ca is not None else np.zeros(3)
                    for line, _, _, _, _, _ in atoms:
                        x = float(line[30:38]) + offset[0]
                        y = float(line[38:46]) + offset[1]
                        z = float(line[46:54]) + offset[2]
                        f.write(f'{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}')
                f.write('END\n')
            paths.append(out)
        return paths, tmpdir

    # ── Line 1: BFN confidence scoring ──

    def score_designs_bfn(self, designs, epitope_sequence, model, device='cuda',
                           scaffold_chain='B', epitope_chain='A', n_confs=5):
        """Score designs using BFN confidence on each conformation."""
        conf_data = self.find_conformations(epitope_sequence)
        if conf_data is None:
            return designs

        ca_positions = conf_data['ca_positions'][:n_confs]
        rmsf = conf_data.get('rmsf', None)
        n_conf = len(ca_positions)

        conf_pdbs, tmpdir = self.build_conformation_pdbs(ca_positions, epitope_chain)
        from modules.bfn_loader import score_bfn_candidate

        cdr_spec = f'{scaffold_chain}:26-33,51-58,97-113'

        enriched = []
        for d in designs:
            candidate_sequence = d.get('sequence')
            if not candidate_sequence:
                raise ValueError('Every design must contain an exact candidate sequence')
            bfn_scores = []
            for pdb_path in conf_pdbs:
                try:
                    score = score_bfn_candidate(
                        pdb_path, cdr_spec, candidate_sequence,
                        context_chains=[epitope_chain], device=device, model=model)
                    bfn_scores.append({
                        'plddt': float(score['plddt'].mean().item()),
                        'iptm': float(score['iptm'].mean().item()),
                        'pae': float(score['pae'].mean().item()),
                        'state_compatibility': float(
                            score['state_compatibility'].mean().item()),
                    })
                except Exception as error:
                    raise RuntimeError(
                        f'Fixed-sequence ensemble scoring failed for {pdb_path}: {error}') from error

            iptms = [s['iptm'] for s in bfn_scores]
            plddts = [s['plddt'] for s in bfn_scores]
            compatibilities = [s['state_compatibility'] for s in bfn_scores]

            d_enriched = dict(d)
            d_enriched['ensemble_n_conf'] = n_conf
            d_enriched['bfn_iptm_mean'] = float(np.mean(iptms))
            d_enriched['bfn_iptm_worst'] = float(np.min(iptms))
            d_enriched['bfn_iptm_best'] = float(np.max(iptms))
            d_enriched['bfn_iptm_std'] = float(np.std(iptms))
            d_enriched['bfn_plddt_mean'] = float(np.mean(plddts))
            d_enriched['bfn_state_compatibility_mean'] = float(np.mean(compatibilities))
            d_enriched['bfn_state_compatibility_worst'] = float(np.min(compatibilities))
            d_enriched['bfn_state_compatibility_std'] = float(np.std(compatibilities))
            d_enriched['bfn_fixed_candidate_across_conformers'] = True
            if rmsf is not None:
                d_enriched['epitope_rmsf_mean'] = float(np.mean(rmsf))
            enriched.append(d_enriched)

        shutil.rmtree(tmpdir, ignore_errors=True)
        return enriched

    # ── Line 2: Physics-based scoring (numpy, no deps) ──

    def score_designs_physics(self, designs, epitope_sequence,
                               ab_chains=None, epitope_chain='A', n_confs=5):
        """Score designs using distance-based physics metrics."""
        if ab_chains is None:
            ab_chains = ['B']

        conf_data = self.find_conformations(epitope_sequence)
        if conf_data is None:
            return designs

        ca_positions = conf_data['ca_positions'][:n_confs]
        rmsf = conf_data.get('rmsf', None)
        n_conf = len(ca_positions)

        conf_pdbs, tmpdir = self.build_conformation_pdbs(ca_positions, epitope_chain)

        # Pre-compute per-conformation metrics (same for all designs since
        # CDR graft doesn't change backbone —this is a scaffold-level metric)
        conf_scores = []
        for pdb_path in conf_pdbs:
            ab_atoms, epi_atoms = [], []
            with open(pdb_path) as f:
                for line in f:
                    if line.startswith('ATOM'):
                        chain = line[21]
                        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                        if chain in ab_chains:
                            ab_atoms.append(np.array([x, y, z]))
                        elif chain == epitope_chain:
                            epi_atoms.append(np.array([x, y, z]))
            if not ab_atoms or not epi_atoms:
                conf_scores.append({'contacts': 0, 'min_dist': 99, 'score': 0.0})
                continue
            ab_arr = np.array(ab_atoms)
            epi_arr = np.array(epi_atoms)
            dists = np.sqrt(((ab_arr[:, None] - epi_arr[None, :]) ** 2).sum(-1))
            n_close = int((dists < 5.0).sum())  # close contacts
            n_medium = int(((dists >= 5.0) & (dists < 8.0)).sum())
            min_dist = float(dists.min())
            score = (n_close * 2.0 + n_medium * 1.0) / max(len(ab_atoms), 1)
            conf_scores.append({
                'contacts_close': n_close, 'contacts_medium': n_medium,
                'min_dist': min_dist, 'score': score,
            })

        scores = [s['score'] for s in conf_scores]
        close = [s['contacts_close'] for s in conf_scores]
        min_d = [s['min_dist'] for s in conf_scores]

        enriched = []
        for d in designs:
            d_enriched = dict(d)
            d_enriched['ensemble_n_conf'] = n_conf
            d_enriched['phys_score_mean'] = float(np.mean(scores))
            d_enriched['phys_score_worst'] = float(np.min(scores))
            d_enriched['phys_score_best'] = float(np.max(scores))
            d_enriched['phys_score_std'] = float(np.std(scores))
            d_enriched['phys_contacts_mean'] = float(np.mean(close))
            d_enriched['phys_min_dist_mean'] = float(np.mean(min_d))
            if rmsf is not None:
                d_enriched['epitope_rmsf_mean'] = float(np.mean(rmsf))
            enriched.append(d_enriched)

        shutil.rmtree(tmpdir, ignore_errors=True)
        return enriched

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None


def ensemble_rank_dual(designs, w_bfn_mean=0.25, w_bfn_worst=0.20,
                        w_phys_mean=0.20, w_phys_worst=0.15,
                        w_std=0.10, w_ppl=0.10):
    """Rank designs combining BFN and Physics ensemble metrics.

    Higher = better design. Both BFN metrics (higher=better) and Physics
    metrics (higher=better) are normalized. Std is inverted (lower=better).
    """
    if not designs:
        return designs

    has_bfn = any('bfn_iptm_mean' in d for d in designs)
    has_phys = any('phys_score_mean' in d for d in designs)

    if not has_bfn and not has_phys:
        return designs

    def norm(vals, invert=False):
        vmin, vmax = min(vals), max(vals)
        if vmax == vmin:
            return [0.5] * len(vals)
        n = [(v - vmin) / (vmax - vmin) for v in vals]
        return [1 - x if invert else x for x in n]

    scores = np.zeros(len(designs))
    active_w = 0.0

    if has_bfn:
        bfn_mean = norm([d.get('bfn_iptm_mean', 0) for d in designs])
        bfn_worst = norm([d.get('bfn_iptm_worst', 0) for d in designs])
        bfn_std = norm([d.get('bfn_iptm_std', 0) for d in designs], invert=True)
        scores += w_bfn_mean * np.array(bfn_mean)
        scores += w_bfn_worst * np.array(bfn_worst)
        scores += w_std * np.array(bfn_std)
        active_w += w_bfn_mean + w_bfn_worst + w_std

    if has_phys:
        phys_mean = norm([d.get('phys_score_mean', 0) for d in designs])
        phys_worst = norm([d.get('phys_score_worst', 0) for d in designs])
        scores += w_phys_mean * np.array(phys_mean)
        scores += w_phys_worst * np.array(phys_worst)
        active_w += w_phys_mean + w_phys_worst

    # PPL (lower = better)
    ppl_vals = [d.get('ppl', 100) or 100 for d in designs]
    ppl_norm = norm(ppl_vals, invert=True)
    scores += w_ppl * np.array(ppl_norm)
    active_w += w_ppl

    if active_w > 0:
        scores = scores / active_w

    for i, d in enumerate(designs):
        d['ensemble_dual_score'] = round(float(scores[i]), 4)

    # Sort descending
    ranked = sorted(designs, key=lambda d: d.get('ensemble_dual_score', 0), reverse=True)
    for i, d in enumerate(ranked):
        d['ensemble_dual_rank'] = i + 1

    return ranked
