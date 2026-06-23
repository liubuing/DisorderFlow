#!/usr/bin/env python
"""Multi-objective scorer and Pareto optimizer for closed-loop antibody design.

Combines AF2 confidence (ipTM, pLDDT, PAE), Rosetta physics (dG, ddG),
and sequence quality (PPL, entropy) into unified multi-objective rankings.

Supports:
  1. Weighted linear scalarization (backward-compatible with cascade_filter)
  2. Non-dominated sorting (NSGA-II style Pareto front extraction)
  3. Hypervolume computation for convergence tracking
"""

import math
from typing import List, Dict, Optional, Tuple, Set

# ── Disorder → AF2 Reliability Mapping ──

def disorder_to_af2_reliability(disorder: float) -> float:
    """Map BFN disorder score → AF2 reliability factor [0, 1].

    Calibrated against multi-conformation library (863 IDPs × 5 seeds):
      disorder < 0.2 (RMSF < 0.4Å): AF2 ensemble tight → trust AF2
      disorder = 0.3 (RMSF ≈ 0.6Å): moderate spread
      disorder > 0.5 (RMSF > 1.1Å): AF2 predictions diverge → distrust

    Sigmoid midpoint at disorder=0.35 (observed RMSF > 1.0Å transition).
    """
    return 1.0 / (1.0 + math.exp(12 * (disorder - 0.35)))


def calibrate_results_with_disorder(results: List[Dict]) -> List[Dict]:
    """Apply disorder-aware AF2 calibration to a batch of design results.

    For each result that has both AF2 scores and epitope_disorder info:
      - Computes AF2 reliability from disorder
      - Blends AF2 scores with BFN self-confidence
      - Adjusts BFN confidence values in-place (iptm, plddt get blended;
        ppl/entropy are BFN-native and kept unchanged)
      - Adds calibration metadata keys

    This is designed as a pre-processing step before composite scoring.
    Call before normalize_scores() or compute_weighted_composite().

    Args:
        results: list of dicts, each with optional:
            - iptm, plddt (BFN self-confidence)
            - af2_iptm, af2_plddt, af2_interface_pae (AF2 validation)
            - epitope_disorder (float [0,1] or None)
            - segment_mean_disorder (fallback if epitope_disorder missing)

    Returns:
        Same list with calibrated values and metadata added.
    """
    for r in results:
        # Get disorder score (try multiple key names)
        epi_disorder = r.get('epitope_disorder')
        if epi_disorder is None:
            epi_disorder = r.get('segment_mean_disorder')
        if epi_disorder is None:
            r['_af2_calibrated'] = None
            continue

        af2_iptm = r.get('af2_iptm')
        af2_plddt = r.get('af2_plddt')
        if af2_iptm is None and af2_plddt is None:
            r['_af2_calibrated'] = None
            continue

        reliability = disorder_to_af2_reliability(epi_disorder)

        bfn_iptm = r.get('iptm', 0) or 0
        bfn_plddt = r.get('plddt', 0) or 0

        # Blend AF2 with BFN based on reliability
        cal_iptm = (af2_iptm or 0) * reliability + bfn_iptm * (1 - reliability)
        cal_plddt = (af2_plddt or 0) * reliability + bfn_plddt * (1 - reliability)

        # Store calibration for reporting
        if reliability >= 0.80:
            label = 'HIGH'
        elif reliability >= 0.40:
            label = 'MEDIUM'
        else:
            label = 'LOW'

        r['_af2_calibrated'] = {
            'reliability': round(reliability, 4),
            'label': label,
            'epitope_disorder': round(epi_disorder, 4),
            'af2_raw_iptm': af2_iptm,
            'af2_raw_plddt': af2_plddt,
            'calibrated_iptm': round(cal_iptm, 4),
            'calibrated_plddt': round(cal_plddt, 4),
        }

        # Override AF2 scores with calibrated values for downstream scoring
        r['af2_iptm'] = cal_iptm
        r['af2_plddt'] = cal_plddt
        if r.get('af2_interface_pae') is not None:
            bfn_pae = r.get('pae', 30) or 30
            r['af2_interface_pae'] = (r['af2_interface_pae'] * reliability
                                       + bfn_pae * (1 - reliability))

    return results


# ── Objective dimension definitions ──
# direction: 'maximize' = higher is better, 'minimize' = lower is better
OBJECTIVE_DIMS: Dict[str, str] = {
    'iptm':       'maximize',
    'plddt':      'maximize',
    'ptm':        'maximize',
    'max_pae':    'minimize',
    'dG':         'minimize',
    'ddG':        'minimize',
    'relax_e':    'minimize',
    'ppl':        'minimize',
    'entropy':    'minimize',
    'contacts':   'maximize',
    'charge_comp':'maximize',
    'recovery':   'maximize',
    # Negative-design + developability dimensions
    'off_target_iptm': 'minimize',   # ipTM against off-target (e.g. Transferrin) — lower = more specific
    'md_rmsd':          'minimize',  # 100ns MD backbone RMSD (Å) — lower = more stable
}

# Default weights for weighted composite scoring
MULTI_OBJECTIVE_WEIGHTS: Dict[str, float] = {
    'iptm':       0.25,
    'plddt':      0.15,
    'ptm':        0.10,
    'max_pae':    0.05,
    'dG':         0.15,
    'ddG':        0.05,
    'relax_e':    0.05,
    'ppl_inv':    0.10,
    'entropy_inv':0.05,
    'contacts':   0.05,
    'recovery':   0.05,
    # Negative-design + developability (light weights; mainly enforced as hard thresholds)
    'off_target_iptm': 0.10,
    'md_rmsd':         0.05,
}

# Hard thresholds for Stage 1 filtering
MULTI_OBJECTIVE_THRESHOLDS: Dict[str, float] = {
    'iptm_min':      0.25,
    'plddt_min':     50.0,
    'ptm_min':       0.25,
    'max_pae_max':   20.0,
    'dG_max':        5.0,      # allow slightly positive (weak binders)
    'ddG_max':       10.0,
    'ppl_max':       200.0,
    'entropy_max':   2.8,
    # Negative-design + developability hard cut-offs
    'off_target_iptm_max': 0.30,   # reject designs that also bind the off-target
    'md_rmsd_max':         3.0,    # reject structurally unstable designs (Å, 100ns)
    'immunogenicity_max':  0.6,    # reject highly immunogenic designs (proxy 0–1)
}


class MultiObjectiveRanker:
    """Normalize, score, and rank designs using multi-objective methods.

    Two strategies:
      1. Weighted composite: linear scalarization → single score
      2. Non-dominated sorting: Pareto front assignmenF → crowding distance
    """

    def __init__(self, weights=None, thresholds=None):
        self.weights = (weights or MULTI_OBJECTIVE_WEIGHTS).copy()
        self.thresholds = (thresholds or MULTI_OBJECTIVE_THRESHOLDS).copy()

    # ── Normalization ──

    @staticmethod
    def normalize_scores(results: List[Dict],
                         dims: Optional[List[str]] = None) -> List[Dict]:
        """Normalize each objective dimension to [0,1] across the batch.

        For 'maximize' dims, values closer to max → 1.0.
        For 'minimize' dims, values closer to min → 1.0.
        Missing dims are left as None in _norm keys.

        Args:
            results: list of dicts, each with objective values (iptm, plddt, ...)
            dims: optional subset of OBJECTIVE_DIMS keys to normalize

        Returns:
            Same list with added '_norm' keys (e.g. 'iptm_norm', 'dG_norm')
        """
        if dims is None:
            dims = list(OBJECTIVE_DIMS.keys())

        for dim in dims:
            vals = []
            for r in results:
                v = r.get(dim)
                if v is not None:
                    vals.append(v)

            if not vals or len(vals) < 2:
                for r in results:
                    r[f'{dim}_norm'] = 1.0 if r.get(dim) is not None else 0.0
                continue

            vmin, vmax = min(vals), max(vals)
            rng = vmax - vmin

            direction = OBJECTIVE_DIMS.get(dim, 'maximize')
            for r in results:
                v = r.get(dim)
                if v is None:
                    r[f'{dim}_norm'] = 0.0
                elif rng < 1e-9:
                    r[f'{dim}_norm'] = 1.0
                elif direction == 'maximize':
                    r[f'{dim}_norm'] = (v - vmin) / rng
                else:
                    r[f'{dim}_norm'] = (vmax - v) / rng

        return results

    # ── Stage 1: Hard thresholds ──

    def apply_hard_thresholds(self, results: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
        """Filter designs that violate hard thresholds.

        Returns:
            passed: list of results passing all thresholds
            rejected_counts: dict of reason → count
        """
        th = self.thresholds
        passed = []
        rejected_counts = {
            'plddt': 0, 'iptm': 0, 'ptm': 0, 'max_pae': 0,
            'dG': 0, 'ddG': 0, 'ppl': 0, 'entropy': 0,
            'off_target': 0, 'md': 0, 'immuno': 0, 'total': 0,
        }

        for r in results:
            rejected = False

            plddt = r.get('plddt')
            if plddt is not None and plddt < th.get('plddt_min', -1):
                rejected_counts['plddt'] += 1; rejected = True; continue

            iptm = r.get('iptm')
            if iptm is not None and iptm < th.get('iptm_min', -1):
                rejected_counts['iptm'] += 1; rejected = True; continue

            ptm = r.get('ptm')
            if ptm is not None and ptm < th.get('ptm_min', -1):
                rejected_counts['ptm'] += 1; rejected = True; continue

            max_pae = r.get('max_pae')
            if max_pae is not None and max_pae > th.get('max_pae_max', 999):
                rejected_counts['max_pae'] += 1; rejected = True; continue

            dG = r.get('dG')
            if dG is not None and dG > th.get('dG_max', 999):
                rejected_counts['dG'] += 1; rejected = True; continue

            ddG = r.get('ddG')
            if ddG is not None and ddG > th.get('ddG_max', 999):
                rejected_counts['ddG'] += 1; rejected = True; continue

            ppl = r.get('ppl')
            if ppl is not None and ppl > th.get('ppl_max', 999):
                rejected_counts['ppl'] += 1; rejected = True; continue

            entropy = r.get('entropy')
            if entropy is not None and entropy > th.get('entropy_max', 999):
                rejected_counts['entropy'] += 1; rejected = True; continue

            # Negative-design: off-target (Transferrin) ipTM
            off_iptm = r.get('off_target_iptm')
            if off_iptm is not None and off_iptm > th.get('off_target_iptm_max', 999):
                rejected_counts['off_target'] += 1; rejected = True; continue

            # Developability: MD backbone RMSD (Å)
            md_rmsd = r.get('md_rmsd')
            if md_rmsd is not None and md_rmsd > th.get('md_rmsd_max', 999):
                rejected_counts['md'] += 1; rejected = True; continue

            # Developability: immunogenicity (proxy, 0–1)
            immuno = r.get('immunogenicity')
            if immuno is not None and immuno > th.get('immunogenicity_max', 999):
                rejected_counts['immuno'] += 1; rejected = True; continue

            passed.append(r)

        rejected_counts['total'] = len(results) - len(passed)
        return passed, rejected_counts

    # ── Stage 3: Weighted composite ──

    def compute_weighted_composite(self, results: List[Dict]) -> List[Dict]:
        """Compute weighted linear scalarization composite score.

        Adds keys to each result:
          mo_composite_score: float in [0, 1]
          mo_rank: int (1-based, best=1)

        Returns results sorted by mo_composite_score descending.
        """
        # First normalize all active dimensions
        active_dims = [k for k in self.weights if k in OBJECTIVE_DIMS
                       or k.replace('_inv', '') in OBJECTIVE_DIMS]
        base_dims = [d for d in active_dims if d in OBJECTIVE_DIMS]
        self.normalize_scores(results, dims=base_dims)

        w = self.weights
        for r in results:
            score = 0.0
            total_w = 0.0

            for key, weight in w.items():
                if key in OBJECTIVE_DIMS:
                    norm_val = r.get(f'{key}_norm', 0)
                    score += weight * (norm_val if norm_val is not None else 0)
                    total_w += weight
                elif key.endswith('_inv'):
                    # Inverse metrics: e.g., ppl_inv → 1/max(ppl, 0.1) normalized
                    base = key[:-4]  # remove '_inv'
                    if base in OBJECTIVE_DIMS:
                        raw = r.get(base)
                        if raw is not None:
                            # Clamp inverse to [0, 1] range
                            inv = 1.0 / max(raw, 0.1)
                            norm_val = min(inv, 1.0)
                            score += weight * norm_val
                            total_w += weight

            r['mo_composite_score'] = round(score / max(total_w, 1e-9), 4)
            r['mo_breakdown'] = {
                k: round(r.get(f'{k}_norm', 0), 3) if k in OBJECTIVE_DIMS
                else round(min(1.0 / max(r.get(k[:-4], 0.1), 0.1), 1.0), 3)
                for k in w if any(r.get(d) is not None for d in [k, k[:-4]])
            }

        results.sort(key=lambda x: x.get('mo_composite_score', 0), reverse=True)
        for i, r in enumerate(results):
            r['mo_rank'] = i + 1

        return results

    # ── Pareto Front (Non-Dominated Sorting) ──

    @staticmethod
    def compute_pareto_fronts(results: List[Dict],
                              dims: Optional[List[str]] = None,
                              epsilon: float = 0.0) -> List[Dict]:
        """Assign each design to a Pareto front using non-dominated sorting.

        Uses epsilon-dominance when epsilon > 0 (for noisy objectives).

        Front 0 = non-dominated (Pareto optimal).
        Front 1 = dominated only by Front 0.
        Higher fronts = successively more dominated.

        Args:
            results: list of dicts with objective values
            dims: dimensions to use for dominance (default: key AF2 + Rosetta dims)
            epsilon: epsilon-dominance tolerance (0 = strict Pareto)

        Returns:
            results with added 'pareto_front' (int) and 'crowding_distance' (float)
        """
        if dims is None:
            dims = ['iptm', 'plddt', 'dG', 'ppl']

        # Normalize scores first for consistent dominance comparison
        MultiObjectiveRanker.normalize_scores(results, dims=dims)

        n = len(results)
        if n == 0:
            return results

        # Build objective matrix: each row = design, each col = normalized objective
        obj_matrix = []
        for r in results:
            row = []
            for d in dims:
                direction = OBJECTIVE_DIMS.get(d, 'maximize')
                norm = r.get(f'{d}_norm', 0)
                # For dominance: want higher value = better for all dims
                # Normalized values already satisfy this (1.0 = best, 0.0 = worst)
                row.append(norm if norm is not None else 0.0)
            obj_matrix.append(row)

        m = len(dims)

        # Dominance counts: S[p] = set of designs dominated by p
        # n[p] = number of designs that dominate p
        domination_counts = [0] * n
        dominated_sets: List[Set[int]] = [set() for _ in range(n)]

        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                p_dominates_q = True
                q_dominates_p = True
                p_strictly_better = False
                q_strictly_better = False

                for d in range(m):
                    pv = obj_matrix[p][d]
                    qv = obj_matrix[q][d]

                    if pv < qv - epsilon:
                        p_dominates_q = False
                        q_strictly_better = True
                    if qv < pv - epsilon:
                        q_dominates_p = False
                        p_strictly_better = True

                if p_dominates_q and p_strictly_better:
                    dominated_sets[p].add(q)
                elif q_dominates_p and q_strictly_better:
                    domination_counts[p] += 1

        # Assign fronts
        fronts: List[List[int]] = []
        assigned = [False] * n
        current_front = []
        for i in range(n):
            if domination_counts[i] == 0:
                current_front.append(i)
                assigned[i] = True

        while current_front:
            fronts.append(current_front)
            next_front = []
            for p in current_front:
                for q in dominated_sets[p]:
                    if not assigned[q]:
                        domination_counts[q] -= 1
                        if domination_counts[q] == 0:
                            next_front.append(q)
                            assigned[q] = True
            current_front = next_front

        # Assign front indices to results
        for front_idx, front_members in enumerate(fronts):
            for member_idx in front_members:
                results[member_idx]['pareto_front'] = front_idx

        # Compute crowding distance within each front
        for front_idx, front_members in enumerate(fronts):
            if len(front_members) <= 2:
                for mi in front_members:
                    results[mi]['crowding_distance'] = float('inf')
                continue

            for mi in front_members:
                results[mi]['crowding_distance'] = 0.0

            for d in range(m):
                sorted_members = sorted(front_members,
                                        key=lambda i: obj_matrix[i][d])
                obj_range = (obj_matrix[sorted_members[-1]][d] -
                             obj_matrix[sorted_members[0]][d])
                if obj_range < 1e-9:
                    continue

                results[sorted_members[0]]['crowding_distance'] = float('inf')
                results[sorted_members[-1]]['crowding_distance'] = float('inf')

                for k in range(1, len(sorted_members) - 1):
                    prev_val = obj_matrix[sorted_members[k - 1]][d]
                    next_val = obj_matrix[sorted_members[k + 1]][d]
                    dist = (next_val - prev_val) / obj_range
                    results[sorted_members[k]]['crowding_distance'] += dist

        return results

    # ── Pareto Ranking ──

    @staticmethod
    def rank_by_pareto(results: List[Dict],
                       dims: Optional[List[str]] = None) -> List[Dict]:
        """Sort by Pareto front (ascending) then crowding distance (descending).

        Adds 'pareto_rank' key (1-based, best design = 1).
        """
        # Ensure Pareto fronts are computed
        if not any('pareto_front' in r for r in results):
            MultiObjectiveRanker.compute_pareto_fronts(results, dims=dims)

        # Sort: front ascending, crowding distance descending within same front
        results.sort(key=lambda x: (
            x.get('pareto_front', 999),
            -(x.get('crowding_distance', 0) or 0),
        ))

        for i, r in enumerate(results):
            r['pareto_rank'] = i + 1

        return results

    # ── Hypervolume (for convergence tracking) ──

    @staticmethod
    def compute_hypervolume(results: List[Dict],
                            dims: Optional[List[str]] = None,
                            ref_point: Optional[List[float]] = None) -> float:
        """Compute hypervolume of the first Pareto front.

        Uses normalized objective values. Reference point is the worst
        possible (0, 0, ..., 0) after normalization.

        For 2-3 dims, uses exact computation. For >3 dims, uses Monte Carlo.

        Args:
            results: list of dicts (needs pareto_front assigned)
            dims: objective dimensions (default: ['iptm', 'dG'])
            ref_point: reference point (default: origin = worst)

        Returns:
            hypervolume: float (higher = better Pareto front coverage)
        """
        if dims is None:
            dims = ['iptm', 'dG']

        # Get only front-0 designs
        front0 = [r for r in results if r.get('pareto_front') == 0]
        if not front0:
            return 0.0

        # Filter dims to only those with real values in front-0
        valid_dims = []
        for d in dims:
            has_real = any(r.get(d) is not None for r in front0)
            if has_real:
                valid_dims.append(d)
        if not valid_dims:
            valid_dims = ['iptm']  # fallback to single dim
        dims = valid_dims

        if ref_point is None:
            ref_point = [0.0] * len(dims)

        # Normalize if not already
        if not any(f'{dims[0]}_norm' in r for r in front0):
            MultiObjectiveRanker.normalize_scores(front0, dims=dims)

        points = []
        for r in front0:
            pt = [r.get(f'{d}_norm', 0) or 0 for d in dims]
            points.append(tuple(pt))

        if len(dims) == 1:
            return max(p[0] for p in points)

        if len(dims) == 2:
            # 2D exact: sort by x, compute contributed area
            points = sorted(points, key=lambda p: p[0], reverse=True)
            hv = 0.0
            prev_y = ref_point[1]
            for x, y in points:
                if y > prev_y:
                    hv += (x - ref_point[0]) * (y - prev_y)
                    prev_y = y
                hv = max(hv, x * y)
            # Recompute properly
            points_sorted = sorted(set(points), key=lambda p: p[0])
            hv = 0.0
            last_x = ref_point[0]
            for x, y in points_sorted:
                hv += (x - last_x) * y
                last_x = x
            return hv

        # For 3+ dims: use Monte Carlo sampling
        import random
        n_samples = 10000
        # Find bounding box
        max_vals = [max(p[i] for p in points) for i in range(len(dims))]
        hits = 0
        rng = random.Random(42)  # Fixed seed for reproducibility
        for _ in range(n_samples):
            sample = [rng.uniform(ref_point[i], max_vals[i]) for i in range(len(dims))]
            # Check if sample is dominated by any Pareto point
            dominated = False
            for pt in points:
                if all(pt[i] >= sample[i] for i in range(len(dims))):
                    dominated = True
                    break
            if dominated:
                hits += 1

        bbox_volume = 1.0
        for mv in max_vals:
            bbox_volume *= mv
        return bbox_volume * (hits / n_samples)


# ── Module-level convenience functions ──

def extract_pareto_front(results: List[Dict],
                         dims: Optional[List[str]] = None) -> List[Dict]:
    """Return only the first Pareto front (non-dominated designs)."""
    scored = MultiObjectiveRanker.compute_pareto_fronts(results, dims=dims)
    return [r for r in scored if r.get('pareto_front') == 0]


def format_mo_ranking_report(results: List[Dict], top_n: int = 10) -> str:
    """Format multi-objective ranking as a text report."""
    if not results:
        return "No results to display."

    lines = [
        "─" * 70,
        "  Multi-Objective Ranking Report",
        "─" * 70,
    ]

    # Summary stats
    front0 = [r for r in results if r.get('pareto_front') == 0]
    n_fronts = max((r.get('pareto_front', 0) for r in results), default=0) + 1

    lines.append(f"  Pareto fronts: {n_fronts}  |  Front 0 size: {len(front0)}")
    if front0:
        best_pf = front0[0]
        lines.append(f"  Best composite: {best_pf.get('mo_composite_score', '?'):.3f}")

    # Per-front summary
    for fi in range(min(n_fronts, 3)):
        f_members = [r for r in results if r.get('pareto_front') == fi]
        if not f_members:
            continue
        iptms = [r.get('iptm', 0) or 0 for r in f_members]
        dGs = [r.get('dG', 999) for r in f_members if r.get('dG') is not None]
        lines.append(f"  Front {fi}: {len(f_members)} designs  "
                     f"| ipTM=[{min(iptms):.2f}-{max(iptms):.2f}]"
                     + (f"  | dG=[{min(dGs):.1f}-{max(dGs):.1f}]" if dGs else ""))

    # Top designs table
    lines.append("")
    header = (f"{'Rank':<5} {'Pareto':<8} {'Composite':<10} {'ipTM':<8} "
              f"{'pLDDT':<8} {'dG':<8} {'PPL':<8} {'Sequence'}")
    lines.append(header)
    lines.append("-" * len(header))

    for r in results[:top_n]:
        pf = r.get('pareto_front', '?')
        comp = r.get('mo_composite_score', 0)
        iptm = r.get('iptm', 0) or 0
        plddt = r.get('plddt', 0) or 0
        dg = r.get('dG', 'N/A')
        ppl = r.get('ppl', 0) or 0
        seq = r.get('sequence', '')[:35]

        dg_str = f"{dg:.1f}" if isinstance(dg, (int, float)) else str(dg)
        lines.append(
            f"{r.get('mo_rank', r.get('pareto_rank', '?')):<5} "
            f"{pf:<8} {comp:<10.3f} {iptm:<8.3f} {plddt:<8.1f} "
            f"{dg_str:<8} {ppl:<8.1f} {seq}"
        )

    lines.append("─" * 70)
    return '\n'.join(lines)


def format_pareto_summary(results: List[Dict]) -> str:
    """One-line Pareto summary for progress display."""
    front0 = [r for r in results if r.get('pareto_front') == 0]
    if not front0:
        return "Pareto front empty"

    iptms = [r.get('iptm', 0) or 0 for r in front0]
    dGs = [r.get('dG', 999) for r in front0 if r.get('dG') is not None]
    return (f"Pareto-0: {len(front0)} designs, "
            f"ipTM∈[{min(iptms):.2f},{max(iptms):.2f}]"
            + (f", dG∈[{min(dGs):.1f},{max(dGs):.1f}]" if dGs else ""))
