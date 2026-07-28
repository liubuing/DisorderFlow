#!/usr/bin/env python3
"""P1-2: IDP Functional Epitope Selector (replaces disorder<0.3 criterion).

IDP_DESIGN_SCHEME_V2 §P1-2:
  "Replace find_ordered_segments disorder<0.3 with composite:
   aggregation_propensity × druggability × disorder_tolerance"

The old approach selected epitopes by "lowest disorder" → this hit 16-24
(KLVFFAED) by coincidence because it's the aggregation core. The new
approach selects by biological function and therapeutic relevance.

Scoring:
  composite = w_agg * aggregation_propensity
            + w_drug * druggability
            + w_dis * disorder_tolerance(disorder_score)

Where disorder_tolerance is HIGH for regions that can be induced to fold
upon binding (moderate disorder, not extreme), and LOW for both fully
ordered (no IDP benefit) and fully disordered (can't be stabilized).

Usage:
    from modules.idp_epitope_selector import select_epitopes
    epitopes = select_epitopes('abeta42', top_k=3)
    # → [{id, sequence, composite, aggregation, druggability, disorder}]
"""
import os, yaml
from typing import List, Dict, Optional

EPITOPE_LIBRARY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     'configs', 'idp', 'abeta_epitopes.yml')


def _load_library() -> Dict:
    """Load the functional epitope library."""
    with open(EPITOPE_LIBRARY_PATH, encoding='utf-8') as f:
        return yaml.safe_load(f)


def disorder_tolerance(disorder_score: float) -> float:
    """Map disorder score to "bindability" — how likely the region can be
    induced to fold upon antibody binding.

    Returns:
        1.0 for moderately disordered (0.3-0.7) — ideal: can be stabilized
        0.5 for mildly disordered (0.1-0.3) — already somewhat ordered
        0.2 for fully disordered (>0.9) — too unstable to stabilize
        0.3 for fully ordered (<0.1) — no IDP benefit, use standard design
    """
    if disorder_score < 0.1:
        return 0.3   # fully ordered — use standard pipeline
    elif disorder_score < 0.3:
        return 0.5   # weakly disordered
    elif disorder_score <= 0.7:
        return 1.0   # optimal: can be induced to fold
    elif disorder_score <= 0.9:
        return 0.4   # strongly disordered — hard to stabilize
    else:
        return 0.1   # extreme IDP — avoid


def score_epitope(epitope: Dict, weights: Optional[Dict] = None) -> Dict:
    """Compute composite score for a single epitope.

    Args:
        epitope: dict from abeta_epitopes.yml
        weights: override selection weights

    Returns:
        epitope dict with added 'composite' and component scores
    """
    lib = _load_library()
    w = weights or lib.get('selection_weights', {})
    w_agg = w.get('aggregation_propensity', 0.40)
    w_drug = w.get('druggability', 0.35)
    w_dis = w.get('disorder_tolerance', 0.25)

    agg = epitope.get('aggregation_propensity', 0.5)
    drug = epitope.get('druggability', 0.5)
    # Disorder tolerance: moderate disorder is IDEAL for IDP targeting
    # (can be stabilized upon binding). We estimate from the epitope context.
    # For known epitopes, this comes from the library.
    dis_tol = epitope.get('disorder_tolerance', 0.7)

    composite = w_agg * agg + w_drug * drug + w_dis * dis_tol

    return {
        **epitope,
        'composite': round(composite, 4),
        'agg_score': agg,
        'druggability_score': drug,
        'disorder_tolerance_score': dis_tol,
    }


def select_epitopes(target: str = 'abeta42', top_k: int = 3,
                    therapeutic_goal: Optional[str] = None,
                    weights: Optional[Dict] = None) -> List[Dict]:
    """Select top-K epitopes for a given IDP target.

    Args:
        target: 'abeta42' (currently the only supported target)
        top_k: number of epitopes to return
        therapeutic_goal: optional filter ('plaque_clearance' / 'oligomer_neutralise')
        weights: override selection weights

    Returns:
        sorted list of epitope dicts with composite scores
    """
    lib = _load_library()
    epitopes = lib.get('epitopes', [])

    scored = [score_epitope(e, weights) for e in epitopes]

    # Filter by therapeutic goal if specified
    if therapeutic_goal:
        scored = [e for e in scored if e.get('therapeutic_target') == therapeutic_goal]

    # Sort by composite descending
    scored.sort(key=lambda e: e['composite'], reverse=True)
    return scored[:top_k]


def compare_to_old_criterion(disorder_scores: Optional[Dict[int, float]] = None):
    """Compare new functional selection vs old disorder<0.3 approach.

    Without real per-residue disorder scores, uses the epitope library's
    known annotations. Prints a comparison table.
    """
    print("=" * 70)
    print("Epitope Selection: OLD (disorder<0.3) vs NEW (functional composite)")
    print("=" * 70)

    lib = _load_library()
    epitopes = lib.get('epitopes', [])

    # Simulate old criterion: pick epitope with lowest disorder estimate
    # (In practice, 16-24 scores lowest because it's the aggregation core,
    #  but that's coincidence — the criterion is wrong)
    print(f"\n{'Epitope':<25} {'Agg':>6} {'Drug':>6} {'DisTol':>6} {'Composite':>10} {'Old?':>6}")
    print("-" * 65)

    scored = [score_epitope(e) for e in epitopes]
    scored.sort(key=lambda e: e['composite'], reverse=True)

    # Old criterion would pick by lowest aggregation (most "ordered" region)
    old_pick = min(epitopes, key=lambda e: e.get('aggregation_propensity', 1.0))

    for e in scored:
        old_flag = 'OLD' if e['id'] == old_pick['id'] else ''
        print(f"{e['id']:<25} {e['agg_score']:>6.2f} {e['druggability_score']:>6.2f} "
              f"{e['disorder_tolerance_score']:>6.2f} {e['composite']:>10.4f} {old_flag:>6}")

    print(f"\nOLD criterion picked: {old_pick['id']} (lowest aggregation → most 'ordered')")
    print(f"  → {old_pick['notes']}")
    best = scored[0]
    print(f"NEW criterion picks: {best['id']} (best composite)")
    print(f"  → {best['notes']}")

    if old_pick['id'] == best['id']:
        print("\nResult: OLD and NEW agree — but for different REASONS.")
        print("  OLD: picked by accident (aggregation core happens to be 'ordered')")
        print("  NEW: picked by design (aggregation core IS the right target)")
    else:
        print(f"\nResult: DIFFERENT — NEW criterion changes epitope selection.")


if __name__ == '__main__':
    compare_to_old_criterion()
