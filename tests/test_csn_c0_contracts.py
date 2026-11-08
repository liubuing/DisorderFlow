"""CSN C0/C2 contract tests.

Bound to M3 (infrastructure bugs corrupting scientific outputs) and N1/N2
(cross-campaign mixing, verdict upgrades). Pure-function tests are always
active; artifact tests activate once the preregistration/outputs are frozen.
No model-score field or author name may appear in any CSN artifact.
"""
from pathlib import Path
import json
import math
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.flex_a2_nmr_rmsf_v1 import kabsch as legacy_kabsch  # noqa: E402
from scripts.csn_c0_conformer_sharing_v1 import (  # noqa: E402
    CAL, MULTI, OVERLAP_MIN, POOL, RMSD_MIN_POINTS, VERDICTS,
    ca_rmsd, dihedral_class, family_groups, kabsch_row, resolve_cif, tier,
)

C0 = ROOT / 'results/csn_c0_v2'
QUEUE = ROOT / 'results/flex_j0_v1/preregistration.json'

FORBIDDEN_SUBSTRINGS = (
    'mpnn_score', 'affinity_value', 'kd_nm', 'model_score',
    'Haoyang', 'Tanyu', 'Yuchao', 'Jing Su', 'sj_12346',
)


# ---------------------------------------------------------------- pure functions

def test_tier_boundaries_are_preregistered_edges():
    assert tier(1.0) == 'same_bound_conformer'
    assert tier(1.0001) == 'related_conformers'
    assert tier(2.5) == 'related_conformers'
    assert tier(2.5001) == 'distinct_induced_fit'
    assert tier(0.0) == 'same_bound_conformer'


def test_dihedral_class_priority_and_terminals():
    assert dihedral_class(None, None) == 'terminal_na'
    assert dihedral_class(-139.0, 150.0) == 'extended'
    assert dihedral_class(-60.0, -40.0) == 'helical'
    assert dihedral_class(-60.0, 140.0) == 'ppii'
    assert dihedral_class(-82.0, 140.0) == 'extended'  # priority extended > ppii
    assert dihedral_class(0.0, 0.0) == 'other'


def test_kabsch_rmsd_roundtrip_and_sensitivity():
    rng = np.random.default_rng(20261108)
    pts = rng.normal(size=(9, 3)) * 5.0
    assert ca_rmsd([p for p in pts], [p for p in pts]) == pytest.approx(0.0, abs=1e-9)
    theta = 0.7
    rot = np.array([[math.cos(theta), -math.sin(theta), 0.0],
                    [math.sin(theta), math.cos(theta), 0.0],
                    [0.0, 0.0, 1.0]])
    rigid = pts @ rot.T + np.array([10.0, -3.0, 2.0])
    assert ca_rmsd([p for p in pts], [p for p in rigid]) == pytest.approx(0.0, abs=1e-8)
    moved = rigid.copy()
    moved[0] += np.array([1.0, 0.0, 0.0])
    one = ca_rmsd([p for p in pts], [p for p in moved])
    # Optimal superposition absorbs part of a single-point displacement by
    # tilting; the contract is 0 < rmsd < naive bound, and monotonic in size.
    assert 0.0 < one < math.sqrt(1.0 / 9.0)
    moved2 = rigid.copy()
    moved2[0] += np.array([2.0, 0.0, 0.0])
    assert ca_rmsd([p for p in pts], [p for p in moved2]) > one
    with pytest.raises(ValueError):
        ca_rmsd([[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [1, 1, 1]])  # < RMSD_MIN_POINTS


def test_corrected_kabsch_beats_legacy_on_random_rigid_motions():
    """Canary for the disclosed legacy-kabsch transpose bug (CSN campaign doc S6).

    The corrected row-convention implementation must align every perfect rigid
    motion to machine precision; the legacy implementation must NOT (it applies
    the transposed rotation). If the legacy module is ever repaired, update the
    disclosure and retire the second assertion deliberately.
    """
    rng = np.random.default_rng(20261107)
    legacy_worst = 0.0
    for _ in range(5):
        pts = rng.normal(size=(12, 3)) * 3.0
        q, _r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q * np.sign(np.linalg.det(q))
        rigid = pts @ q + rng.normal(size=3)
        assert ca_rmsd([p for p in pts], [p for p in rigid]) < 1e-6
        legacy_worst = max(legacy_worst, float(np.sqrt(
            ((legacy_kabsch(pts, rigid) - rigid) ** 2).sum(axis=1).mean())))
    assert legacy_worst > 1.0, 'legacy kabsch now aligns correctly; update disclosure'


def test_family_substring_closure():
    groups = family_groups(['KEDAFSKLKEKFMNEL', 'KEDAFSKLKEKFMN', 'EDAFSKLKEKFMN',
                            'TQAQEY', 'QAQEY', 'DYEPEA', 'GSVQIVYKPK'])
    assert groups['KEDAFSKLKEKFMNEL'] == ['EDAFSKLKEKFMN', 'KEDAFSKLKEKFMN',
                                          'KEDAFSKLKEKFMNEL']
    assert groups['TQAQEY'] == ['QAQEY', 'TQAQEY']
    assert groups['DYEPEA'] == ['DYEPEA']
    assert groups['GSVQIVYKPK'] == ['GSVQIVYKPK']


def test_resolution_order_hits_all_three_sources():
    assert resolve_cif('2x6m', 'A')[0][0] == 'sabdab2_pool'
    assert resolve_cif('8op0', 'A')[0][0] == 'multiscaffold'
    assert resolve_cif('6i2g', 'A')[0][0] == 'calibration_full_entry'
    assert resolve_cif('6zwk', 'B') == []          # known gap, fail-closed
    assert resolve_cif('5ivn', 'A') == []


def test_queue_universe_is_28_components_14_pdbs():
    comps = json.loads(QUEUE.read_text(encoding='utf-8'))['cohort']['components']
    assert len(comps) == 28
    assert len({c['pdb_id'] for c in comps}) == 14


def test_preregistered_constants_are_coherent():
    assert RMSD_MIN_POINTS == 4
    assert OVERLAP_MIN == 5
    assert set(VERDICTS) == {
        'conformer_sharing', 'related_conformers', 'distinct_induced_fit', 'mixed',
        'no_cross_entry_evidence', 'no_valid_coordinates', 'coordinate_starvation'}


# ------------------------------------------------- artifact contracts (post-freeze)

def _prereg():
    return json.loads((C0 / 'preregistration.json').read_text(encoding='utf-8'))


@pytest.mark.skipif(not (C0 / 'preregistration.json').exists(),
                    reason='prereg not frozen yet')
def test_prereg_frozen_content():
    prereg = _prereg()
    assert prereg['classification'] == 'development_descriptive_not_confirmatory_claim'
    assert prereg['tiers']['grounding'].startswith('T2.1 v2')
    assert prereg['stop_rules']['starvation']
    assert set(prereg['verdict_closed_set']) == set(VERDICTS)
    text = json.dumps(prereg)
    for bad in FORBIDDEN_SUBSTRINGS:
        assert bad not in text
    script_sha = prereg['provenance']['script_sha256']
    import hashlib
    current = hashlib.sha256(
        (ROOT / 'scripts/csn_c0_conformer_sharing_v1.py').read_bytes()).hexdigest()
    assert script_sha == current, 'script edited after prereg freeze (M4)'


@pytest.mark.skipif(not (C0 / 'summary.json').exists(),
                    reason='C0 not run yet')
def test_summary_contract():
    summary = json.loads((C0 / 'summary.json').read_text(encoding='utf-8'))
    assert summary['global_verdict'] in VERDICTS
    assert summary['classification'] == 'development_descriptive_not_confirmatory_claim'
    for fam in summary['families']:
        assert fam['verdict'] in VERDICTS
    text = json.dumps(summary)
    for bad in FORBIDDEN_SUBSTRINGS:
        assert bad not in text


# ------------------------------------------------------------------ C2 contracts

from scripts.csn_c2_ensemble_magnitude_v1 import (  # noqa: E402
    NEAR_RIGID_A, T1_LIKE_A, tier as c2_tier,
)

C2 = ROOT / 'results/csn_c2_v1'
C2_VERDICTS = {'marginalization_trivial_explanation',
               'magnitude_parity_preserved',
               'intermediate_requires_deeper_resolution'}


def test_c2_tier_boundaries():
    assert c2_tier(NEAR_RIGID_A - 0.001) == 'near_rigid'
    assert c2_tier(NEAR_RIGID_A) == 'modest'
    assert c2_tier(T1_LIKE_A - 0.001) == 'modest'
    assert c2_tier(T1_LIKE_A) == 't1_like'


@pytest.mark.skipif(not (C2 / 'protocol.json').exists(),
                    reason='C2 protocol not frozen yet')
def test_c2_protocol_contract():
    protocol = json.loads((C2 / 'protocol.json').read_text(encoding='utf-8'))
    assert protocol['classification'] == 'development_diagnostic_not_gate_revision'
    rules = protocol['tiers']['interpretation_closed_rules']
    assert set(rules.values()) == C2_VERDICTS
    assert any('read-only' in line for line in protocol['red_lines'])
    text = json.dumps(protocol)
    for bad in FORBIDDEN_SUBSTRINGS:
        assert bad not in text


@pytest.mark.skipif(not (C2 / 'summary.json').exists(),
                    reason='C2 not run yet')
def test_c2_summary_contract():
    summary = json.loads((C2 / 'summary.json').read_text(encoding='utf-8'))
    assert summary['classification'] == 'development_diagnostic_not_gate_revision'
    assert summary['verdict'] in C2_VERDICTS
    tiers = summary['tier_counts']
    assert tiers['near_rigid'] + tiers['modest'] + tiers['t1_like'] == summary['n_components']
    median = summary['median_spread_corrected_complex_frame']
    if summary['verdict'] == 'marginalization_trivial_explanation':
        assert median < NEAR_RIGID_A
    elif summary['verdict'] == 'magnitude_parity_preserved':
        assert median >= T1_LIKE_A
    else:
        assert NEAR_RIGID_A <= median < T1_LIKE_A
    text = json.dumps(summary)
    for bad in FORBIDDEN_SUBSTRINGS:
        assert bad not in text
