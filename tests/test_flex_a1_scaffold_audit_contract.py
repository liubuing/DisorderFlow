"""FLEX-BINDER Phase A3 contract tests.

Two contracts that previously failed silently and cost months:

1. AF2 multimer chain separation (M3 bug class): ``build_multimer_features``
   must create one chain per ':'-separated antibody segment. The original
   incident concatenated VH+VL, collapsing two chains into one and
   underestimating ipTM by 4.5x.

2. Nonstandard-residue counting (the A1 false-negative incident): 3-letter
   residue names must be converted before comparison against the 1-letter
   standard set; comparing 3-letter codes directly marks every residue
   nonstandard, empties the numbering pool and yields a false all-negative
   audit.

Both tests run on CPU without JAX.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.af2_jax_runner import build_multimer_features


def _chain_sizes(asym_id):
    values = sorted(set(np.asarray(asym_id).tolist()))
    return [int((np.asarray(asym_id) == v).sum()) for v in values]


def test_multimer_features_splits_antibody_chains_on_colon():
    features = build_multimer_features('AAAA:GGGG', 'TTTT')
    assert _chain_sizes(features['asym_id']) == [4, 4, 4]


def test_multimer_features_concatenated_sequence_stays_one_antibody_chain():
    # The incident mode: a concatenated VH+VL string yields a single antibody
    # chain. The test pins the difference so a future refactor cannot silently
    # lose the separator semantics.
    features = build_multimer_features('AAAAGGGG', 'TTTT')
    assert _chain_sizes(features['asym_id']) == [8, 4]


def test_multimer_features_chain_count_tracks_segment_count():
    segments = 'A:C:E:G'
    features = build_multimer_features(segments, 'PP')
    assert len(set(np.asarray(features['asym_id']).tolist())) == 5


class _FakeResidue:
    def __init__(self, resname, hetflag=' '):
        self.resname = resname
        self.id = (hetflag, 1, ' ')


def test_nonstandard_count_converts_three_letter_to_one_letter():
    from scripts.flex_a1_scaffold_audit_v1 import count_nonstandard
    residues = [_FakeResidue('ALA'), _FakeResidue('MSE'), _FakeResidue('SEP')]
    # MSE is accepted (converted to M); SEP is genuinely nonstandard.
    assert count_nonstandard(residues) == 1


def test_nonstandard_count_ignores_hetero_records():
    from scripts.flex_a1_scaffold_audit_v1 import count_nonstandard
    residues = [_FakeResidue('ALA'), _FakeResidue('HOH', hetflag='W'),
                _FakeResidue('GOL', hetflag='H_M')]
    assert count_nonstandard(residues) == 0


def test_nonstandard_count_rejects_three_letter_comparison_bug():
    # Direct evidence of the bug mode: without seq1 conversion every standard
    # residue would be counted nonstandard. The contract requires exactly one
    # nonstandard residue here (SEP), not four.
    from scripts.flex_a1_scaffold_audit_v1 import count_nonstandard
    residues = [_FakeResidue(res) for res in ('ALA', 'GLY', 'VAL', 'SEP')]
    assert count_nonstandard(residues) == 1
