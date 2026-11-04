"""ESEL E0 contract tests.

Bound to two incident patterns (M3): infrastructure bugs that once silently
corrupted scientific outputs, and constraint drift on "no model scores in
candidate selection" (DECISION_20261101 constraint #3). The matrix is
epitope-level; if any model-score field or author name leaks in, these tests
fail the build before the freeze is trusted.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.esel_e0_epitope_selection_v1 import (  # noqa: E402
    ANCHOR_ACCESSION,
    ANCHOR_PEPTIDE,
    OUT,
    build_variants,
    core_variant,
    fail,
)

PREREG = ROOT / 'results/flex_j0_v1/preregistration.json'
CAMPAIGN = ROOT / 'docs/ESEL_CAMPAIGN_V1.md'

FORBIDDEN_SUBSTRINGS = (
    'mpnn_score', 'affinity_value', 'kd_nm', 'model_score',
    'Haoyang', 'Tanyu', 'Yuchao', 'Jing Su', 'sj_12346',
)


def _matrix():
    return json.loads(
        (OUT / 'epitope_presentation_matrix.json').read_text(encoding='utf-8'))


def _protocol():
    return json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))


def test_cohort_reconstruction_matches_campaign_cascade():
    """28 cohort components over 14 unique PDBs (campaign doc section 7 cascade)."""
    cohort = json.loads(PREREG.read_text(encoding='utf-8'))['cohort']
    pdbs = {c['pdb_id'] for c in cohort['components']}
    assert len(cohort['components']) == 28
    assert len(pdbs) == 14
    matrix = _matrix()
    cohort_rows = [r for r in matrix['rows'] if r['source'] == 'flex_j0_cohort']
    assert {r['source_pdb'] for r in cohort_rows} == pdbs
    assert {c['instance'] for c in cohort['components']} == {
        r['instance'] for r in cohort_rows}


def test_anchor_row_present_exactly_once_with_real_counts():
    matrix = _matrix()
    anchors = [r for r in matrix['rows'] if r['parent_peptide'] == ANCHOR_PEPTIDE]
    assert len(anchors) == 1
    precedent = anchors[0]['iedb_precedent']
    assert precedent['status'] == 'anchored_with_counts'
    assert precedent['accession'] == ANCHOR_ACCESSION
    identity = precedent['identity']
    assert 'alpha-synuclein' not in json.dumps(identity).lower(), (
        'the P05877 head_antigens_note mislabel must not leak into the anchor row')
    assert precedent['n_own_negative_sequences'] > 0


def test_variants_are_mechanical_substrings_with_single_biotin():
    cohort = json.loads(PREREG.read_text(encoding='utf-8'))['cohort']
    parents = {c['peptide_sequence'] for c in cohort['components']}
    parents.add(ANCHOR_PEPTIDE)
    for parent in parents:
        core = core_variant(parent)
        assert core in parent, 'core must be a contiguous substring of the parent'
        variants = build_variants(parent, 'X')
        types = [v['variant_type'] for v in variants]
        assert len(types) == len(set(types))
        biotin = [v for v in variants if v['variant_type'].startswith('biotin_')]
        assert len(biotin) == 2, 'exactly N-terminal and C-terminal biotin options'
        for v in variants:
            assert v['sequence'] in parent or v['variant_type'] == 'core_15'


def test_matrix_is_free_of_model_scores_and_author_names():
    text = json.dumps(_matrix(), ensure_ascii=False)
    for forbidden in FORBIDDEN_SUBSTRINGS:
        assert forbidden.lower() not in text.lower(), (
            f'forbidden token leaked into matrix: {forbidden}')
    assert 'no model affinity scores participate' in _protocol()['selection_basis']


def test_fail_closed_guards_raise():
    for bad_call in (lambda: fail('probe'), lambda: build_variants('', 'X')):
        try:
            bad_call()
        except RuntimeError:
            pass
        else:
            raise AssertionError('fail-closed guard did not raise RuntimeError')


def test_frozen_gates_all_pass_and_stop_rule_is_honest():
    protocol = _protocol()
    gates = protocol['gates']
    assert gates['universe_size']['pass']
    assert gates['anchor_present']['pass']
    assert gates['presentation_complete']['pass']
    assert gates['no_model_scores']['pass']
    stop = protocol['stop_rule_evaluation']
    assert stop['outcome'] == 'escalate_epitope_starvation_risk'
    assert stop['anchored_row_count'] == 1
    assert 'E0b' in stop['escalation_note']
