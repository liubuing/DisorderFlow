from scripts.prepare_abbibench_structural_audit import resolve_path
from scripts.audit_abbibench_short_sequences import ungapped_hit


def test_case_correction_is_explicit_and_ambiguous_paths_are_rejected():
    assert resolve_path('./data/A.pdb', ['A.pdb']) == ('A.pdb', 'exact')
    assert resolve_path('./data/a.pdb', ['A.pdb']) == ('A.pdb', 'unique_case_correction')
    assert resolve_path('a.PDB', ['A.pdb', 'a.pdb']) == (None, 'missing_or_ambiguous')
    assert resolve_path('absent.csv', ['A.pdb']) == (None, 'missing_or_ambiguous')


def test_ungapped_screen_uses_both_coverages_and_terminal_offsets():
    hit = ungapped_hit('ACDEF', 'CDEF', .9)
    assert hit['query_coverage'] == .8
    assert hit['identity'] == 1
    assert ungapped_hit('ACDEFGHIKL', 'CDEF', .9) is None


def test_short_sequence_threshold_includes_boundary_but_not_lower_identity():
    assert ungapped_hit('AAAA', 'AACC', .5) is not None
    assert ungapped_hit('AAAA', 'ACCC', .5) is None
    assert ungapped_hit('', 'AAAA', .5) is None
