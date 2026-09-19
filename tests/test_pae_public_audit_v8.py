from scripts.audit_pae_public_sequences_v8 import ungapped
from scripts.audit_pae_public_sources_v8 import normalize


def test_pdb_id_dedup_preserves_case_mapping():
    assert normalize('pdb_00006HHD') == normalize('6HHD_2') == '6hhd'


def test_short_alignment_requires_both_coverages():
    assert ungapped('ACDEFGHIKL', 'ACDEFGHIKL', .5)
    assert not ungapped('ACDEF', 'ACDEFGHIKLMNPQRSTVWY', .5)
    assert not ungapped('AAAAA', 'CCCCC', .3)
    assert not ungapped('', 'ACDEF', .3)


def test_terminal_offset_is_considered():
    assert ungapped('ACDEFGHIKL', 'XXACDEFGHIK', .9)
