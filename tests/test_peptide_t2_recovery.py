import pytest

from modules.peptide_t2_recovery import recovery_metrics, split_contact_pairs


def test_contact_split_is_deterministic_and_disjoint():
    contacts = {(('H', str(i)), ('P', str(i))) for i in range(6)}
    supplied, held_out = split_contact_pairs(contacts, "target")
    assert supplied.isdisjoint(held_out)
    assert supplied | held_out == contacts
    assert split_contact_pairs(contacts, "target") == (supplied, held_out)


def test_recovery_metrics_have_positive_direction():
    observed = recovery_metrics(2.5, 1.5, 0.2, 0.6)
    assert observed["rmsd_recovery_angstrom"] == 1.0
    assert observed["held_out_contact_recovery"] == pytest.approx(0.4)
    assert observed["rmsd_improved"] is True
