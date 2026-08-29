from scripts.materialize_idp_ensemble_expanded_development_v2 import residue_count


class Residue:
    def __init__(self, has_ca=True):
        self.resname = "ALA"
        self.has_ca = has_ca

    def __contains__(self, key):
        return key == "CA" and self.has_ca

    def get_resname(self):
        return self.resname


def test_residue_count_requires_observed_ca():
    assert residue_count([Residue(), Residue(False)]) == 1
