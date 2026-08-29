from scripts.audit_idp_ensemble_development_extension import contact_metrics


def test_contact_metrics_counts_contacting_h3_positions():
    class Atom:
        element = "C"

        def __init__(self, coord):
            self.coord = coord

    class Residue:
        def __init__(self, coord):
            self.atoms = [Atom(coord)]

        def __iter__(self):
            return iter(self.atoms)

    heavy = [{"residue": Residue([0.0, 0.0, 0.0])}]
    antigen = [{"residue": Residue([0.0, 0.0, 4.0])}]
    result = contact_metrics(heavy, antigen, [0], 4.5)
    assert result["contacting_h3_positions"] == 1
    assert result["h3_peptide_residue_contact_pairs"] == 1
