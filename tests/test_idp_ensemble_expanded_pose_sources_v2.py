from scripts.audit_idp_ensemble_expanded_pose_sources_v2 import chain_sequence


def test_chain_sequence_reads_standard_ca_residues():
    class Residue:
        resname = "ALA"

        def __contains__(self, key):
            return key == "CA"

        def get_resname(self):
            return self.resname

    assert chain_sequence([Residue(), Residue()]) == "AA"
