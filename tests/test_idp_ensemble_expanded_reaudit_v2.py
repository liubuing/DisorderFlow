from scripts.reaudit_idp_ensemble_expanded_development_v2 import (
    resolve_antigen_chains,
    typed_antigen_chains,
)


def test_typed_antigen_chains_excludes_ions_and_haptens():
    component = {
        "antigen_chains": ["A", "B", "C"],
        "antigen_type": ["PROTEIN", "ION", "HAPTEN"],
    }
    assert typed_antigen_chains(component) == ["A"]


def test_antigen_chain_fallback_uses_observed_amino_acid_chain():
    class Chain(list):
        pass

    class Residue:
        resname = "ALA"

        def __contains__(self, key):
            return key == "CA"

        def get_resname(self):
            return self.resname

    component = {
        "antigen_chains": ["A", "B", "BBB"],
        "antigen_type": ["HAPTEN", "PEPTIDE"],
        "heavy_chain": "AAA",
        "light_chain": None,
    }
    chains, method = resolve_antigen_chains(
        component, {"AAA": Chain([Residue()]), "BBB": Chain([Residue()])}
    )
    assert chains == ["BBB"]
    assert method == "coordinate_observed_amino_acid_fallback"
