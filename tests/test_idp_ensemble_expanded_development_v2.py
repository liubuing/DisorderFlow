from scripts.build_idp_ensemble_expanded_development_v2 import (
    classify_target,
    normalize_pdb,
)


def test_expanded_target_classification_and_pdb_normalization():
    ontology = {
        "tau": {"patterns": ["microtubule-associated protein tau"]},
        "huntingtin": {"patterns": ["huntingtin"]},
    }
    assert classify_target("Tau peptide", ontology) is None
    assert classify_target("microtubule-associated protein tau", ontology) == "tau"
    assert classify_target("huntingtin peptide", ontology) == "huntingtin"
    assert normalize_pdb("pdb_00004TQE") == "4tqe"
