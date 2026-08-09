from disorderflow.utils.protein.constants import Fragment
from modules.bfn_loader import build_region_batch


def test_generic_complex_assigns_explicit_antigen_chain():
    batch = build_region_batch(
        "data/anti_abeta_refs/4HIX.pdb",
        "H:96-107",
        context_chains=["L", "A"],
        antigen_chains=["A"],
        device="cpu",
    )
    antigen = batch["fragment_type"] == int(Fragment.Antigen)
    generated = batch["generate_flag"].bool()
    assert int(antigen.sum()) == 6
    assert int(generated.sum()) == 12
    assert not (antigen & generated).any()
