from disorderflow.utils.protein.constants import Fragment
from modules.bfn_loader import build_region_batch, parse_region_spec


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


def test_frozen_4hix_h3_mask_is_twelve_one_based_positions():
    assert parse_region_spec("H:96-107") == {"H": list(range(95, 107))}
