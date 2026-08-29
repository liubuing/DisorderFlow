import importlib.util
from pathlib import Path

from Bio.Seq import Seq

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "build_abeta_4hix_expression_package.py"
    spec = importlib.util.spec_from_file_location("build_abeta_4hix_expression_package", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_reverse_translation_round_trips():
    mod = load_script()
    protein = "MDWTWRILFLVAAATGAHS" + "ACDEFGHIKLMNPQRSTVWY"
    dna = mod.reverse_translate_reference(protein)
    assert str(Seq(dna).translate()) == protein


def test_shuffled_control_preserves_h3_composition():
    mod = load_script()
    native = "VRYDHYSGSSDY"
    shuffled = mod.shuffled_h3(native, 6401)
    assert shuffled != native
    assert sorted(shuffled) == sorted(native)


def test_reference_dna_audit_marks_high_gc_as_not_order_ready():
    mod = load_script()
    audit = mod.dna_audit("GCC" * 50)
    assert audit["gc_percent"] > 68


def test_blinded_layout_is_complete_and_has_no_construct_ids():
    mod = load_script()
    config = {
        "construct": {
            "external_negative_control": "external",
            "process_blank": "mock",
        },
        "bli": {
            "blind_seed": 1,
            "technical_replicates": 3,
            "analytes": [{"id": "target", "role": "positive"}],
        },
    }
    constructs = [
        {"construct_id": "candidate", "role": "candidate"},
        {"construct_id": "native", "role": "positive_control"},
    ]
    blind = mod.assign_blind_ids(config, constructs)
    public, keyed = mod.build_bli_layout(config, constructs, blind)
    assert len(public) == len(keyed) == 12
    assert all("construct_id_key_only" not in row for row in public)
    assert {row["technical_replicate"] for row in public} == {1, 2, 3}
