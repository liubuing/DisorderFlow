import csv
import hashlib
import json
from pathlib import Path

from Bio.Seq import Seq


ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = (
    ROOT
    / "results/v5_1_candidates/abeta42_biological_constructs/expression_package"
)


def load_package():
    return json.loads((PACKAGE_DIR / "expression_package.json").read_text(encoding="utf-8"))


def test_expression_package_has_resolved_cloning_contract():
    package = load_package()
    contract = package["expression_contract"]

    assert package["status"] == "order_ready_for_pcDNA3_4_topo_ta"
    assert package["required_manual_decisions"] == []
    assert contract["vector"] == "pcDNA3.4-TOPO TA"
    assert contract["vector_catalog"] == "Thermo Fisher A14697"
    assert contract["cloning_flanks"].startswith("none encoded")
    assert contract["signal_peptide_policy"].startswith("retain")


def test_order_inserts_translate_and_match_checksums():
    package = load_package()

    for chain in package["chains"].values():
        insert = chain["expression_insert"]
        assert insert.startswith("GCCACCATG")
        assert insert.endswith("TGA")
        assert str(Seq(chain["cds"]).translate()) == chain["precursor_protein"]
        assert hashlib.sha256(insert.encode("ascii")).hexdigest().upper() == chain["insert_sha256"]


def test_order_manifest_matches_packaged_chains():
    package = load_package()
    with (PACKAGE_DIR / "order_manifest.csv").open(newline="", encoding="utf-8") as handle:
        manifest = {row["order_id"]: row for row in csv.DictReader(handle)}

    assert manifest.keys() == package["chains"].keys()
    for chain_id, chain in package["chains"].items():
        row = manifest[chain_id]
        assert row["insert_sha256"] == chain["insert_sha256"]
        assert int(row["insert_length_bp"]) == len(chain["expression_insert"])
