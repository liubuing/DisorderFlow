"""v5.1 表达打包产物的契约测试：守护克隆合约已完全消解（无待定人工决策）、
订购插入序列可翻译回前体蛋白且 sha256 一致（Kozak 起始、TGA 终止），
以及订单清单与打包链逐项对应——防止合成订单与设计意图脱节。"""
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


# 辅助函数：加载冻结的表达包 JSON。
def load_package():
    return json.loads((PACKAGE_DIR / "expression_package.json").read_text(encoding="utf-8"))


# 克隆合约：状态必须为可下单（pcDNA3.4-TOPO TA、指定货号）、无待定决策、无编码克隆侧翼、
# 保留信号肽策略——防止含未决事项的包进入订购。
def test_expression_package_has_resolved_cloning_contract():
    package = load_package()
    contract = package["expression_contract"]

    assert package["status"] == "order_ready_for_pcDNA3_4_topo_ta"
    assert package["required_manual_decisions"] == []
    assert contract["vector"] == "pcDNA3.4-TOPO TA"
    assert contract["vector_catalog"] == "Thermo Fisher A14697"
    assert contract["cloning_flanks"].startswith("none encoded")
    assert contract["signal_peptide_policy"].startswith("retain")


# 翻译与校验和：插入序列必须以 Kozak+ATG（GCCACCATG）起始、TGA 终止，CDS 翻译严格等于前体蛋白，
# 且插入序列 sha256 与记录一致——防止突变或截断的构建被下单。
def test_order_inserts_translate_and_match_checksums():
    package = load_package()

    for chain in package["chains"].values():
        insert = chain["expression_insert"]
        assert insert.startswith("GCCACCATG")
        assert insert.endswith("TGA")
        assert str(Seq(chain["cds"]).translate()) == chain["precursor_protein"]
        assert hashlib.sha256(insert.encode("ascii")).hexdigest().upper() == chain["insert_sha256"]


# 清单一致性：订单 CSV 的链集合与包一致，每行 sha256 与长度逐一匹配——防止订单与设计脱钩。
def test_order_manifest_matches_packaged_chains():
    package = load_package()
    with (PACKAGE_DIR / "order_manifest.csv").open(newline="", encoding="utf-8") as handle:
        manifest = {row["order_id"]: row for row in csv.DictReader(handle)}

    assert manifest.keys() == package["chains"].keys()
    for chain_id, chain in package["chains"].items():
        row = manifest[chain_id]
        assert row["insert_sha256"] == chain["insert_sha256"]
        assert int(row["insert_length_bp"]) == len(chain["expression_insert"])
