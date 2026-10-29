"""IBEC E2 IdP 证据审计的契约测试：守护区间指标的重叠区域合并、DisProt 精确注释获得实验层级，
以及预测/构象代理来源永不升级为实验证据——保证 IdP 分类声明与证据强度严格对应。"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# 辅助函数：按路径动态加载审计脚本模块。
def load_script():
    path = ROOT / "scripts" / "audit_ibec_e2_idp_evidence.py"
    spec = importlib.util.spec_from_file_location("audit_ibec_e2_idp_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 辅助函数：构造分类阈值与来源标签语义配置。
def config():
    return {
        "classification": {
            "fully_disordered_minimum_fraction": 0.7,
            "partially_disordered_minimum_fraction": 0.3,
            "idr_minimum_length": 20,
            "source_label_semantics": {
                "ebi_mobidb_lite": "prediction_only_disorder",
                "af2_plddt_fallback": "conformational_proxy_only",
                "folded": "no_disorder_evidence",
            },
        }
    }


# 区间合并：重叠区间 (1,10) 与 (5,20) 合并为 20 个无序残基、最长连续区 16——
# 防止重叠计数导致无序比例被重复放大。
def test_interval_metrics_merge_overlapping_regions():
    mod = load_script()
    metrics = mod.interval_metrics([(1, 10), (5, 20)], 100)
    assert metrics["disorder_residues"] == 20
    assert metrics["disorder_fraction"] == 0.2
    assert metrics["longest_disorder_region"] == 16


# 实验层级：DisProt 精确结构态注释必须标注为 experimentally_curated_fully_disordered，
# 防止实验注释与预测混用导致证据层级虚高或丢失。
def test_exact_disprot_regions_receive_experimental_tiers():
    mod = load_script()
    source = {"sequence": "A" * 100, "disorder_source": "folded"}
    entry = {"disprot_id": "DP1", "acc": "P1", "regions": [{
        "term_namespace": "Structural state", "term_name": "disorder",
        "start": 1, "end": 75,
    }]}
    result = mod.classify(config(), source, [entry])
    assert result["evidence_tier"] == "experimentally_curated_fully_disordered"


# 层级天花板：MobiDB-lite 只能是 prediction_only、AF2 pLDDT 回退只能是 conformational_proxy，
# 两者都不得被升级为实验证据——守护分类声明的证据诚信。
def test_prediction_and_proxy_never_upgrade_to_experimental():
    mod = load_script()
    predicted = mod.classify(
        config(), {"sequence": "A" * 50, "disorder_source": "ebi_mobidb_lite"}, []
    )
    proxy = mod.classify(
        config(), {"sequence": "A" * 50, "disorder_source": "af2_plddt_fallback"}, []
    )
    assert predicted["evidence_tier"] == "prediction_only_disorder"
    assert proxy["evidence_tier"] == "conformational_proxy_only"
