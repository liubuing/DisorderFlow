"""候选界面置信度数据集构建（v1）的契约测试：守护各家族合约中候选位点的 CDR 区间定义，
以及冻结 bundle 发现时训练/验证家族与来源身份的不相交性。"""
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


# 辅助函数：按路径动态加载构建脚本模块（脚本不在包内）。
def load_script():
    path = ROOT / "scripts/build_candidate_interface_confidence_v1.py"
    spec = importlib.util.spec_from_file_location(
        "build_candidate_interface_confidence_v1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 家族合约：抗原链指定正确，候选位点必须恰好落在 Kabat CDR 区间（H1/H2/H3），且各链非空——
# 防止可变位定义漂移到框架区导致训练信号失效。
def test_variable_positions_are_shared_scaffold_masks():
    mod = load_script()
    constructs, manifest = mod.load_constructs()
    contracts = mod.family_contracts(constructs, manifest)
    assert contracts["8B9V"]["antigen_chain"] == "C"
    assert contracts["5MP3"]["antigen_chain"] == "C"
    assert contracts["3STB"]["candidate_positions"]["A"] == (
        list(range(26, 34)) + list(range(51, 59)) + list(range(97, 114)))
    assert all(contracts["8B9V"]["candidate_positions"].values())
    assert all(contracts["5MP3"]["candidate_positions"].values())


# 拆分隔离：训练家族（8B9V/3STB）与验证家族（5MP3）按支架划分，且 source_identity 严格不相交，
# 防止同源结构跨拆分泄漏造成验证虚高。
def test_frozen_bundle_discovery_is_scaffold_disjoint():
    mod = load_script()
    constructs, _ = mod.load_constructs()
    train = mod.discover_bundles(mod.TRAIN_ROOTS, constructs)
    validation = mod.discover_bundles(mod.VAL_ROOTS, constructs)
    assert {row["family"] for row in train} == {"8B9V", "3STB"}
    assert {row["family"] for row in validation} == {"5MP3"}
    assert {row["source_identity"] for row in train}.isdisjoint(
        {row["source_identity"] for row in validation})
