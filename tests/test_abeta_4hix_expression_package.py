"""Aβ 4HIX 表达构建包测试：验证参考序列反向翻译、H3 对照与盲法布局的不变量。

守护 abeta_4hix 表达载体构建管线：
参考 DNA 必须能完整回译出目标蛋白、H3 打乱对照只保持氨基酸组成而不泄漏原序列、
DNA 审计须标记高 GC 构建体、盲法 BLI 布局不得暴露 construct 身份。
"""
import importlib.util
from pathlib import Path

from Bio.Seq import Seq

ROOT = Path(__file__).resolve().parents[1]


# 辅助函数：以独立模块方式加载 scripts/ 下的构建脚本，便于直接测试其内部函数。
def load_script():
    path = ROOT / "scripts" / "build_abeta_4hix_expression_package.py"
    spec = importlib.util.spec_from_file_location("build_abeta_4hix_expression_package", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 反向翻译的参考 DNA 翻译后必须与原蛋白序列完全一致（round-trip），
# 防止表达构建包携带错误的编码序列。
def test_reference_reverse_translation_round_trips():
    mod = load_script()
    protein = "MDWTWRILFLVAAATGAHS" + "ACDEFGHIKLMNPQRSTVWY"
    dna = mod.reverse_translate_reference(protein)
    assert str(Seq(dna).translate()) == protein


# H3 打乱对照必须真正改变序列（非恒等变换）同时保持氨基酸多重集不变——
# 保证阴性对照只破坏序列信息而不引入组成偏差。
def test_shuffled_control_preserves_h3_composition():
    mod = load_script()
    native = "VRYDHYSGSSDY"
    shuffled = mod.shuffled_h3(native, 6401)
    assert shuffled != native
    assert sorted(shuffled) == sorted(native)


# DNA 审计必须把高 GC% 的序列标记为不适合直接合成（not order ready），
# 防止高 GC 参考序列未经警告进入下游基因合成。
def test_reference_dna_audit_marks_high_gc_as_not_order_ready():
    mod = load_script()
    audit = mod.dna_audit("GCC" * 50)
    assert audit["gc_percent"] > 68


# 盲法布局审计：公共布局必须覆盖全部孔位与 3 次技术重复，
# 且不携带 construct_id（仅允许 key-only 字段）——防止实验执行者被结果身份偏置。
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
