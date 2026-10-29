"""Aβ 4HIX AF2 门槛管线的契约测试：守护候选实体清单（含对照）的构成、H3 替换不越出框架区，
以及多样性筛选不得用近似重复序列回填短名单。"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# 辅助函数：按文件路径动态加载待测脚本模块（脚本非包成员，只能以 importlib 方式导入）。
def load_script():
    path = ROOT / "scripts" / "run_abeta_4hix_af2.py"
    spec = importlib.util.spec_from_file_location("run_abeta_4hix_af2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 实体清单契约：24 条候选 + 3 条指定对照（native / proteinmpnn_best / composition_shuffle）共 27 条，
# 且所有 H3 序列长度固定为 12，防止对照缺失或序列规格漂移导致基准不可比。
def test_entities_include_frozen_shortlist_and_required_controls():
    import yaml

    mod = load_script()
    config = yaml.safe_load((
        ROOT / "configs/benchmarks/abeta_4hix_af2_gate_v1.yml"
    ).read_text())
    entities = mod.build_entities(config)
    assert len(entities) == 27
    assert sum(row["entity_type"] == "candidate" for row in entities) == 24
    assert {row["entity_id"] for row in entities if row["entity_type"] == "control"} == {
        "native", "proteinmpnn_best", "composition_shuffle"
    }
    assert all(len(row["h3_sequence"]) == 12 for row in entities)


# H3 替换必须精确限定在 [95:107) 区间，框架区（前后）逐残基不变，防止替换逻辑越界破坏支架。
def test_h3_replacement_preserves_framework():
    mod = load_script()
    heavy = "A" * 95 + "VRYDHYSGSSDY" + "C" * 10
    replaced = mod.replace_h3(heavy, "LRWDHYSGSSDY")
    assert replaced[:95] == heavy[:95]
    assert replaced[95:107] == "LRWDHYSGSSDY"
    assert replaced[107:] == heavy[107:]


# 多样性筛选：贪心选择遇到 Hamming 距离不足的近重复（b 距 a 为 1）时必须跳过而非回填，
# 保证短名单内的序列两两差异至少达到 minimum_pairwise_hamming。
def test_af2_gate_diversity_does_not_backfill_near_duplicates():
    path = ROOT / "scripts" / "analyze_abeta_4hix_af2_gate.py"
    spec = importlib.util.spec_from_file_location("analyze_abeta_4hix_af2_gate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    config = {"selection": {"minimum_pairwise_hamming": 2, "shortlist_target": 3}}
    rows = [
        {"entity_id": "a", "entity_type": "candidate", "h3_sequence": "AAAA",
         "all_gates_pass": True, "fixed_rank_score": 3.0, "median_iptm": 0.5,
         "median_interface_pae": 10.0},
        {"entity_id": "b", "entity_type": "candidate", "h3_sequence": "AAAT",
         "all_gates_pass": True, "fixed_rank_score": 2.0, "median_iptm": 0.5,
         "median_interface_pae": 10.0},
        {"entity_id": "c", "entity_type": "candidate", "h3_sequence": "AATT",
         "all_gates_pass": True, "fixed_rank_score": 1.0, "median_iptm": 0.5,
         "median_interface_pae": 10.0},
    ]
    selected = mod.diverse_shortlist(config, rows)
    assert [row["entity_id"] for row in selected] == ["a", "c"]
