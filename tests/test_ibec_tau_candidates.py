"""IBEC tau 候选库生成测试：验证保守库可复现性、组成保持对照与冻结参考区间。

守护 scripts.run_ibec_tau_candidates 的 tau 候选生成管线：
保守库同种子可复现、无半胱氨酸且 Hamming 距离受控、
组成打乱严格保持氨基酸多重集、
5MP3 参考的 H3 区间与残基编号冻结（防止参考重编号漂移）、
numpy 布尔值必须转为 JSON 可序列化类型。
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


# 辅助函数：以独立模块方式加载被测脚本。
def load_script():
    path = ROOT / "scripts" / "run_ibec_tau_candidates.py"
    spec = importlib.util.spec_from_file_location("run_ibec_tau_candidates", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 保守库生成：同参数两次调用结果完全一致（种子化可复现）、无重复、
# 不引入半胱氨酸、每条序列与亲本的 Hamming 距离在 [1,4]——防止突变预算失控。
def test_conservative_library_is_reproducible_and_cys_free():
    mod = load_script()
    first = mod.conservative_library("ARDYYGTSFAMDY", 50, 7301, 1, 4)
    second = mod.conservative_library("ARDYYGTSFAMDY", 50, 7301, 1, 4)
    assert first == second
    assert len(set(first)) == 50
    assert all("C" not in row for row in first)
    assert all(1 <= mod.hamming(row, "ARDYYGTSFAMDY") <= 4 for row in first)


# 组成打乱对照：30 条序列互不相同且氨基酸组成与亲本完全一致——
# 保证阴性对照只破坏顺序信息，不引入组成混淆变量。
def test_composition_shuffles_preserve_exact_composition():
    mod = load_script()
    rows = mod.composition_shuffles("ARDYYGTSFAMDY", 30, 7303)
    assert len(set(rows)) == 30
    assert all(sorted(row) == sorted("ARDYYGTSFAMDY") for row in rows)


# 冻结参考溯源：tau 参考结构 A 链的 H3 序列、1-based 残基编号以及两端锚点残基
# （C97/W109）必须与冻结值一致——防止 PDB 重新解析方式改变导致设计区间漂移。
def test_reference_h3_and_resid_window_are_frozen():
    mod = load_script()
    path = ROOT / "data/non_abeta_idp/ensembles_v1/tau/5MP3_tau_core_complex.pdb"
    heavy, resids = mod.chain_sequence_and_resids(path, "A")
    assert heavy[95:108] == "ARDYYGTSFAMDY"
    assert resids[95:108] == list(range(97, 110))
    assert heavy[94] == "C" and heavy[108] == "W"


# JSON 序列化防护：numpy 标量参与的布尔比较必须转为原生 bool——
# 防止 json.dumps 抛 TypeError 或输出 numpy 特有的字符串形式。
def test_numpy_gate_comparisons_are_normalized_for_json():
    mod = load_script()
    checks = mod.normalize_gate_checks({
        "bfn_mean_noninferior": np.float64(0.1) >= np.float64(0.0),
    })
    assert json.loads(json.dumps(checks)) == {"bfn_mean_noninferior": True}
