"""多支架候选界面构建脚本测试：验证冻结划分、冒烟产物有效性与聚合隔离。

守护 candidate_interface_multiscaffold_v1 管线：
split_manifest 运行时与发布版本逐字节一致且三划分互斥完备、
冒烟结果的 PAE/pLDDT 产物自洽、
v2 不确定性用样本标准差并更新 SEM 且清理中间复测字段、
目标聚合必须在加载封存组件之前完成筛选（test 组件不得被加载）。
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


# 辅助函数：以独立模块方式加载 scripts/build_candidate_interface_multiscaffold_v1.py。
def load_script():
    path = ROOT / "scripts/build_candidate_interface_multiscaffold_v1.py"
    spec = importlib.util.spec_from_file_location(
        "build_candidate_interface_multiscaffold_v1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 冻结划分溯源：运行时 manifest 必须与 publication 版本逐字节相同，
# train/calibration/test 大小固定、两两不相交且并集覆盖全部 20 个实体——
# 防止划分被静默改写或样本跨划分泄漏。
def test_frozen_split_is_disjoint_and_complete():
    runtime_path = ROOT / "data/candidate_interface_multiscaffold_v1/split_manifest.json"
    versioned_path = ROOT / "publication/candidate_interface_multiscaffold_v1_split_manifest.json"
    assert runtime_path.read_bytes() == versioned_path.read_bytes()
    split = json.loads(runtime_path.read_text(encoding="utf-8"))
    train = set(split["train"])
    calibration = set(split["calibration"])
    test = set(split["test"])
    assert len(train) == 15
    assert len(calibration) == 2
    assert len(test) == 3
    assert not train & calibration
    assert not train & test
    assert not calibration & test
    assert len(train | calibration | test) == 20


# 冒烟产物自洽性：PAE 矩阵有限且非常数、pLDDT 与记录的汇总值一致、
# 三条链齐全——防止残缺或伪造的预测产物进入下游。（产物缺失时跳过）
def test_smoke_pae_and_pdb_plddt_are_valid():
    mod = load_script()
    path = ROOT / "results/candidate_interface_multiscaffold_v1/smoke/results.json"
    if not path.exists():
        return
    results = json.loads(path.read_text(encoding="utf-8"))
    row = results["results"][0]
    plddt, sequences = mod.residue_plddt_and_sequences(ROOT / row["pdb"])
    pae = np.load(ROOT / row["pae_npz"], allow_pickle=False)["pae"]
    assert list(sequences) == ["A", "B", "C"]
    assert pae.shape == (len(plddt), len(plddt))
    assert np.isfinite(pae).all()
    assert np.ptp(pae) > 0
    assert abs(float(plddt.mean() / 100.0) - row["plddt"]) <= 5e-4


# v2 不确定性必须使用样本标准差（ddof=1）推导 SEM、删除中间复测字段，
# 并把 PAE 监督掩码限制到抗原索引上——防止统计口径错误或中间数据泄漏进产物。
def test_v2_uncertainty_uses_sample_sd_and_updates_sem():
    mod = load_script()
    rows = []
    for index, value in enumerate((0.2, 0.4, 0.6)):
        rows.append({
            "construct_id": "entity",
            "protocol_id": "model_1" if index < 2 else "model_2",
            "af2_seed": index,
            "_replicate_candidate_plddt": value,
            "_replicate_iptm": value + 0.1,
            "_replicate_candidate_to_antigen_pae": {
                10: value + 0.2,
                11: value + 0.3,
            },
            "_patch_global_indices": [0, 10, 11],
            "batch": {},
        })
    mod.apply_v2_uncertainty(rows)
    expected_std = np.std([0.2, 0.4, 0.6], ddof=1)
    for row in rows:
        assert row["af2_candidate_plddt_std"] == pytest.approx(expected_std)
        assert row["af2_candidate_plddt_sem"] > 0
        assert "_replicate_candidate_to_antigen_pae" not in row
        assert torch.equal(
            row["batch"]["pae_supervision_antigen_mask"],
            torch.tensor([False, True, True]))


# 消融/评估隔离：allowed_components 之外的组件（TEST）绝不能被加载，
# 加载须发生在筛选之后——防止未授权组件的封存预测数据被读取。
def test_target_aggregation_filters_before_loading_sealed_components(monkeypatch):
    mod = load_script()
    loaded = []

    def fake_load(row):
        loaded.append(row["component_id"])
        return (
            np.asarray([0.5, 0.6]),
            np.asarray([[0.0, 1.0], [1.0, 0.0]]),
            {"A": "A", "B": "", "C": "A"},
        )

    monkeypatch.setattr(mod, "load_prediction_arrays", fake_load)
    result_sets = [{
        "requested": {"seeds": [1]},
        "results": [
            {"entity_id": "train", "component_id": "TRAIN", "status": "success",
             "iptm": 0.5, "pdb": "train"},
            {"entity_id": "test", "component_id": "TEST", "status": "success",
             "iptm": 0.5, "pdb": "test"},
        ],
    }]
    entities = {
        "train": {"component_id": "TRAIN", "heavy_sequence": "A",
                  "light_sequence": "", "antigen_sequence": "A"},
        "test": {"component_id": "TEST", "heavy_sequence": "A",
                 "light_sequence": "", "antigen_sequence": "A"},
    }
    components = {
        "TRAIN": {"h3_heavy_indices_zero_based": [0]},
        "TEST": {"h3_heavy_indices_zero_based": [0]},
    }
    mod.aggregate_targets(
        result_sets, entities, components, allowed_components={"TRAIN"})
    assert loaded == ["TRAIN"]
