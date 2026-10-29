"""外部无序度构建与 CAID 基准的契约测试：守护外部预测分数解析（含序列校验）、
外部产物拆分标记不得隐式当作 val、CAID 指标包含阈值与校准项，以及聚类隔离对训练重叠的拒绝。"""
import csv
import pickle

import numpy as np
import pytest

from scripts.benchmark_caid import assert_external_cluster_isolation, evaluate
from scripts.build.build_external_disorder_lookup import parse_scores


# 解析契约：无表头 CSV 按 行ID,分数... 解析为逐残基 profile，防止格式漂移静默错位。
def test_parse_metapredict_scores(tmp_path):
    path = tmp_path / 'scores.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['sample-a', '0.1', '0.8'])
    profiles = parse_scores(path)
    assert np.allclose(profiles['sample-a'], [0.1, 0.8])


# v3 格式：提供参考序列时必须校验 CSV 中序列列与之一致，防止分数与序列错配进入查找表。
def test_parse_metapredict_v3_scores_checks_sequence(tmp_path):
    path = tmp_path / 'scores.csv'
    path.write_text('sample-a, AC, 0.1, 0.8\n', encoding='utf-8')
    profiles = parse_scores(path, {'sample-a': 'AC'})
    assert np.allclose(profiles['sample-a'], [0.1, 0.8])


# 拆分溯源：外部盲测产物的 split 必须显式为 external_blind，防止外部数据被默认路由进验证集。
def test_external_artifact_split_is_not_implicit_val(tmp_path):
    artifact = {'schema_version': 2, 'split': 'external_blind', 'profiles': {}}
    path = tmp_path / 'artifact.pkl'
    path.write_bytes(pickle.dumps(artifact))
    assert pickle.loads(path.read_bytes())['split'] == 'external_blind'


# 指标完备：可完美分离的输入下 AUC/F1/MCC 均为 1，且 Brier/ECE 评估校准质量——
# 保证外部基准报告同时覆盖判别与校准两个维度。
def test_caid_metrics_include_threshold_and_calibration():
    metrics = evaluate([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert metrics['auc_roc'] == 1.0
    assert metrics['f1'] == 1.0
    assert metrics['mcc'] == 1.0
    assert metrics['brier'] < 0.05
    assert metrics['ece'] < 0.2


# 泄漏拒绝：目标聚簇与训练聚簇重叠必须报 leakage 错误，防止外部评估混入训练同源序列。
def test_caid_cluster_isolation_rejects_training_overlap():
    targets = [{"id": "caid-a", "cluster_id": "cluster-1"}]
    with pytest.raises(ValueError, match="leakage"):
        assert_external_cluster_isolation(targets, ["cluster-1"])
