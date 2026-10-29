"""IdP 系综修正 v3 的 fail-closed 契约测试：守护修正门槛的逐项核对（自回归生成器、未触动队列、
独立成分数、更广突变空间、实验测量），以及状态文件一次性写入不可覆盖。"""
import json
from pathlib import Path

import pytest

from scripts.check_idp_ensemble_correction_v3 import check

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/benchmarks/idp_ensemble_correction_v3.yml"


# fail-closed：缺实验测量时整体必须 blocked 且通过计数冻结在 4，状态落盘一致——
# 防止证据不全的修正流程被放行。
def test_correction_contract_is_fail_closed(tmp_path):
    result = check(CONFIG, tmp_path / "status.json")
    assert result["status"] == "correction_blocked"
    assert result["passed_count"] == 4
    assert result["checks"]["autoregressive_generator"] is True
    assert result["checks"]["untouched_cohort"] is True
    assert result["checks"]["minimum_independent_components"] is True
    assert result["checks"]["broader_mutation_space"] is True
    assert result["checks"]["experimental_measurements"] is False
    assert json.loads((tmp_path / "status.json").read_text(encoding="ascii"))["status"] == (
        "correction_blocked"
    )


# write-once：状态文件已存在时再次执行必须报错，防止修正状态被事后覆盖或重置。
def test_correction_status_is_write_once(tmp_path):
    output = tmp_path / "status.json"
    check(CONFIG, output)
    with pytest.raises(FileExistsError):
        check(CONFIG, output)
