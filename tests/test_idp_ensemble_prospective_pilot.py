"""IdP 系综前瞻试点（v2）的契约测试：守护"先验证契约再访问候选"的执行顺序、六组件与
每臂 144 候选的预算冻结，以及候选清单一次性写入不可覆盖——保证试点设计在看到数据前已锁定。"""
import json
from pathlib import Path

import pytest

from scripts.prepare_idp_ensemble_prospective_pilot import prepare

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/benchmarks/idp_ensemble_prospective_pilot_v2.yml"


# 契约先行：prepare 必须在候选访问前完成契约校验，物化恰好 6 个干净组件、
# 每臂 144 个生成候选并把清单落盘——防止预算或组件数事后变更。
def test_pilot_contract_materializes_six_clean_components(tmp_path, monkeypatch):
    output = tmp_path / "manifest.json"
    config = CONFIG.read_text(encoding="ascii").replace(
        "reviewer_outputs/idp_ensemble_prospective_pilot_v2/candidate_manifest.json",
        str(output).replace("\\", "/"),
    )
    config_path = tmp_path / "config.yml"
    config_path.write_text(config, encoding="ascii")
    monkeypatch.setattr(
        "scripts.prepare_idp_ensemble_prospective_pilot.ROOT", ROOT
    )
    result = prepare(config_path)
    assert result["status"] == "contract_validated_before_candidate_access"
    assert len(result["components"]) == 6
    assert result["budget"]["total_generated_candidates_per_arm"] == 144
    assert json.loads(output.read_text(encoding="ascii"))["components"]


# write-once：重复执行必须报错，防止前瞻清单被第二次生成覆盖（前瞻设计不可重抽）。
def test_pilot_manifest_is_write_once(tmp_path):
    output = tmp_path / "manifest.json"
    config = CONFIG.read_text(encoding="ascii").replace(
        "reviewer_outputs/idp_ensemble_prospective_pilot_v2/candidate_manifest.json",
        str(output).replace("\\", "/"),
    )
    config_path = tmp_path / "config.yml"
    config_path.write_text(config, encoding="ascii")
    prepare(config_path)
    with pytest.raises(FileExistsError):
        prepare(config_path)
