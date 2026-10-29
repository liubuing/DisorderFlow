"""未触动队列准入的契约测试：守护"历史登记组件不得构成确认队列"的隔离原则——
确认集必须来自发现之后的新结构，旧成分一律阻塞准入。"""
import json

from scripts.admit_idp_ensemble_untouched_cohort import admit


# 隔离准入：发现清单为空且登记表只含旧组件时，必须 blocked、观测新结构数为 0、
# 决策以 do_not 开头——防止确认集被历史数据污染。
def test_historical_registry_cannot_admit_confirmation_cohort(tmp_path):
    discovery = tmp_path / "discovery.json"
    registry = tmp_path / "components.csv"
    output = tmp_path / "admission.json"
    discovery.write_text(json.dumps({"candidate_entry_ids": []}), encoding="ascii")
    registry.write_text("component_id,target\nOLD,tau\n", encoding="ascii")
    result = admit(discovery, registry, output)
    assert result["status"] == "untouched_cohort_blocked"
    assert result["observed"]["exact_new_structures"] == 0
    assert result["decision"].startswith("do_not")
