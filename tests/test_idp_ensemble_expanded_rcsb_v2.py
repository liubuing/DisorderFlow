from datetime import datetime, timezone
from pathlib import Path

from scripts.run_idp_ensemble_expanded_rcsb_v2 import build_plan

ROOT = Path(__file__).resolve().parents[1]


def test_expanded_rcsb_plan_is_cadence_guarded():
    plan = build_plan(
        ROOT / "configs/benchmarks/idp_ensemble_expanded_development_v2.yml",
        now=datetime(2026, 8, 18, tzinfo=timezone.utc),
    )
    assert plan["status"] == "cadence_gate_closed"
    assert plan["eligible_to_execute"] is False
    assert len(plan["query_spec"]["query"]["nodes"][0]["nodes"]) >= 6
    assert len(plan["query_spec"]["query"]["nodes"][1]["nodes"]) >= 8
