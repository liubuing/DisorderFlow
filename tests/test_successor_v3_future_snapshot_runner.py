from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.run_successor_v3_future_snapshot import build_plan, execute_plan

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs/benchmarks/successor_v3_future_snapshot_policy.yml"
SABDAB = ROOT / "data/sabdab2_snapshot_2026_08_06_successor_v3/all-summary.csv"
REFERENCE = ROOT / "data/successor_v3_confirmatory/reference_union_manifest_v3.json"


def test_future_snapshot_plan_blocks_before_exact_cadence_timestamp():
    plan = build_plan(
        POLICY,
        SABDAB,
        REFERENCE,
        now=datetime(2026, 8, 14, 9, 58, 12, tzinfo=timezone.utc),
    )
    assert plan["eligible_to_execute"] is False
    assert plan["next_permitted_at_utc"] == "2026-08-14T09:58:13.565916Z"
    with pytest.raises(RuntimeError, match="Cadence gate closed"):
        execute_plan(plan)


def test_future_snapshot_plan_opens_at_exact_cadence_timestamp():
    plan = build_plan(
        POLICY,
        SABDAB,
        REFERENCE,
        now=datetime(2026, 8, 14, 9, 58, 14, tzinfo=timezone.utc),
    )
    assert plan["eligible_to_execute"] is True
    assert plan["outputs"]["snapshot_directory"].endswith(
        "successor_v3_rcsb_snapshot_2026_08_14"
    )
    assert "no coordinate" in plan["access_boundary"]
