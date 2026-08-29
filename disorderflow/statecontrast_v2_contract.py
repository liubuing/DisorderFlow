"""Validation helpers for explicit target/apo/off-target training manifests."""

from __future__ import annotations

from collections import Counter, defaultdict


VALID_STATES = {"target", "apo", "off_target"}


def validate_statecontrast_v2_records(records, minimum_teacher_coverage=0.0):
    """Validate complete state groups and summarize source/teacher coverage."""
    records = list(records)
    groups = defaultdict(list)
    errors = []
    for index, record in enumerate(records):
        group_id = record.get("group_id")
        state = record.get("state") or {}
        state_type = state.get("type")
        if not group_id:
            errors.append(f"record[{index}] missing group_id")
            continue
        if state_type not in VALID_STATES:
            errors.append(f"record[{index}] invalid state.type={state_type!r}")
        if not state.get("source"):
            errors.append(f"record[{index}] missing state.source")
        prior = state.get("prior_weight")
        if prior is None or float(prior) <= 0:
            errors.append(f"record[{index}] requires positive state.prior_weight")
        if "state_training_weight" not in (record.get("targets") or {}):
            errors.append(f"record[{index}] missing targets.state_training_weight")
        groups[group_id].append(record)

    group_summary = []
    for group_id, rows in groups.items():
        states = Counter((row.get("state") or {}).get("type") for row in rows)
        complete = states["target"] > 0 and (states["apo"] > 0 or states["off_target"] > 0)
        if not complete:
            errors.append(
                f"group {group_id} requires target plus apo or off_target")
        group_summary.append({
            "group_id": group_id,
            "records": len(rows),
            "state_counts": dict(states),
            "sources": sorted({(row.get("state") or {}).get("source") for row in rows}),
            "complete": complete,
        })
    teacher_required = [
        record for record in records
        if (record.get("targets") or {}).get("independent_teacher_required", True)
    ]
    teacher_count = sum(
        (record.get("targets") or {}).get("independent_teacher_score") is not None
        for record in teacher_required
    )
    teacher_coverage = teacher_count / len(teacher_required) if teacher_required else 0.0
    if teacher_coverage < float(minimum_teacher_coverage):
        errors.append(
            f"independent teacher coverage {teacher_coverage:.3f} below "
            f"{float(minimum_teacher_coverage):.3f}")
    return {
        "valid": not errors,
        "record_count": len(records),
        "group_count": len(groups),
        "complete_group_count": sum(row["complete"] for row in group_summary),
        "teacher_count": teacher_count,
        "teacher_required_count": len(teacher_required),
        "teacher_coverage": teacher_coverage,
        "groups": group_summary,
        "errors": errors,
    }
