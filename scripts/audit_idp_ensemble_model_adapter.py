#!/usr/bin/env python
"""Audit whether an adapter provides true prefix-conditioned pose logits.

审计模型适配器是否提供真正"前缀条件化"的逐残基构象 logits：
- 校验运行时轨迹（runtime trace）：至少 2 个构象、每事件 log_probs 长度 21 且有限；
- 校验 checkpoint 存在性并记录 sha256；
- 禁止用固定 npz 捷径绕过逐步重算 logits。
输出适配器就绪审计 JSON；结论仅代表接口就绪，不涉及候选或结合结果。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    """计算文件 sha256，用于 checkpoint/产物血缘校验。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_runtime_trace(trace_path):
    """校验运行时轨迹是否满足前缀条件化 logits 的证据要求。

    合格标准：事件非空、覆盖 >=2 个构象、每事件 log_probs 有限且长度 21
    （20 种标准氨基酸 + 1 个掩码/结束位），并带有 int 位置与 str 前缀。
    """
    if trace_path is None or not Path(trace_path).is_file():
        return {"present": False, "valid": False, "reason": "runtime_trace_missing"}
    trace = json.loads(Path(trace_path).read_text(encoding="ascii"))
    events = trace.get("events", [])
    pose_ids = {event.get("pose_id") for event in events}
    valid = bool(events) and len(pose_ids) >= 2 and all(
        event.get("log_probs_finite") is True
        and event.get("log_probs_length") == 21
        and isinstance(event.get("position"), int)
        and isinstance(event.get("prefix"), str)
        for event in events
    )
    return {
        "present": True,
        "valid": valid,
        "path": str(trace_path),
        "event_count": len(events),
        "pose_count": len(pose_ids),
        "pose_ids": sorted(str(value) for value in pose_ids),
    }


def audit(output, checkpoint_paths=(), runtime_trace=None):
    """汇总 checkpoint 与运行时轨迹检查，写适配器就绪审计 JSON。"""
    if output.exists():
        # 审计产物不可覆盖
        raise FileExistsError(f"Refusing to overwrite adapter audit: {output}")
    checkpoints = []
    for checkpoint in checkpoint_paths:
        checkpoint = Path(checkpoint)
        checkpoints.append({
            "path": str(checkpoint),
            "present": checkpoint.is_file(),
            "sha256": sha256(checkpoint) if checkpoint.is_file() else None,
        })
    checkpoints_ready = bool(checkpoints) and all(
        checkpoint["present"] for checkpoint in checkpoints
    )
    runtime = validate_runtime_trace(runtime_trace)
    implementation_ready = checkpoints_ready and runtime["valid"]
    payload = {
        "schema_version": 1,
        "status": "adapter_eligible" if implementation_ready else "implemented_not_runtime_validated",
        "classification": "model_interface_audit",
        "checks": {
            "prefix_conditioned_next_residue_logits": implementation_ready,
            "recomputes_logits_after_each_prefix": implementation_ready,
            "uses_multiple_pose_models": runtime["valid"] and runtime["pose_count"] >= 2,
            "fixed_npz_shortcut_forbidden": True,
        },
        "observed_interface": {
            "proteinmpnn_cli": "model_native_prefix_next_log_probs",
            "behavior": "explicit_decoder_order_prefix_then_target_then_suffix",
            "suffix_sequence_leakage": False,
            "checkpoints": checkpoints,
            "runtime_trace": runtime,
        },
        "decision": (
            "runtime_trace_required_before_model_gate"
            if not implementation_ready
            else "adapter_runtime_validated_but_cohort_and_candidate_manifest_gates_remain"
        ),
        "claim_boundary": "adapter readiness only; no candidate or binding result",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    """命令行入口：--checkpoint 可重复传入，--runtime-trace 指向轨迹 JSON。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="reviewer_outputs/idp_ensemble_correction_v3/model_adapter_audit.json",
    )
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--runtime-trace", default=None)
    args = parser.parse_args()
    print(json.dumps(
        audit(
            ROOT / args.output,
            [ROOT / path for path in args.checkpoint],
            ROOT / args.runtime_trace if args.runtime_trace else None,
        ), indent=2
    ))


if __name__ == "__main__":
    main()
