#!/usr/bin/env python
"""Run direct ensemble-conditioned design and held-out evaluation for non-A-beta IDPs."""
# 本脚本对非 Aβ IDP 靶标（tau / α-synuclein）做"集合条件化"直接设计与
# 留一构象评估：以通过审计的 pose 面板为训练集合，在束搜索中按
# native_delta 的 p25/最小值（稳健下界）选择候选；tau 采用非劣性目标
# （p25 阈值 -0.005），其他靶标要求改进。留一评估在留出构象上检验
# 所选候选能否击败天然 paratope。产出声明仅限"同靶标计算留出构象证据"。

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "modules"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "pipeline"))

from design_abeta_midregion_ensemble import design_reference, ensemble_metrics  # noqa: E402
from ensemble_pose_transfer import fixed_paratope_contact_map  # noqa: E402
from state_contact_scorer import extract_contact_map, score_sequence_on_contact_map  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/non_abeta_idp_ensemble_prep_v1/ensemble_manifest.json")
    parser.add_argument("--pose-audit", default="outputs/non_abeta_idp_pose_panels_v1/pose_panel_audit.json")
    parser.add_argument("--out", default="outputs/non_abeta_idp_ensemble_design_v1")
    parser.add_argument("--beam-width", type=int, default=250)
    parser.add_argument("--max-mutations", type=int, default=4)
    parser.add_argument("--top-per-target", type=int, default=20)
    return parser.parse_args()


def heldout_evaluation(target, candidates, pose_rows):
    # 留一构象评估：每次留出一个 pose 作验证，其余作训练集合；
    # 用训练集合的稳健指标（native_delta_p25/min）选出赢家，
    # 再在留出构象上比较赢家与天然 paratope 的得分
    if not candidates:
        return []
    template = PROJECT_ROOT / target["trimmed_reference_pdb"]
    template_map = extract_contact_map(str(template), peptide_chain=target["reference_peptide_chain"])
    maps = [
        fixed_paratope_contact_map(PROJECT_ROOT / row["pose_pdb"], template_map)
        for row in sorted(pose_rows, key=lambda item: item["conformer"])
    ]
    native = template_map["paratope_sequence"]
    native_scores = np.asarray([
        score_sequence_on_contact_map(native, contact_map)["state_contact_score"]
        for contact_map in maps
    ])
    sequences = [row["sequence"] for row in candidates]
    matrix = np.asarray([
        [score_sequence_on_contact_map(sequence, contact_map)["state_contact_score"] for contact_map in maps]
        for sequence in sequences
    ])
    rows = []
    for holdout in range(len(maps)):
        training = [index for index in range(len(maps)) if index != holdout]
        training_native = native_scores[training]
        winner = max(
            range(len(sequences)),
            key=lambda index: (
                ensemble_metrics(sequences[index], [maps[i] for i in training], training_native)["native_delta_p25"],
                ensemble_metrics(sequences[index], [maps[i] for i in training], training_native)["native_delta_min"],
            ),
        )
        score = float(matrix[winner, holdout])
        native_score = float(native_scores[holdout])
        rows.append({
            "target": target["target"],
            "heldout_conformer": holdout,
            "selected_candidate_uid": candidates[winner]["candidate_uid"],
            "heldout_score": score,
            "native_score": native_score,
            "heldout_delta_native": score - native_score,
            "beats_native": score > native_score,
        })
    return rows


def main():
    # 主流程：校验 pose 面板已通过 → 逐靶标设计（tau 为非劣性目标，
    # 其余为改进目标；全链风险上限 0.55、最小 native_delta -0.01）→
    # 留一评估 → 写候选/突变计划/全长链 FASTA/留一明细与总结报告
    args = parse_args()
    with open(PROJECT_ROOT / args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    with open(PROJECT_ROOT / args.pose_audit, encoding="utf-8") as handle:
        pose_audit = json.load(handle)
    if pose_audit["status"] != "pass":
        raise SystemExit("Strict pose panel has not passed")
    selected = []
    plans = []
    summaries = []
    heldout = []
    for target in manifest["targets"]:
        objective_mode = "noninferiority" if target["target"] == "tau" else "improvement"
        design_args = SimpleNamespace(
            beam_width=args.beam_width,
            max_mutations=args.max_mutations,
            top_per_reference=args.top_per_target,
            max_full_chain_risk=0.55,
            min_native_delta=-0.01,
            min_native_delta_p25=(-0.005 if objective_mode == "noninferiority" else 0.0),
        )
        reference = {
            "pdb": target["reference_id"],
            "complex_pdb": target["trimmed_reference_pdb"],
            "peptide_chain": target["reference_peptide_chain"],
        }
        target_pose_rows = [
            {**row, "reference_pdb": target["reference_id"]}
            for row in pose_audit["entries"]
            if row["target"] == target["target"] and row.get("selected_for_panel")
        ]
        target_selected, target_plans, summary = design_reference(
            reference, target_pose_rows, design_args
        )
        for row in target_selected:
            row["target"] = target["target"]
            row["objective_mode"] = objective_mode
        selected.extend(target_selected)
        plans.extend(target_plans)
        summaries.append({
            **summary,
            "target": target["target"],
            "objective_mode": objective_mode,
            "min_native_delta_p25": design_args.min_native_delta_p25,
            "min_native_delta": design_args.min_native_delta,
        })
        heldout.extend(heldout_evaluation(target, target_selected, [
            row for row in pose_audit["entries"]
            if row["target"] == target["target"] and row.get("selected_for_panel")
        ]))

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ensemble_designed_candidates.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    with open(out_dir / "ensemble_designed_mutation_plan.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plans[0]))
        writer.writeheader()
        writer.writerows(plans)
    with open(out_dir / "ensemble_designed_full_chains.fasta", "w", encoding="ascii") as handle:
        for row in selected:
            handle.write(f">{row['candidate_uid']}|{row['target']}|heavy\n{row['heavy_sequence']}\n")
            handle.write(f">{row['candidate_uid']}|{row['target']}|light\n{row['light_sequence']}\n")
    with open(out_dir / "leave_one_conformer_out.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(heldout[0]))
        writer.writeheader()
        writer.writerows(heldout)
    by_target = {
        target["target"]: {
            "selected": sum(row["target"] == target["target"] for row in selected),
            "heldout_mean_delta_native": float(np.mean([
                row["heldout_delta_native"] for row in heldout if row["target"] == target["target"]
            ])) if any(row["target"] == target["target"] for row in heldout) else None,
            "heldout_beat_native_rate": float(np.mean([
                row["beats_native"] for row in heldout if row["target"] == target["target"]
            ])) if any(row["target"] == target["target"] for row in heldout) else None,
        }
        for target in manifest["targets"]
    }
    report = {
        "schema_version": "nonabeta.idp_ensemble_design.v1",
        "status": "pass" if all(
            row["selected"] >= 2
            and row["heldout_mean_delta_native"] is not None
            and row["heldout_mean_delta_native"] >= (-0.005 if target == "tau" else 0.0)
            for target, row in by_target.items()
        ) else "partial",
        "by_target": by_target,
        "design_summaries": summaries,
        "claim_boundary": "Within-target computational held-out conformer evidence only.",
    }
    with open(out_dir / "ensemble_design_summary.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Non-A-beta ensemble design: selected={len(selected)} status={report['status']}")
    if report["status"] != "pass":
        raise SystemExit("Non-A-beta design/heldout gate partial")


if __name__ == "__main__":
    main()
