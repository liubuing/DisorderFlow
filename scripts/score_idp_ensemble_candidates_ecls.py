#!/usr/bin/env python
"""Score matched H3 candidates by ProteinMPNN epitope-conditioning gain."""

# ECLS（epitope-conditioning likelihood score）打分脚本：对匹配生成的
# IDP 候选 H3 序列，用 ProteinMPNN 的"表位条件增益"打分——
# 即抗原存在(complex)与不存在(apo)两种骨架下 H3 序列对数似然之差。
# 增益越大说明序列越依赖抗原表位信息，理论上越倾向于真实结合构象。
# 复用 benchmark_h3_epitope_delta / score_idp_ensemble_bfn 的既有实现，
# 保证与其他基准的统计口径一致。
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 延迟导入仓库内模块（需先把 ROOT 加入 sys.path）。
from scripts.benchmark_h3_epitope_delta import sequence_nll  # noqa: E402
from scripts.generate_idp_ensemble_matched_baselines import canonicalize_component  # noqa: E402
from scripts.score_idp_ensemble_candidates_bfn import summarize  # noqa: E402
from scripts.score_idp_ensemble_matched_candidates import (  # noqa: E402
    candidate_pools,
    leave_one_out,
    pose_rows_by_component,
)


def sha256(path):
    # 计算文件 sha256，用于冻结输入的完整性校验与缓存失效判断。
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_apo(complex_path, apo_path):
    # 从复合物 PDB 中剥离肽链 P 的所有原子行，得到仅含受体(H/L)的 apo 结构。
    # 保留非 P 链的 ATOM/HETATM 记录以及所有其他行（如晶体水、备注等）。
    lines = [
        line for line in complex_path.read_text(encoding="ascii").splitlines(keepends=True)
        if not line.startswith(("ATOM  ", "HETATM")) or line[21] != "P"
    ]
    apo_path.write_text("".join(lines), encoding="ascii")


def run_conditional(config, pdb_path, h3_indices, heavy_length, work_dir):
    # 调用 ProteinMPNN 计算条件对数概率：固定重链上除 H3 环区以外的所有位置，
    # 只输出 H3 区在给定骨架下的 log_p（conditional_probs_only 模式）。
    # 结果缓存到 npz；已存在则跳过子进程调用（幂等重跑）。
    settings = config["proteinmpnn"]
    name = pdb_path.stem
    work_dir.mkdir(parents=True, exist_ok=True)
    # fixed = 1-based 的重链固定位置：H3 区（h3_indices 为 0-based）之外全部固定。
    fixed = [
        position for position in range(1, heavy_length + 1)
        if position - 1 not in set(h3_indices)
    ]
    fixed_path = work_dir / "fixed_positions.jsonl"
    fixed_path.write_text(json.dumps({name: {"H": fixed}}) + "\n", encoding="ascii")
    output_dir = work_dir / "mpnn"
    result_path = output_dir / "conditional_probs_only" / f"{name}.npz"
    command = [
        sys.executable,
        str(ROOT / settings["script"]),
        "--pdb_path", pdb_path.resolve().as_posix(),
        "--pdb_path_chains", "H",
        "--fixed_positions_jsonl", fixed_path.resolve().as_posix(),
        "--path_to_model_weights", (ROOT / settings["weights"]).resolve().as_posix(),
        "--model_name", settings["model_name"],
        "--conditional_probs_only", "1",
        "--conditional_probs_only_backbone", "1",
        "--num_seq_per_target", "1",
        "--batch_size", "1",
        "--seed", str(settings["seed"]),
        "--suppress_print", "1",
        "--out_folder", output_dir.resolve().as_posix(),
    ]
    if not result_path.exists():
        # timeout=900：单次 ProteinMPNN 前向不应超过 15 分钟，防止子进程挂死。
        completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
        if completed.returncode:
            raise RuntimeError(completed.stderr[-4000:])
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    result = np.load(result_path)
    # 校验 H3 区掩码有效性：design_mask/mask 在 H3 位置必须为正，
    # 否则说明 PDB 编号与 H3 定义不一致，结果不可信。
    for index in h3_indices:
        if result["design_mask"][index] <= 0 or result["mask"][index] <= 0:
            raise ValueError(f"Invalid ProteinMPNN H3 mask at index {index}")
    return result["log_p"][0], {
        "npz": str(result_path.relative_to(ROOT)),
        "npz_sha256": sha256(result_path),
        "command": command,
    }


def score_component(config, component, pose_rows, method_pools, work_dir):
    # 对单个组分(component)在全部构象(pose)上计算 ECLS：
    # 每个 pose 分别求 complex 与 apo 两种条件下的 H3 log_p，
    # 再对每条候选序列求逐 pose 的"条件增益"矩阵（序列 x pose）。
    start, end = component["h3_positions_1_indexed"]
    h3_indices = list(range(start - 1, end))
    conformers = []
    complex_log_probs = []
    apo_log_probs = []
    for index, pose in enumerate(pose_rows):
        # 以 pose 的 PDB 为参考骨架重新规范化组分；肽链在 pose 中编号为 P。
        pose_component = {
            **component,
            "reference_path": pose["pose_pdb"],
            "antigen_chain": "P",
        }
        pose_dir = work_dir / f"pose_{index}"
        complex_path = pose_dir / "complex_HLP.pdb"
        pose_dir.mkdir(parents=True, exist_ok=True)
        canonical = canonicalize_component(pose_component, complex_path)
        heavy_length = int(canonical["chains"]["H"]["length"])
        apo_path = pose_dir / "apo_HL.pdb"
        write_apo(complex_path, apo_path)
        complex_logp, complex_provenance = run_conditional(
            config, complex_path, h3_indices, heavy_length, pose_dir / "complex"
        )
        apo_logp, apo_provenance = run_conditional(
            config, apo_path, h3_indices, heavy_length, pose_dir / "apo"
        )
        complex_log_probs.append(complex_logp)
        apo_log_probs.append(apo_logp)
        # 记录每个 pose 的来源与哈希，保证结果可追溯。
        conformers.append({
            "conformer": pose["conformer"],
            "source_pose": pose["pose_pdb"],
            "source_pose_sha256": sha256(ROOT / pose["pose_pdb"]),
            "canonical": canonical,
            "complex": complex_provenance,
            "apo": apo_provenance,
        })

    def gains(sequence):
        # 逐 pose 的条件增益：apo 的 NLL 减去 complex 的 NLL
        # （NLL 越低 likelihood 越高，故差值为正说明 complex 条件更有利）。
        return [
            sequence_nll(apo, sequence, h3_indices)
            - sequence_nll(complex_logp, sequence, h3_indices)
            for complex_logp, apo in zip(complex_log_probs, apo_log_probs, strict=True)
        ]

    # 天然 H3 序列作为参照基线；各设计方法的候选池逐一打分。
    native = gains(component["native_h3"])
    methods = []
    for method, sequences in method_pools.items():
        methods.append({
            "method": method,
            "sequences": sequences,
            "epitope_conditioning_gain": [gains(sequence) for sequence in sequences],
        })
    return {
        "component_id": component["component_id"],
        "native_h3": component["native_h3"],
        "native_epitope_conditioning_gain": native,
        "conformers": conformers,
        "methods": methods,
    }


def cache_valid(payload, config_hash, generation_hash, component, pools):
    # 缓存有效性判断：配置哈希、生成结果哈希、组分 id 与候选池
    # 任一发生变化即判定缓存失效，必须重算。
    if payload.get("config_sha256") != config_hash:
        return False
    if payload.get("raw_generation_sha256") != generation_hash:
        return False
    scores = payload.get("scores", {})
    if scores.get("component_id") != component["component_id"]:
        return False
    actual = {row["method"]: row["sequences"] for row in scores.get("methods", [])}
    return actual == pools


def pose_rows_from_manifest(components):
    # 从组分清单直接读取 pose 路径（config 的 pose_source 为
    # component_manifest_pose_paths 时使用）。
    # 常规组分要求恰好 5 个 pose；sensitivity_only 组分至少 1 个即可。
    output = {}
    for component in components:
        paths = component.get("pose_paths", [])
        sensitivity_only = bool(component.get("sensitivity_only"))
        if not sensitivity_only and len(paths) != 5:
            raise ValueError(
                f"{component['component_id']} requires exactly five "
                f"manifest pose paths, found {len(paths)}"
            )
        if sensitivity_only and not paths:
            raise ValueError(
                f"{component['component_id']} sensitivity component requires at least "
                "one manifest pose path"
            )
        output[component["component_id"]] = [
            {"conformer": index, "pose_pdb": path}
            for index, path in enumerate(paths)
        ]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/benchmarks/idp_ensemble_ecls_scoring_dev_v1.yml"
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # 冻结输入清单：所有参与打分的上游产物必须与 config 记录的 sha256 一致，
    # 防止在未经审计的输入上继续产出结果。
    frozen = {
        "component_manifest": config["component_manifest"],
        "raw_generation": config["raw_generation"],
    }
    if "preparation" in config:
        frozen["preparation"] = config["preparation"]
    if "pose_audits" in config:
        frozen.update({
            "abeta_pose_audit": config["pose_audits"]["abeta"],
            "nonabeta_pose_audit": config["pose_audits"]["nonabeta"],
        })
    paths = {name: ROOT / item["path"] for name, item in frozen.items()}
    for name, path in paths.items():
        if sha256(path) != frozen[name]["sha256"]:
            raise ValueError(f"Frozen hash mismatch: {path}")
    checkpoint = ROOT / config["proteinmpnn"]["checkpoint"]
    if sha256(checkpoint) != config["proteinmpnn"]["checkpoint_sha256"]:
        raise ValueError("ProteinMPNN checkpoint hash mismatch")

    manifest = json.loads(paths["component_manifest"].read_text(encoding="utf-8"))
    raw = json.loads(paths["raw_generation"].read_text(encoding="utf-8"))
    # 匹配生成必须完整成功（status=complete 且无失败项），否则候选池不完整。
    if raw["status"] != "complete" or raw["failures"]:
        raise ValueError("Matched generation must be complete")
    components = manifest["components"]
    methods = config["candidate_pool"]["methods"]
    pools = candidate_pools(raw, methods)
    if config.get("pose_source") == "component_manifest_pose_paths":
        poses = pose_rows_from_manifest(components)
    else:
        # 默认从 abeta / non-abeta 两份 pose 审计文件中提取 pose 行。
        poses = pose_rows_by_component(
            components,
            json.loads(paths["abeta_pose_audit"].read_text(encoding="utf-8")),
            json.loads(paths["nonabeta_pose_audit"].read_text(encoding="utf-8")),
        )

    out_path = ROOT / (args.out or config["output"])
    work_dir = out_path.parent / "work"
    config_hash = sha256(config_path)
    generation_hash = sha256(paths["raw_generation"])
    component_scores = []
    for component in components:
        # 逐组分打分，结果按组分缓存到 work/<id>/ecls_scores.json；
        # 命中缓存可跳过昂贵的 ProteinMPNN 子进程调用。
        component_id = component["component_id"]
        component_pools = {method: pools[(component_id, method)] for method in methods}
        cache_path = work_dir / component_id / "ecls_scores.json"
        cached = None
        if cache_path.exists():
            candidate = json.loads(cache_path.read_text(encoding="utf-8"))
            if cache_valid(candidate, config_hash, generation_hash, component, component_pools):
                cached = candidate["scores"]
        if cached is None:
            scores = score_component(
                config,
                component,
                poses[component_id],
                component_pools,
                work_dir / component_id / "states",
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({
                "config_sha256": config_hash,
                "raw_generation_sha256": generation_hash,
                "component": component,
                "scores": scores,
            }, indent=2) + "\n", encoding="ascii")
            cached = scores
        component_scores.append(cached)
        print(f"ECLS scoring complete: {component_id}", flush=True)

    # 汇总阶段：对每个"方法 x 组分"做 leave-one-conformer-out（留一构象），
    # 用天然 H3 增益作为参照，按 config 指定的聚合方式产出统计行。
    loo_rows = []
    aggregation = config["primary"]["ensemble_aggregation"]
    for component in component_scores:
        native = np.asarray(component["native_epitope_conditioning_gain"], dtype=np.float64)
        for method_row in component["methods"]:
            matrix = np.asarray(method_row["epitope_conditioning_gain"], dtype=np.float64)
            loo_rows.extend(leave_one_out(
                component["component_id"],
                method_row["method"],
                method_row["sequences"],
                matrix,
                native,
                aggregation,
            ))
    component_ids = [row["component_id"] for row in components]
    method_summary = summarize(
        loo_rows, methods, component_ids, config["statistics"], config["development_gates"]
    )
    # 最终输出：带 provenance（全部输入哈希）与 claim_boundary（结论边界声明）。
    output = {
        "schema_version": 1,
        "status": (
            "all_generator_development_gates_passed"
            if all(row["passed"] for row in method_summary.values())
            else "development_gate_failed"
        ),
        "classification": config["classification"],
        "provenance": {
            "config": args.config,
            "config_sha256": config_hash,
            "inputs": {name: sha256(path) for name, path in paths.items()},
            "checkpoint_sha256": sha256(checkpoint),
        },
        "method_summary": method_summary,
        "leave_one_conformer_out": loo_rows,
        "component_scores": component_scores,
        "claim_boundary": config["claim_boundary"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps({"status": output["status"], "method_summary": method_summary}, indent=2))


if __name__ == "__main__":
    main()
