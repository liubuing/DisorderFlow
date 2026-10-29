#!/usr/bin/env python
"""Score frozen 4HIX AF2 structures with independent PRODIGY affinity."""

# 用独立的 PRODIGY 工具对已冻结的 4HIX AF2 复合物结构批量预测结合亲和力
# (ΔG)。输入是 validate_4hix_matched_af2.py 产出的 results.json 及其同目录
# 的 pdb/；输出逐条预测、按候选(seed 聚合)与按实验臂(arm)两级汇总。
# 注意：结果只是计算打分，不构成实验亲和力结论（见输出中的 caveat 字段）。
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--af2", default=str(ROOT / "results/ablation/4hix_matched_af2_v1/results.json")
    )
    parser.add_argument("--out", default=str(ROOT / "results/ablation/4hix_prodigy_v1.json"))
    args = parser.parse_args()
    af2_path = Path(args.af2)
    af2 = json.loads(af2_path.read_text())
    pdb_dir = af2_path.parent / "pdb"
    # PRODIGY 以独立可执行文件方式调用（与当前解释器同目录下的 prodigy 可执行），
    # 避免依赖当前 Python 环境的包导入状态。
    executable = Path(sys.executable).parent / ("prodigy.exe" if os.name == "nt" else "prodigy")
    # --selection "A,B C"：同时统计 A-B 界面与 A/B-C 界面（抗体-肽 vs 抗体-受体）。
    # -q 安静模式，-np 4 使用 4 个进程。
    command = [str(executable), str(pdb_dir), "--selection", "A,B", "C", "-q", "-np", "4"]
    env = dict(os.environ)
    # 强制子进程按 UTF-8 输出，避免 Windows 控制台编码差异破坏解析。
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(command, capture_output=True, text=True, env=env, timeout=300)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-4000:])
    # 解析 PRODIGY 输出："文件名 ΔG" 两列；去掉 _model0 后缀得到候选 id。
    energies = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2:
            energies[fields[0].removesuffix("_model0")] = float(fields[1])

    # 逐条记录：保留 provenance（源 PDB 及其 sha256）以便追溯。
    rows = []
    for row in af2["results"]:
        if not row.get("pdb"):
            continue
        stem = Path(row["pdb"]).stem
        if stem not in energies:
            raise ValueError(f"Missing PRODIGY output for {stem}")
        rows.append(
            {
                "id": row["id"],
                "arm": row["arm"],
                "h3": row["h3"],
                "seed": row["seed"],
                "predicted_delta_g_kcal_mol": energies[stem],
                "source_pdb": row["pdb"],
                "source_pdb_sha256": row["pdb_sha256"],
            }
        )

    # 候选级汇总：同一候选跨多个 AF2 seed 的 ΔG，取中位数与极值。
    by_candidate = defaultdict(list)
    candidate_arm = {}
    for row in rows:
        by_candidate[row["id"]].append(row["predicted_delta_g_kcal_mol"])
        candidate_arm[row["id"]] = row["arm"]
    candidates = {
        candidate_id: {
            "arm": candidate_arm[candidate_id],
            "n_seeds": len(values),
            "median_delta_g_kcal_mol": float(np.median(values)),
            "min_delta_g_kcal_mol": float(np.min(values)),
            "max_delta_g_kcal_mol": float(np.max(values)),
        }
        for candidate_id, values in by_candidate.items()
    }
    # 臂级(arm)汇总：对候选中位数再求均值/极值，用于不同实验组之间对比。
    by_arm = defaultdict(list)
    for candidate in candidates.values():
        by_arm[candidate["arm"]].append(candidate["median_delta_g_kcal_mol"])
    arms = {
        arm: {
            "n_candidates": len(values),
            "mean_candidate_median_delta_g_kcal_mol": float(np.mean(values)),
            "min_candidate_median_delta_g_kcal_mol": float(np.min(values)),
            "max_candidate_median_delta_g_kcal_mol": float(np.max(values)),
        }
        for arm, values in by_arm.items()
    }
    # schema_version=1；status 明确标注这是独立计算打分而非实验亲和力；
    # PRODIGY 默认预测 25 摄氏度下的结合自由能。
    output = {
        "schema_version": 1,
        "status": "independent_computational_score_not_experimental_affinity",
        "prodigy_version": importlib.metadata.version("prodigy-prot"),
        "selection": "A,B C",
        "temperature_celsius": 25.0,
        "caveat": (
            "PRODIGY is applied to low-confidence single-sequence AF2 models; "
            "large seed spread prohibits an affinity claim."
        ),
        "rows": rows,
        "candidates": candidates,
        "arms": arms,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # ASCII 编码写出，保证结果文件对下游工具与 diff 审查友好。
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps(arms, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
