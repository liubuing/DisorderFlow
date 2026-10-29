#!/usr/bin/env python
"""Summarize matched 4HIX generation, AF2 uncertainty, and developability."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# 4HIX 抗体的天然 H3 环序列（12 aa），作为 native recovery / Hamming 比较的基准。
NATIVE_H3 = "VRYDHYSGSSDY"
AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def chain_sequences():
    # 从 data/anti_abeta_refs/4HIX.pdb 提取三条链的一级序列：
    # A = 抗原（Aβ），H = 抗体重链，L = 抗体轻链；非标准残基直接跳过。
    path = ROOT / "data/anti_abeta_refs/4HIX.pdb"
    model = PDBParser(QUIET=True).get_structure("4hix", str(path))[0]
    return {
        chain.id: "".join(
            AA3_TO_1[residue.resname] for residue in chain if residue.resname in AA3_TO_1
        )
        for chain in model
        if chain.id in {"A", "H", "L"}
    }


def hamming(left, right):
    # 两条 H3 序列必须等长（固定 scaffold 上逐位替换），否则直接报错而非截断，
    # 以保证配对比较的严谨性；返回差异位置比例（0~1）。
    if len(left) != len(right):
        raise ValueError("Cannot compare unequal H3 sequences")
    return sum(a != b for a, b in zip(left, right, strict=True)) / len(left)


def generation_summary(bfn, mpnn):
    # 汇总各生成臂（BFN 消融各臂 + ProteinMPNN 基线）的样本数、去重后序列数、
    # native recovery 均值与耗时；并按 (seed, sample_index) 配对
    # current_disorder_on / off 两臂，计算 Hamming 差异以衡量条件化的响应度。
    arms = {arm["name"]: arm["results"] for arm in bfn["arms"]}
    arms["proteinmpnn"] = mpnn["results"]
    output = {}
    for name, rows in arms.items():
        output[name] = {
            "n": len(rows),
            "n_unique": len({row["sequence"] for row in rows}),
            "mean_native_recovery": float(np.mean([row["native_recovery"] for row in rows])),
            "wall_seconds": (
                next(arm["wall_seconds"] for arm in bfn["arms"] if arm["name"] == name)
                if name != "proteinmpnn"
                else mpnn["wall_seconds"]
            ),
        }
    keyed_on = {(row["seed"], row["sample_index"]): row for row in arms["current_disorder_on"]}
    keyed_off = {(row["seed"], row["sample_index"]): row for row in arms["current_disorder_off"]}
    differences = [
        hamming(keyed_on[key]["sequence"], keyed_off[key]["sequence"]) for key in sorted(keyed_on)
    ]
    output["current_on_vs_off_paired"] = {
        "n_pairs": len(differences),
        "mean_hamming_fraction": float(np.mean(differences)),
        "identical_fraction": float(np.mean(np.asarray(differences) == 0)),
        "pairs_changed": int(np.sum(np.asarray(differences) > 0)),
    }
    return output, arms


def af2_summary(data):
    # 汇总 AF2 预测结果：按候选 id 分组（同一候选多次预测），统计 ipTM
    # （越高越好）/ 界面 PAE（越低越好）/ pLDDT 的中位数与极值，再按臂聚合。
    grouped = defaultdict(list)
    metadata = {}
    for row in data["results"]:
        if row["success"]:
            grouped[row["id"]].append(row)
            metadata[row["id"]] = {key: row[key] for key in ("arm", "h3", "pre_af2_rank")}
    candidates = {}
    by_arm = defaultdict(list)
    for candidate_id, rows in grouped.items():
        iptm = [row["iptm"] for row in rows]
        summary = {
            **metadata[candidate_id],
            "n_success": len(rows),
            "median_iptm": float(np.median(iptm)),
            "min_iptm": float(np.min(iptm)),
            "max_iptm": float(np.max(iptm)),
            "median_interface_pae": float(np.median([row["interface_pae"] for row in rows])),
            "median_plddt": float(np.median([row["plddt"] for row in rows])),
        }
        candidates[candidate_id] = summary
        by_arm[summary["arm"]].append(summary["median_iptm"])
    arms = {
        arm: {
            "n_candidates": len(values),
            "mean_candidate_median_iptm": float(np.mean(values)),
            "min_candidate_median_iptm": float(np.min(values)),
            "max_candidate_median_iptm": float(np.max(values)),
        }
        for arm, values in by_arm.items()
    }
    return {"candidates": candidates, "arms": arms}


def developability_summary(arms):
    # 可开发性评估：把每条生成的 H3 拼回重链骨架（H 链前 95 位 + H3 + 第 107 位起，
    # 对应 H3 区域的编号边界），与轻链风险分取最大值；
    # risk <= 0.55 视为可开发性通过的经验阈值。
    from scripts.pipeline.score_shortlist_developability import score_sequence

    sequences = chain_sequences()
    light_score = score_sequence(sequences["L"])
    output = {}
    for arm, rows in arms.items():
        risks = []
        passes = 0
        flags = defaultdict(int)
        for row in rows:
            heavy = sequences["H"][:95] + row["sequence"] + sequences["H"][107:]
            heavy_score = score_sequence(heavy)
            risk = max(heavy_score["sequence_risk"], light_score["sequence_risk"])
            risks.append(risk)
            passes += risk <= 0.55
            for flag in heavy_score["flags"].split(";"):
                if flag:
                    flags[flag] += 1
        output[arm] = {
            "n": len(rows),
            "pass_fraction_at_0_55": passes / len(rows),
            "mean_combined_risk": float(np.mean(risks)),
            "candidate_heavy_flags": dict(sorted(flags.items())),
        }
    return output


def main():
    # 汇总入口：读入 BFN 消融、ProteinMPNN 基线、AF2 评估三份 JSON，
    # 输出统一摘要；status 字段明确声明这是"单 scaffold 诊断"而非设计验证，
    # claim_boundary 界定了不可过度外推的结论范围。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfn", default=str(ROOT / "results/ablation/4hix_bfn_matched_v1.json"))
    parser.add_argument("--mpnn", default=str(ROOT / "results/ablation/4hix_mpnn_matched_v1.json"))
    parser.add_argument(
        "--af2", default=str(ROOT / "results/ablation/4hix_matched_af2_v1/results.json")
    )
    parser.add_argument(
        "--out", default=str(ROOT / "results/ablation/4hix_matched_summary_v1.json")
    )
    args = parser.parse_args()
    bfn = json.loads(Path(args.bfn).read_text())
    mpnn = json.loads(Path(args.mpnn).read_text())
    af2 = json.loads(Path(args.af2).read_text())
    generation, arms = generation_summary(bfn, mpnn)
    output = {
        "schema_version": 1,
        "status": "single_scaffold_diagnostic_not_design_validation",
        "generation": generation,
        "af2": af2_summary(af2),
        "developability": developability_summary(arms),
        "claim_boundary": (
            "One short-peptide scaffold cannot establish BFN design success, "
            "IDP preference, affinity, or external generalization."
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
