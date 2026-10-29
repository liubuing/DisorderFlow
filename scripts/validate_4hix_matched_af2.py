#!/usr/bin/env python
"""AF2 multi-seed validation for pre-AF2-selected matched 4HIX candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# 固定的参考结构（4HIX：抗淀粉样 beta 抗体）与 AF2 前匹配筛选产物路径。
PDB_PATH = ROOT / "data/anti_abeta_refs/4HIX.pdb"
BFN_RESULTS = ROOT / "results/ablation/4hix_bfn_matched_v1.json"
MPNN_RESULTS = ROOT / "results/ablation/4hix_mpnn_matched_v1.json"
# 天然 H3 序列与 AF2 的多 seed 设定（seed 间隔取 10 避免相邻随机流相关）。
NATIVE_H3 = "VRYDHYSGSSDY"
AF2_SEEDS = [5101, 5111, 5121]
# 三字母 -> 单字母氨基酸映射（PDB resname 解析用）。
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
    # 解析 4HIX 的 A(抗原)/H(重链)/L(轻链) 三条链的一字母序列。
    model = PDBParser(QUIET=True).get_structure("4hix", str(PDB_PATH))[0]
    return {
        chain.id: "".join(
            AA3_TO_1[residue.resname] for residue in chain if residue.resname in AA3_TO_1
        )
        for chain in model
        if chain.id in {"A", "H", "L"}
    }


def selected_candidates(top_n):
    # 汇集 AF2 前已冻结的候选集（选择规则在 AF2 之前确定，避免事后挑选）：
    # - BFN 各实验臂按 ppl 升序取 top_n；
    # - ProteinMPNN 按 score 升序取 top_n；
    # - 两个阴性对照：天然 H3、以及固定 seed 洗牌的组分打乱对照。
    bfn = json.loads(BFN_RESULTS.read_text())
    mpnn = json.loads(MPNN_RESULTS.read_text())
    selected = []
    for arm in bfn["arms"]:
        rows = sorted(
            arm["results"], key=lambda row: (row["ppl"], row["seed"], row["sample_index"])
        )
        for rank, row in enumerate(rows[:top_n], 1):
            selected.append(
                {
                    "id": f"{arm['name']}_r{rank}",
                    "arm": arm["name"],
                    "h3": row["sequence"],
                    "pre_af2_rank": rank,
                    "selection_metric": "ppl",
                    "selection_value": row["ppl"],
                }
            )
    rows = sorted(mpnn["results"], key=lambda row: (row["score"], row["seed"], row["sample_index"]))
    for rank, row in enumerate(rows[:top_n], 1):
        selected.append(
            {
                "id": f"proteinmpnn_r{rank}",
                "arm": "proteinmpnn",
                "h3": row["sequence"],
                "pre_af2_rank": rank,
                "selection_metric": "proteinmpnn_score",
                "selection_value": row["score"],
            }
        )
    selected.append(
        {
            "id": "native",
            "arm": "native",
            "h3": NATIVE_H3,
            "pre_af2_rank": 1,
            "selection_metric": "control",
            "selection_value": None,
        }
    )
    # 组分对照：保留氨基酸组成但打乱排列（seed 固定保证可复现）。
    shuffled = list(NATIVE_H3)
    random.Random(5303).shuffle(shuffled)
    selected.append(
        {
            "id": "composition_shuffle",
            "arm": "shuffle",
            "h3": "".join(shuffled),
            "pre_af2_rank": 1,
            "selection_metric": "control",
            "selection_value": None,
        }
    )
    return selected


def full_heavy(native_heavy, h3):
    # 把候选 H3 嫁接回天然重链骨架：H3 位于 95..107（0-based 半开区间），
    # 长度必须与天然 H3 一致才能原位替换。
    if len(h3) != len(NATIVE_H3):
        raise ValueError(f"Invalid H3 length: {h3}")
    return native_heavy[:95] + h3 + native_heavy[107:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=AF2_SEEDS)
    parser.add_argument("--recycles", type=int, default=3)
    parser.add_argument("--out-dir", default=str(ROOT / "results/ablation/4hix_matched_af2_v1"))
    args = parser.parse_args()

    # 延迟导入 AF2 runner（导入开销大）。
    from modules.af2_jax_runner import run_multimer_prediction

    sequences = chain_sequences()
    # 完整性哨兵：4HIX 的 H3 区与抗原肽序列必须与预期一致，
    # 否则说明参考结构被改动，继续运行没有意义。
    if sequences["H"][95:107] != NATIVE_H3 or sequences["A"] != "DAEFRH":
        raise ValueError("4HIX sequence or H3 mapping changed")
    candidates = selected_candidates(args.top_n)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    pdb_dir = out_dir / "pdb"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for candidate in candidates:
        # 重链(带候选 H3)+轻链为一条多聚体输入，抗原肽为第二条。
        heavy = full_heavy(sequences["H"], candidate["h3"])
        for seed in args.seeds:
            result = run_multimer_prediction(
                f"{heavy}:{sequences['L']}",
                sequences["A"],
                num_recycle=args.recycles,
                jax_random_seed=seed,
                return_structure=True,
            )
            row = {
                **candidate,
                "seed": seed,
                "success": bool(result.get("success")),
                "elapsed": result.get("elapsed"),
                "plddt": result.get("plddt"),
                "ptm": result.get("ptm"),
                "iptm": result.get("iptm"),
                "interface_pae": result.get("interface_pae"),
                "max_pae": result.get("max_pae"),
                "error": result.get("error"),
            }
            if result.get("success") and result.get("pdb"):
                # 成功的预测结构落盘并记录 sha256 溯源。
                pdb_path = pdb_dir / f"{candidate['id']}_seed{seed}.pdb"
                pdb_path.write_text(result["pdb"], encoding="ascii")
                row["pdb"] = str(pdb_path.relative_to(ROOT))
                row["pdb_sha256"] = hashlib.sha256(pdb_path.read_bytes()).hexdigest()
            rows.append(row)
            print(candidate["id"], seed, row["success"], row["iptm"], flush=True)

    # 输出明确声明：这是"单脚手架多 seed 诊断"，不构成结合验证。
    output = {
        "schema_version": 1,
        "status": "single_scaffold_multi_seed_diagnostic_not_binding_validation",
        "selection_frozen_before_af2": True,
        "selection_rule": "top_n_within_arm_by_pre_af2_native_score",
        "top_n": args.top_n,
        "seeds": args.seeds,
        "recycles": args.recycles,
        "candidates": candidates,
        "results": rows,
    }
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
