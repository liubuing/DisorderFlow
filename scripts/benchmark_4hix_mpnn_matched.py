#!/usr/bin/env python
"""Run ProteinMPNN on the frozen 12-residue 4HIX H3 mask."""

# 本脚本在冻结的 4HIX（抗 Aβ 抗体）结构上运行 ProteinMPNN 作为匹配基线：
# 固定 A（抗原）链与 L（轻链）及重链非 H3 区，仅重设计 H 链 96-107 的
# 12 个 H3 残基；按多个种子采样并解析输出 FASTA，统计序列唯一数与
# 相对天然 H3 的恢复率。定位为"匹配生成诊断"，不是设计验证。

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[1]
PDB_PATH = ROOT / "data/anti_abeta_refs/4HIX.pdb"
MPNN_SCRIPT = ROOT / "ProteinMPNN/protein_mpnn_run.py"
WEIGHTS = ROOT / "ProteinMPNN/vanilla_model_weights"
# 4HIX 天然 H3 序列（对应 H 链 96-107，共 12 个残基）
NATIVE_H3 = "VRYDHYSGSSDY"
# 允许重设计的 H 链位置（Chothia H3 区段，其余位置全部固定）
DESIGN_POSITIONS = list(range(96, 108))
# 默认三个采样种子（间隔 10 便于追溯，无特殊统计含义）
DEFAULT_SEEDS = [4101, 4111, 4121]
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


def sha256(path):
    # 计算文件 SHA-256 摘要，记录 PDB 输入的来源指纹
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# 从冻结 PDB 提取 A/H/L 三条链的一字母序列（跳过水与非标准残基）
def chain_sequences():
    model = PDBParser(QUIET=True).get_structure("4hix", str(PDB_PATH))[0]
    return {
        chain.id: "".join(
            AA3_TO_1[residue.resname] for residue in chain if residue.resname in AA3_TO_1
        )
        for chain in model
        if chain.id in {"A", "H", "L"}
    }


# 生成 ProteinMPNN 的固定位置清单：除 H 链 96-107 设计区外全部固定
def fixed_positions(sequences):
    return {
        "4HIX": {
            chain: [
                position
                for position in range(1, len(sequence) + 1)
                if chain != "H" or position not in DESIGN_POSITIONS
            ]
            for chain, sequence in sequences.items()
        }
    }


# 解析 ProteinMPNN 输出 FASTA：第 1 条为天然序列（校验链序为 A/H/L 且
# H3 段与冻结序列一致），其后每两行为一条采样序列（header + 序列）；
# 从重链段按设计位置切出 H3，计算相对天然 H3 的逐位恢复率。
def parse_fasta(path, seed, expected_lengths):
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    native_segments = lines[1].split("/")
    if len(native_segments) != 3 or native_segments[1][95:107] != NATIVE_H3:
        raise ValueError("ProteinMPNN FASTA chain order is not A/H/L as frozen")
    rows = []
    for index in range(2, len(lines), 2):
        header = lines[index]
        segments = lines[index + 1].split("/")
        if [len(segment) for segment in segments] != [
            expected_lengths["A"],
            expected_lengths["H"],
            expected_lengths["L"],
        ]:
            raise ValueError(f"Unexpected ProteinMPNN chain lengths: {list(map(len, segments))}")
        heavy = segments[1]
        h3 = "".join(heavy[position - 1] for position in DESIGN_POSITIONS)
        fields = {}
        for field in header[1:].split(","):
            if "=" in field:
                key, value = field.split("=", 1)
                fields[key.strip()] = value.strip()
        rows.append(
            {
                "arm": "proteinmpnn",
                "seed": seed,
                "sample_index": int(fields.get("sample", len(rows) + 1)) - 1,
                "sequence": h3,
                "native_recovery": sum(a == b for a, b in zip(h3, NATIVE_H3, strict=True))
                / len(NATIVE_H3),
                "score": float(fields.get("score", "nan")),
                "global_score": float(fields.get("global_score", "nan")),
            }
        )
    return rows


# 主入口：校验冻结 H3 掩码与天然序列一致 → 写 fixed_positions →
# 逐种子调用 ProteinMPNN 子进程（单种子超时 900 秒）→ 解析并汇总 → 写 JSON。
# v_48_020 为 ProteinMPNN 的 vanilla 预训练权重之一；温度 0.5 为常规采样强度。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--temperature", default="0.5")
    parser.add_argument("--work-dir", default=str(ROOT / "results/mpnn_workdir/4hix_matched_v1"))
    parser.add_argument("--out", default=str(ROOT / "results/ablation/4hix_mpnn_matched_v1.json"))
    args = parser.parse_args()

    sequences = chain_sequences()
    if sequences["H"][95:107] != NATIVE_H3:
        raise ValueError("Frozen H:96-107 mask does not match native 4HIX H3")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    fixed_path = work_dir / "fixed_positions.jsonl"
    fixed_path.write_text(json.dumps(fixed_positions(sequences)) + "\n", encoding="ascii")

    rows = []
    commands = []
    total_started = time.perf_counter()
    for seed in args.seeds:
        seed_dir = work_dir / f"seed_{seed}"
        fasta = seed_dir / "seqs/4HIX.fa"
        command = [
            sys.executable,
            str(MPNN_SCRIPT),
            "--pdb_path",
            PDB_PATH.resolve().as_posix(),
            "--pdb_path_chains",
            "A H L",
            "--fixed_positions_jsonl",
            fixed_path.resolve().as_posix(),
            "--path_to_model_weights",
            WEIGHTS.resolve().as_posix(),
            "--model_name",
            "v_48_020",
            "--num_seq_per_target",
            str(args.samples),
            "--batch_size",
            "1",
            "--sampling_temp",
            args.temperature,
            "--seed",
            str(seed),
            "--suppress_print",
            "1",
            "--out_folder",
            seed_dir.resolve().as_posix(),
        ]
        started = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
        if completed.returncode:
            raise RuntimeError(completed.stderr[-4000:])
        elapsed = time.perf_counter() - started
        seed_rows = parse_fasta(fasta, seed, {key: len(value) for key, value in sequences.items()})
        for row in seed_rows:
            row["seed_wall_seconds"] = elapsed
        rows.extend(seed_rows)
        commands.append(command)

    output = {
        "schema_version": 1,
        "status": "matched_generation_diagnostic_not_design_validation",
        "pdb": str(PDB_PATH.relative_to(ROOT)),
        "pdb_sha256": sha256(PDB_PATH),
        "region_spec": "H:96-107",
        "native_h3": NATIVE_H3,
        "context_chains": ["L", "A"],
        "seeds": args.seeds,
        "samples_per_seed": args.samples,
        "temperature": float(args.temperature),
        "wall_seconds": time.perf_counter() - total_started,
        "commands": commands,
        "results": rows,
        "summary": {
            "n": len(rows),
            "n_unique": len({row["sequence"] for row in rows}),
            "mean_native_recovery": sum(row["native_recovery"] for row in rows) / len(rows),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps(output["summary"], indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
