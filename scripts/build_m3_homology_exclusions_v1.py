#!/usr/bin/env python3
"""Find training PDBs homologous to the frozen temporal panel.

Two criteria:
1. a panel H3 exact or 1-mismatch window anywhere in a training chain;
2. full-VH 3-mer cosine similarity >= threshold to any panel VH.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_h3_epitope_delta import read_record  # noqa: E402

PANEL = ROOT / "results" / "publication" / "h3_ecls_temporal_final_v1" / "results.json"
LMDB = ROOT / "data" / "peptide_h3_temporal_split_v3" / "sealed_final.lmdb"
TRAIN = ROOT / "data" / "controlled_replication_v1" / "m3_dataset_full" / "list.csv"
OUT = ROOT / "results" / "ecls_factorial_v1" / "homology_exclusions.json"
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_CODE = {aa: index for index, aa in enumerate(AA)}


def mismatches_leq_one(window: str, target: str) -> bool:
    if len(window) != len(target):
        return False
    differences = 0
    for first, second in zip(window, target):
        if first != second:
            differences += 1
            if differences > 1:
                return False
    return True


def kmer_vector(sequence: str, k: int = 3) -> np.ndarray:
    codes = []
    for start in range(len(sequence) - k + 1):
        code = 0
        valid = True
        for char in sequence[start:start + k]:
            if char not in AA_CODE:
                valid = False
                break
            code = code * 20 + AA_CODE[char]
        if valid:
            codes.append(code)
    vector = np.bincount(codes, minlength=20 ** k).astype(np.float32)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--lmdb", type=Path, default=LMDB)
    parser.add_argument("--train", type=Path, default=TRAIN)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--vh-cosine", type=float, default=0.90)
    args = parser.parse_args(argv)

    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    panel_h3 = sorted({record["native"]["sequence"] for record in panel["results"]})
    panel_vh = []
    for record in panel["results"]:
        vh = read_record(args.lmdb, record["id"]).get("vh_sequence")
        if vh:
            panel_vh.append(vh)
    panel_vh = sorted(set(panel_vh))
    panel_vectors = np.stack([kmer_vector(vh) for vh in panel_vh])

    train = pd.read_csv(args.train)
    by_pdb = {}
    for chain_id, sequence in zip(train["CHAINID"], train["SEQUENCE"]):
        by_pdb.setdefault(chain_id.split("_")[0], []).append((chain_id, sequence))

    excluded = {}
    for pdb, chains in by_pdb.items():
        reason = None
        for chain_id, sequence in chains:
            for h3 in panel_h3:
                length = len(h3)
                for start in range(0, len(sequence) - length + 1):
                    window = sequence[start:start + length]
                    if mismatches_leq_one(window, h3):
                        reason = {
                            "criterion": "h3_exact_or_1mm",
                            "chain": chain_id,
                            "h3": h3,
                            "window": window,
                            "start": start,
                            "exact": window == h3,
                        }
                        break
                if reason:
                    break
            if reason:
                break
        if reason:
            excluded[pdb] = reason
            continue
        best = 0.0
        best_chain = None
        for chain_id, sequence in chains:
            vector = kmer_vector(sequence)
            similarity = float(np.max(panel_vectors @ vector))
            if similarity > best:
                best = similarity
                best_chain = chain_id
        if best >= args.vh_cosine:
            excluded[pdb] = {
                "criterion": "vh_3mer_cosine",
                "chain": best_chain,
                "cosine": best,
            }

    payload = {
        "schema_version": 1,
        "panel_h3": panel_h3,
        "n_panel_vh": len(panel_vh),
        "vh_cosine_threshold": args.vh_cosine,
        "n_training_pdbs": len(by_pdb),
        "n_excluded": len(excluded),
        "exclusion_rule": (
            "panel H3 exact/1-mismatch window OR full-VH 3-mer cosine "
            f">= {args.vh_cosine}"
        ),
        "excluded": excluded,
    }
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    criteria = {}
    for item in excluded.values():
        criteria[item["criterion"]] = criteria.get(item["criterion"], 0) + 1
    print(json.dumps({
        "n_training_pdbs": len(by_pdb),
        "n_excluded": len(excluded),
        "criteria": criteria,
        "vh_cosine_threshold": args.vh_cosine,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
