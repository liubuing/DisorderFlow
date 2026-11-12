#!/usr/bin/env python
"""Analyze AntiFold 4HIX H3 designs and compare with ProteinMPNN shortlist.

AntiFold output: results/hf_aux_benchmark_v1/antifold_4hix/run100/4HIX_HLA.fasta
H/L are separated by '/'; H3 is H[95:107] (native VRYDHYSGSSDY).

Developability filters mirror the MPNN candidate package:
  no Cys, no N-X-S/T, homopolymer <=3, H3 charge in [-4,+6], Pro <=3.
Selection uses AntiFold score + diversity, not ECLS/BFN affinity scores.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FASTA = ROOT / "results/hf_aux_benchmark_v1/antifold_4hix/run100/4HIX_HLA.fasta"
MPNN = ROOT / "results/design_v1_20261101/shortlist.csv"
OUT = ROOT / "results/hf_aux_benchmark_v1/antifold_4hix"
NATIVE = "VRYDHYSGSSDY"
H3_START = 95


def parse_fasta(path: Path):
    records = []
    header = None
    chunks = []
    for line in path.open(encoding="ascii"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header = line[1:]
            chunks = []
        else:
            chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def header_value(header: str, key: str):
    match = re.search(rf"{re.escape(key)}=([^,]+)", header)
    return match.group(1).strip() if match else None


def dev(h3: str) -> dict:
    cys = "C" in h3
    glyco = bool(re.search(r"N[^P][ST]", h3))
    max_run = max((len(m.group(0)) for m in re.finditer(r"(.)\1*", h3)), default=0)
    charge = sum(1 for aa in h3 if aa in "KR") - sum(1 for aa in h3 if aa in "DE")
    pro = h3.count("P")
    hydrophobic = sum(1 for aa in h3 if aa in "AVILMFWY") / max(1, len(h3))
    passed = (not cys and not glyco and max_run <= 3
              and -4 <= charge <= 6 and pro <= 3)
    return {
        "cys": cys, "glyco": glyco, "max_run": max_run,
        "charge": charge, "pro": pro, "hydrophobic": hydrophobic,
        "dev_pass": passed,
    }


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))


def main():
    records = parse_fasta(FASTA)
    rows = []
    for header, sequence in records:
        if "/" not in sequence:
            continue
        heavy, light = sequence.split("/", 1)
        if len(heavy) < H3_START + 12:
            continue
        h3 = heavy[H3_START:H3_START + 12]
        score = header_value(header, "score")
        mutations = header_value(header, "mutations") or ""
        rows.append({
            "source": "AntiFold",
            "header": header,
            "h3": h3,
            "af_score": float(score) if score else None,
            "n_mut": hamming(h3, NATIVE),
            "native_recovery": 1 - hamming(h3, NATIVE) / len(NATIVE),
            "mutations": mutations,
            "heavy": heavy,
            "light": light,
            **dev(h3),
        })
    frame = pd.DataFrame(rows)
    # exclude the WT record (no mutations) from design counts
    designs = frame[frame["n_mut"] > 0].copy()
    print("records", len(frame), "designs", len(designs))
    print("unique H3", designs["h3"].nunique())
    print("dev pass", designs["dev_pass"].mean())
    print("mean n_mut", designs["n_mut"].mean(), "range", designs["n_mut"].min(), designs["n_mut"].max())

    # MPNN comparison
    mpnn = pd.read_csv(MPNN)
    mpnn = mpnn[mpnn["design_id"] != "4HIX-native"].copy()
    mpnn["n_mut"] = mpnn["h3"].map(lambda h3: hamming(h3, NATIVE))
    mpnn_dev = pd.DataFrame([dev(h3) for h3 in mpnn["h3"]])
    mpnn = pd.concat([mpnn.reset_index(drop=True), mpnn_dev], axis=1)
    print("\nMPNN shortlist", len(mpnn), "unique", mpnn["h3"].nunique(),
          "dev pass", mpnn["dev_pass"].mean(), "mean n_mut", mpnn["n_mut"].mean())

    # select AntiFold candidates: dev pass, unique H3, score descending, greedy diversity
    candidates = designs[designs["dev_pass"]].drop_duplicates("h3").sort_values(
        "af_score", ascending=False)
    selected = []
    for _, row in candidates.iterrows():
        if all(hamming(row["h3"], other["h3"]) >= 3 for other in selected):
            selected.append(row)
        if len(selected) >= 10:
            break
    selected = pd.DataFrame(selected)
    if len(selected):
        selected = selected.reset_index(drop=True)
        selected["design_id"] = [f"4HIX-AF{i+1:03d}" for i in range(len(selected))]
    selected.to_csv(OUT / "antifold_candidates.csv", index=False)
    frame.to_csv(OUT / "antifold_all_scores.csv", index=False)

    summary = {
        "antifold_designs": int(len(designs)),
        "antifold_unique_h3": int(designs["h3"].nunique()),
        "antifold_dev_pass_fraction": float(designs["dev_pass"].mean()),
        "antifold_mean_n_mut": float(designs["n_mut"].mean()),
        "antifold_mutated_positions": sorted(set(
            int(m.group(1)) for m in re.finditer(r"H:(\d+)", " ".join(designs["mutations"])))),
        "mpnn_shortlist_n": int(len(mpnn)),
        "mpnn_unique_h3": int(mpnn["h3"].nunique()),
        "mpnn_dev_pass_fraction": float(mpnn["dev_pass"].mean()),
        "mpnn_mean_n_mut": float(mpnn["n_mut"].mean()),
        "selected": selected[["design_id", "h3", "af_score", "n_mut"]].to_dict(orient="records")
        if len(selected) else [],
        "claim_boundary": "generator comparison only; no binding/affinity claim",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
