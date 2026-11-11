#!/usr/bin/env python3
"""Antigen-context and peptide-identity ablation for ECLS models.

For each frozen temporal record, keep the H/L/peptide coordinates fixed and
replace the peptide sequence with a composition-matched shuffle. Compare the
native H3 NLL in three contexts:

  complex        native peptide coordinates + native peptide sequence
  apo            no peptide
  antigen-shuf   native peptide coordinates + shuffled peptide sequence

Derived quantities:
  delta_context = NLL(apo) - NLL(complex)      >0 means peptide presence helps
  delta_peptide = NLL(antigen-shuf) - NLL(complex) >0 means native peptide identity helps

Native complex/apo NLLs are read from the existing model summaries; only the
antigen-shuffled condition is computed here.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_h3_epitope_delta import (  # noqa: E402
    ALPHABET,
    chain_sequence,
    read_record,
    run_mpnn_state,
    sequence_nll,
    sha256_text,
    write_record_backbone,
)
from benchmark_h3_scorer_sanity import composition_shuffles  # noqa: E402
from ecls_antifold_zero_shot_v1 import logp_matrix, run_antifold  # noqa: E402

MODELS = {
    "M0": {
        "summary": ROOT / "results" / "publication" / "h3_ecls_temporal_final_v1" / "results.json",
        "config": ROOT / "configs" / "benchmarks" / "peptide_h3_ecls_temporal_final_v1.yml",
    },
    "M2": {
        "summary": ROOT / "results" / "ecls_cross_arch_v1" / "antifold" / "antifold_results.json",
        "config": None,
    },
    "M3": {
        "summary": ROOT / "results" / "ecls_cross_arch_v1" / "m3_full10_ecls" / "results.json",
        "config": ROOT / "configs" / "benchmarks" / "peptide_h3_ecls_m3_full10_v1.yml",
    },
}


def load_native(summary_path: Path):
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    records = data["results"] if isinstance(data, dict) and "results" in data else data
    return {
        record["id"]: {
            "complex": float(record["native"]["complex_h3_nll"]),
            "apo": float(record["native"]["apo_h3_nll"]),
        }
        for record in records
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs" / "benchmarks" / "peptide_h3_ecls_temporal_final_v1.yml")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results" / "ecls_factorial_v1")
    parser.add_argument("--shuffles", type=int, default=3)
    parser.add_argument("--models", default="M0,M3,M2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args(argv)

    base_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    audit = json.loads((ROOT / base_config["split_audit"]).read_text(encoding="utf-8"))
    records = audit["records"][base_config["benchmark_split"]]
    if args.max_records:
        records = records[:args.max_records]
    lmdb_path = ROOT / base_config["lmdb"]
    models = [name.strip() for name in args.models.split(",") if name.strip()]
    native = {name: load_native(MODELS[name]["summary"]) for name in models}
    configs = {}
    for name in models:
        if MODELS[name]["config"] is not None:
            configs[name] = yaml.safe_load(
                MODELS[name]["config"].read_text(encoding="utf-8"))

    out = args.out if args.out.is_absolute() else ROOT / args.out
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, record in enumerate(records, 1):
        h3 = record["axis_values"]["cdr_h3_sequence_exact"][0]
        h3_indices = record["h3_coordinate_mapping"]["parsed_h3_indices"]
        data = read_record(lmdb_path, record["id"])
        peptide = chain_sequence(data["antigen"])
        shuffles = composition_shuffles(
            peptide, args.shuffles, 20261111 + sum(ord(char) for char in record["id"]))
        print(f"[{index}/{len(records)}] {record['id']} peptide={peptide}", flush=True)
        for model in models:
            for shuffle_index, shuffled in enumerate(shuffles):
                shuffled_record = copy.deepcopy(data)
                shuffled_record["id"] = f"{record['id']}_agshuf{shuffle_index}"
                shuffled_record["antigen"]["aa"] = [
                    ALPHABET.index(char) for char in shuffled]
                if model in {"M0", "M3"}:
                    log_probs, _manifest, _command = run_mpnn_state(
                        configs[model], shuffled_record, h3_indices, "complex",
                        work / model)
                    nll = sequence_nll(log_probs, h3, h3_indices)
                else:
                    pdb_path = work / model / shuffled_record["id"] / "complex.pdb"
                    write_record_backbone(shuffled_record, pdb_path, True)
                    frame = run_antifold(
                        pdb_path, pdb_path.parent / "antifold", True, args.device)
                    logp = logp_matrix(frame, h3_indices, h3)
                    nll = -sum(
                        float(logp[offset, ALPHABET.index(aa)])
                        for offset, aa in enumerate(h3)) / len(h3)
                rows.append({
                    "record": record["id"],
                    "model": model,
                    "shuffle": shuffle_index,
                    "native_complex_nll": native[model][record["id"]]["complex"],
                    "native_apo_nll": native[model][record["id"]]["apo"],
                    "antigen_shuffled_nll": float(nll),
                })
    frame = pd.DataFrame(rows)
    per_record = frame.groupby(["record", "model"], as_index=False).agg(
        native_complex_nll=("native_complex_nll", "first"),
        native_apo_nll=("native_apo_nll", "first"),
        antigen_shuffled_nll=("antigen_shuffled_nll", "mean"),
    )
    per_record["delta_context"] = (
        per_record["native_apo_nll"] - per_record["native_complex_nll"])
    per_record["delta_peptide"] = (
        per_record["antigen_shuffled_nll"] - per_record["native_complex_nll"])
    summary = {}
    for model in models:
        subset = per_record[per_record["model"] == model]
        summary[model] = {
            "n": int(len(subset)),
            "mean_delta_context": float(subset["delta_context"].mean()),
            "median_delta_context": float(subset["delta_context"].median()),
            "fraction_context_positive": float((subset["delta_context"] > 0).mean()),
            "mean_delta_peptide": float(subset["delta_peptide"].mean()),
            "median_delta_peptide": float(subset["delta_peptide"].median()),
            "fraction_peptide_positive": float((subset["delta_peptide"] > 0).mean()),
            "peptide_to_context_ratio": float(
                subset["delta_peptide"].mean() / subset["delta_context"].mean())
            if subset["delta_context"].mean() != 0 else None,
        }
    out.mkdir(parents=True, exist_ok=True)
    per_record.to_csv(out / "per_record.csv", index=False)
    payload = {
        "schema_version": 1,
        "status": "development_descriptive_not_confirmatory",
        "shuffles_per_record": args.shuffles,
        "models": models,
        "summary": summary,
        "note": (
            "complex = native peptide; apo = no peptide; antigen-shuf = same "
            "coordinates, composition-matched peptide sequence. Native "
            "complex/apo NLLs are read from the model summaries."
        ),
        "per_record_sha256": sha256_text((out / "per_record.csv").read_text(encoding="utf-8")),
    }
    (out / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
