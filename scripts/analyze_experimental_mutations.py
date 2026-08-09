#!/usr/bin/env python
"""Audit the heterogeneous public antibody-IDP mutation evidence panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="publication/experimental_mutations_v1.json")
    parser.add_argument("--out", default="results/ablation/experimental_mutations_v1.json")
    args = parser.parse_args()
    input_path = ROOT / args.input
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    records = payload["records"]
    tiers = Counter(row["extraction_tier"] for row in records)
    endpoints = Counter(row["endpoint"] for row in records)
    point_mutations = [row for row in records if row["mutation_count"] == 1]
    exact_uncensored = [
        row for row in point_mutations
        if row["extraction_tier"] == "exact_numeric_text"
        and row["endpoint"] in {"KD_M", "fold_affinity_loss"}
    ]
    output = {
        "schema_version": 1,
        "status": "external_evidence_audit_complete",
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "n_records": len(records),
        "n_point_mutations": len(point_mutations),
        "n_antibodies": len({row["antibody"] for row in records}),
        "n_targets": len({row["target"] for row in records}),
        "extraction_tiers": dict(tiers),
        "endpoints": dict(endpoints),
        "n_exact_uncensored_quantitative_point_mutations": len(exact_uncensored),
        "quantitative_correlation_permitted": len(exact_uncensored) >= 10,
        "decision": (
            "Public evidence supports qualitative/censored mutation-effect checks only; "
            "the panel is too small and heterogeneous for a pooled quantitative correlation."
        ),
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
