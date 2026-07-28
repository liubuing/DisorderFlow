#!/usr/bin/env python
"""Extract partial ColabFold metrics from log files when JSON/PDB is absent."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PATTERN = re.compile(
    r"(?P<model>alphafold\S+) recycle=(?P<recycle>\d+) "
    r"pLDDT=(?P<plddt>[0-9.]+) pTM=(?P<ptm>[0-9.]+) ipTM=(?P<iptm>[0-9.]+)"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Parse partial ColabFold log metrics")
    parser.add_argument("--log", required=True)
    parser.add_argument("--construct-id", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(args.log, encoding="utf-8") as f:
        for line in f:
            match = PATTERN.search(line)
            if not match:
                continue
            rows.append({
                "construct_id": args.construct_id,
                "model": match.group("model"),
                "recycle": match.group("recycle"),
                "plddt": match.group("plddt"),
                "ptm": match.group("ptm"),
                "iptm": match.group("iptm"),
                "evidence_status": "partial_log_metric_no_final_json_or_pdb",
                "log": args.log,
            })
    fields = ["construct_id", "model", "recycle", "plddt", "ptm", "iptm", "evidence_status", "log"]
    with open(out_dir / "colabfold_partial_log_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote partial ColabFold log metrics to {out_dir}")
    print(f"Parsed={len(rows)}")


if __name__ == "__main__":
    main()
