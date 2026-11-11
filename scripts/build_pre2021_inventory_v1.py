#!/usr/bin/env python3
"""Build a reproducible inventory of pre-cutoff SAbDab2 antibody entries.

This is a read-only inventory step for the controlled-replication model. It
does not extract structures, train a model, or write to any frozen release
artifact. The output is a CSV plus a manifest with input/output hashes.

Example:
    python scripts/build_pre2021_inventory_v1.py \
        --out results/controlled_replication_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "data" / "sabdab2_current" / "all-summary.csv"
DEFAULT_CUTOFF = "2021-08-02"
KEEP_COLUMNS = [
    "INSTANCE",
    "PDB",
    "SABDAB_ID",
    "HEAVY_ID",
    "LIGHT_ID",
    "Hchain",
    "Lchain",
    "antigen_chain",
    "antigen_type",
    "antigen_name",
    "date",
    "SABDABdepo_date",
    "method",
    "resolution",
    "heavy_species",
    "light_species",
    "heavy_subclass",
    "light_subclass",
    "type",
]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build(summary: Path, cutoff: str, max_resolution: float) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(summary, low_memory=False)
    frame["deposition_date"] = pd.to_datetime(
        frame["date"], format="%Y/%m/%d", errors="coerce"
    )
    frame["resolution_numeric"] = pd.to_numeric(frame["resolution"], errors="coerce")
    frame["is_peptide"] = frame["antigen_type"].fillna("").str.contains("PEPTIDE")
    frame["has_light"] = frame["Lchain"].notna()
    frame["has_antigen"] = frame["antigen_chain"].notna()

    cutoff_ts = pd.Timestamp(cutoff)
    selected = frame[
        frame["deposition_date"].notna()
        & (frame["deposition_date"] <= cutoff_ts)
        & frame["Hchain"].notna()
        & frame["resolution_numeric"].notna()
        & (frame["resolution_numeric"] <= max_resolution)
    ].copy()
    selected = selected.drop_duplicates(subset=["INSTANCE"]).reset_index(drop=True)
    selected["deposition_year"] = selected["deposition_date"].dt.year

    columns = [column for column in KEEP_COLUMNS if column in selected.columns]
    inventory = selected[columns + [
        "deposition_date",
        "deposition_year",
        "resolution_numeric",
        "is_peptide",
        "has_light",
        "has_antigen",
    ]]

    manifest = {
        "schema_version": 1,
        "created": date.today().isoformat(),
        "source": str(summary.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": sha256(summary),
        "cutoff_inclusive": cutoff,
        "max_resolution_a": max_resolution,
        "filters": [
            "deposition_date <= cutoff",
            "Hchain present",
            "resolution present and <= max_resolution",
            "deduplicate by INSTANCE",
        ],
        "counts": {
            "source_rows": int(len(frame)),
            "selected_rows": int(len(inventory)),
            "unique_pdb": int(inventory["PDB"].nunique()),
            "peptide_rows": int(inventory["is_peptide"].sum()),
            "peptide_unique_pdb": int(
                inventory.loc[inventory["is_peptide"], "PDB"].nunique()
            ),
            "no_light_chain_rows": int((~inventory["has_light"]).sum()),
            "with_antigen_rows": int(inventory["has_antigen"].sum()),
            "xray_rows": int((inventory["method"] == "XRAY").sum()),
        },
    }
    return inventory, manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    parser.add_argument("--max-resolution", type=float, default=3.5)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "controlled_replication_v1",
    )
    args = parser.parse_args(argv)

    summary = args.summary if args.summary.is_absolute() else ROOT / args.summary
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    if not summary.exists():
        raise FileNotFoundError(summary)

    inventory, manifest = build(summary, args.cutoff, args.max_resolution)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "pre2021_inventory.csv"
    inventory.to_csv(csv_path, index=False)
    manifest["output"] = str(csv_path.relative_to(ROOT)).replace("\\", "/")
    manifest["output_sha256"] = sha256(csv_path)
    manifest_path = out_dir / "pre2021_inventory_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(json.dumps(manifest["counts"], indent=2))
    print(f"wrote {csv_path} ({csv_path.stat().st_size:,} bytes)")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
