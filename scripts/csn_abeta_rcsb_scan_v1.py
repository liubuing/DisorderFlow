#!/usr/bin/env python3
"""Scan the full PDB sequence archive for DAEFRH-containing antibody entries.

Two passes over pdb_seqres.txt.gz:
1. collect PDB IDs with a chain containing DAEFRH;
2. collect all chain headers for those PDBs and flag antibody keywords.

This is a heuristic metadata scan, not a structural validation. New candidates
must be inspected manually before being added to the cohort.
"""
from __future__ import annotations

import gzip
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEQRES = ROOT / "data" / "rcsb_derived" / "pdb_seqres.txt.gz"
SUMMARY = ROOT / "data" / "sabdab2_current" / "all-summary.csv"
OUT = ROOT / "results" / "csn_abeta_v1"
TARGET = "DAEFRH"
ANTIBODY_RE = re.compile(
    r"(?i)(\bfab\b|antibody|immunoglobulin|\bnanobody\b|\bvhh\b|\bscfv\b|"
    r"single[- ]chain|\bigg\b|\bmab\b|light chain|heavy chain|"
    r"variable domain|fv fragment)"
)


def iter_records(path: Path):
    with gzip.open(path, "rt", errors="replace") as handle:
        header = None
        sequence = []
        for line in handle:
            if line.startswith(">"):
                if header is not None and sequence:
                    yield header.strip(), "".join(sequence).strip()
                header = line
                sequence = []
            else:
                sequence.append(line.strip())
        if header is not None and sequence:
            yield header.strip(), "".join(sequence).strip()


def pdb_id(header: str) -> str:
    return header.split()[0].lstrip(">").split("_")[0].upper()


def main() -> int:
    targets = {
        pdb_id(header)
        for header, sequence in iter_records(SEQRES)
        if TARGET in sequence
    }
    print(f"PDBs with {TARGET}: {len(targets)}")

    chains = {}
    for header, sequence in iter_records(SEQRES):
        if pdb_id(header) not in targets:
            continue
        chains.setdefault(pdb_id(header), []).append({
            "header": header,
            "length": len(sequence),
            "sequence": sequence,
            "contains_target": TARGET in sequence,
        })

    summary = pd.read_csv(SUMMARY, low_memory=False)
    summary["pid4"] = summary["PDB"].str[-4:].str.upper()
    sabdab = set(summary["pid4"])

    entries = []
    for pdb, chain_list in chains.items():
        antibody_chains = [
            chain for chain in chain_list
            if ANTIBODY_RE.search(chain["header"])
        ]
        daefrh_chains = [chain for chain in chain_list if chain["contains_target"]]
        entries.append({
            "pdb": pdb,
            "in_sabdab": pdb in sabdab,
            "antibody_flagged": bool(antibody_chains),
            "antibody_chain_headers": [chain["header"] for chain in antibody_chains],
            "daefrh_chain_headers": [chain["header"] for chain in daefrh_chains],
        })
    entries.sort(key=lambda item: item["pdb"])
    new_candidates = [
        entry for entry in entries
        if entry["antibody_flagged"] and not entry["in_sabdab"]
    ]
    payload = {
        "schema_version": 1,
        "created": date.today().isoformat(),
        "analysis": "full_pdb_metadata_scan",
        "classification": "heuristic_not_confirmatory",
        "target": TARGET,
        "counts": {
            "pdbs_with_target": len(entries),
            "antibody_flagged": sum(1 for entry in entries if entry["antibody_flagged"]),
            "in_sabdab": sum(1 for entry in entries if entry["in_sabdab"]),
            "new_antibody_candidates": len(new_candidates),
        },
        "new_candidates": new_candidates,
        "entries": entries,
        "caveat": (
            "Antibody detection is keyword-based on pdb_seqres headers. "
            "Every new candidate must be inspected structurally."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "rcsb_scan.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame(entries).to_csv(OUT / "rcsb_scan.csv", index=False)
    print(json.dumps(payload["counts"], indent=2))
    print("new candidates:")
    for entry in new_candidates:
        print(" ", entry["pdb"], entry["daefrh_chain_headers"][:2])
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
