#!/usr/bin/env python
"""ANARCI/IMGT H3 numbering for the public peptide-Kd cohort (WSL)."""
from __future__ import annotations

import json
import os
from pathlib import Path

from anarci import run_anarci

ROOT = Path(os.environ.get("DISORDERFLOW_ROOT") or Path(__file__).resolve().parents[2])
FASTA = ROOT / "results/hf_aux_benchmark_v1/public_peptide_kd/heavy_sequences.fasta"
OUT = ROOT / "results/hf_aux_benchmark_v1/public_peptide_kd/anarci_h3.json"
HMMER = "/tmp/hmmer_pkg/usr/bin"


def read_fasta(path: Path):
    records = []
    name = None
    chunks = []
    for line in path.open(encoding="ascii"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(chunks)))
            name = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line)
    if name is not None:
        records.append((name, "".join(chunks)))
    return records


def main():
    records = read_fasta(FASTA)
    print(f"sequences {len(records)}", flush=True)
    _sequences, numbering_all, details_all, _hit_tables = run_anarci(
        records, scheme="imgt", ncpu=4, hmmerpath=HMMER)
    output = {}
    for (name, sequence), numbering_list, details_list in zip(
            records, numbering_all, details_all):
        if not numbering_list:
            output[name] = {"status": "failed"}
            continue
        numbering, start, end = numbering_list[0]
        details = details_list[0] if details_list else {}
        h3 = []
        indices = []
        input_index = 0
        for (position, insertion), aa in numbering:
            if aa == "-":
                continue
            if position is not None and 105 <= position <= 117:
                h3.append(aa)
                indices.append(input_index)
            input_index += 1
        output[name] = {
            "status": "ok", "sequence": sequence,
            "h3": "".join(h3), "indices": indices,
            "start": start, "end": end,
            "species": details.get("species") if isinstance(details, dict) else None,
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    ok = sum(1 for value in output.values() if value.get("status") == "ok")
    print(f"ok {ok}/{len(records)} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
