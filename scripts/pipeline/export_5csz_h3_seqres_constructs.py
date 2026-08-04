#!/usr/bin/env python3
"""Export run-scoped complete 5CSZ SEQRES constructs for an H3 library."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_v5_1_biological_constructs import apply_mutations, parse_seqres  # noqa: E402

ABETA42 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"


def sequence_sha256(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest().upper()


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", required=True, type=Path)
    parser.add_argument("--mutation-plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", default="5CSZ_h3b4_v1")
    parser.add_argument("--paired-nas-controls", type=int, default=2)
    args = parser.parse_args()

    selected = load_rows(args.selected)
    plan_rows = load_rows(args.mutation_plan)
    plans = defaultdict(list)
    for row in plan_rows:
        plans[row["candidate_id"]].append(row["mutation"])
    seqres = parse_seqres(ROOT / "data/anti_abeta_refs/5CSZ.pdb")
    parent_heavy, parent_light, abeta11 = seqres["A"], seqres["B"], seqres["E"]
    if (len(parent_heavy), len(parent_light), abeta11) != (228, 215, "DAEFRHDSGYE"):
        raise ValueError("Unexpected 5CSZ SEQRES contract")

    records = []
    for index, row in enumerate(selected):
        h3_text = ";".join(plans[row["candidate_id"]])
        backgrounds = [("N52H", "A:52:N>H", "rescued_N52H")]
        if index < args.paired_nas_controls:
            backgrounds.append(("N52", "", "retained_risk"))
        for background, background_text, nas_status in backgrounds:
            mutation_text = ";".join(filter(None, (h3_text, background_text)))
            heavy, heavy_applied = apply_mutations(parent_heavy, mutation_text, "A")
            light, light_applied = apply_mutations(parent_light, mutation_text, "B")
            expected = len([token for token in mutation_text.split(";") if token])
            if len(heavy_applied) + len(light_applied) != expected:
                raise ValueError(f"Not all mutations applied for {row['candidate_id']}")
            h3_mutations = [token for token in heavy_applied if 95 <= int(token.split(":")[1]) <= 102]
            if len(h3_mutations) != int(row["n_mutations"]):
                raise ValueError(f"H3 mutation count mismatch for {row['candidate_id']}")
            construct_id = f"{args.run_id}_{row['candidate_id']}__bg-{background}"
            records.append({
                "shortlist_rank": len(records) + 1,
                "construct_id": construct_id,
                "design_run_id": args.run_id,
                "reference_pdb": "5CSZ",
                "candidate_id": row["candidate_id"],
                "selection_rank": row["selection_rank"],
                "h3_mutations": ";".join(h3_mutations),
                "background_mutations": background_text,
                "applied_mutations": mutation_text,
                "h3_mutation_count": len(h3_mutations),
                "total_mutation_count": expected,
                "n_mutations": expected,
                "nas_status": nas_status,
                "nglyco_rescue": background == "N52H",
                "candidate_score": row["candidate_score"],
                "contact_retention": row["contact_retention"],
                "heavy_sequence": heavy,
                "light_sequence": light,
                "heavy_length": len(heavy),
                "light_length": len(light),
                "sequence_sha256": sequence_sha256(f"{heavy}:{light}"),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = [key for key in records[0] if key not in ("heavy_sequence", "light_sequence")]
    with (args.output_dir / "synthesis_candidate_shortlist.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    with (args.output_dir / "synthesis_candidate_shortlist.fasta").open("w", encoding="ascii") as handle:
        for record in records:
            handle.write(f">{record['construct_id']}|heavy\n{record['heavy_sequence']}\n")
            handle.write(f">{record['construct_id']}|light\n{record['light_sequence']}\n")
    with (args.output_dir / "fab.fasta").open("w", encoding="ascii") as fab, \
            (args.output_dir / "fab_abeta11.fasta").open("w", encoding="ascii") as short, \
            (args.output_dir / "fab_abeta42.fasta").open("w", encoding="ascii") as full:
        for record in records:
            chains = f"{record['heavy_sequence']}:{record['light_sequence']}"
            fab.write(f">{record['construct_id']}\n{chains}\n")
            short.write(f">{record['construct_id']}\n{chains}:{abeta11}\n")
            full.write(f">{record['construct_id']}\n{chains}:{ABETA42}\n")
    parent = {
        "construct_id": "5CSZ_complete_parent__bg-N52",
        "heavy_sequence": parent_heavy,
        "light_sequence": parent_light,
        "nas_status": "retained_positive_control",
        "sequence_sha256": sequence_sha256(f"{parent_heavy}:{parent_light}"),
    }
    manifest = {
        "run_id": args.run_id,
        "sequence_contract": "5CSZ SEQRES heavy=228, light=215",
        "primary_background": "N52H",
        "paired_nas_controls": args.paired_nas_controls,
        "parent_control": parent,
        "records": records,
        "ranking_contract": "developability/contact diversity first; multimer ipTM is an absolute rejection gate only",
    }
    (args.output_dir / "construct_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"constructs": len(records), "n52h": sum(r["nglyco_rescue"] for r in records),
                      "nas_controls": sum(not r["nglyco_rescue"] for r in records)}, indent=2))


if __name__ == "__main__":
    main()
