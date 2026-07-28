#!/usr/bin/env python3
"""Convert the audited SAbDab2 external CIF subset to Phase3 LMDB format."""

import argparse
import csv
import json
from pathlib import Path

from preprocess_sabdab_phase3 import build_lmdb, preprocess_one


def chain_ids(value):
    return [item.strip() for item in str(value).split("/") if item.strip() and item.strip() != "+"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/sabdab2_abag_external")
    parser.add_argument("--output", default="data/sabdab2_abag_external/test.lmdb")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    with (input_dir / "test.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    processed = []
    failures = []
    for index, row in enumerate(rows):
        instance = row["instance"]
        cif_path = input_dir / "cif" / f"{instance}.cif"
        result = preprocess_one(
            instance,
            cif_path,
            row.get("Hchain") or None,
            row.get("Lchain") or None,
            chain_ids(row.get("agchains", "")),
        )
        if result is None or result.get("antigen") is None:
            failures.append(instance)
        else:
            result["pdb_id"] = row["pdb_code"]
            result["official_instance"] = instance
            result["external_split"] = "sabdab2_abag_interface_eligible"
            processed.append(result)
        if (index + 1) % 50 == 0:
            print(f"Processed {index + 1}/{len(rows)} CIF files", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    build_lmdb(processed, str(output))
    report = {
        "input_complexes": len(rows),
        "processed_complexes": len(processed),
        "failed_complexes": len(failures),
        "failures": failures,
        "output": str(output),
    }
    (input_dir / "preprocess_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "failures"}, indent=2))
    if not processed:
        raise RuntimeError("No SAbDab2 external complexes could be preprocessed")


if __name__ == "__main__":
    main()
