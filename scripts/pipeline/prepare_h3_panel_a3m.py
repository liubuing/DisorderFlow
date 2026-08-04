#!/usr/bin/env python3
"""Prepare query-replaced 5CSZ A3Ms for the H3 structure panel."""

import argparse
import json
from pathlib import Path


def write_replaced(template, output, header, query):
    lines = template.read_text(encoding="utf-8").splitlines()
    lines[1] = header
    lines[2] = query
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--fab-template", required=True, type=Path)
    parser.add_argument("--abeta11-template", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    records = {row["construct_id"]: row for row in manifest["records"]}
    fab_dir = args.output_dir / "fab_a3m"
    abeta_dir = args.output_dir / "fab_abeta11_a3m"
    fab_dir.mkdir(parents=True, exist_ok=True)
    abeta_dir.mkdir(parents=True, exist_ok=True)
    for selected in panel["records"]:
        record = records[selected["construct_id"]]
        antibody = record["heavy_sequence"] + record["light_sequence"]
        write_replaced(args.fab_template, fab_dir / f"{record['construct_id']}.a3m", ">101\t102", antibody)
        write_replaced(
            args.abeta11_template,
            abeta_dir / f"{record['construct_id']}.a3m",
            ">101\t102\t103",
            antibody + "DAEFRHDSGYE",
        )
    print(json.dumps({"prepared": len(panel["records"]), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
