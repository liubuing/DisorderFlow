#!/usr/bin/env python3
"""Compare ANARCI IMGT, Kabat and Chothia assignments for the Abeta cohort."""
from __future__ import annotations

import json
import sys
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from csn_abeta_rmsd_v1 import sha256  # noqa: E402

RESULTS = ROOT / "results" / "csn_abeta_v1"
SCHEMES = {
    "imgt": {"vh": "abeta_vh_anarci.csv_H.csv", "vl": "abeta_vl_anarci.csv_KL.csv"},
    "kabat": {"vh": "abeta_vh_kabat.csv_H.csv", "vl": "abeta_vl_kabat.csv_KL.csv"},
    "chothia": {"vh": "abeta_vh_chothia.csv_H.csv", "vl": "abeta_vl_chothia.csv_KL.csv"},
}
IMGT_POS = [str(value) for value in range(105, 118)] + ["111A", "111B", "112B", "112A"]
KABAT_CHOTHIA_VH_POS = [str(value) for value in range(95, 103)]
KABAT_CHOTHIA_VL_POS = [str(value) for value in range(89, 98)]


def cdr3(row, positions):
    letters = []
    for position in positions:
        if position not in row.index:
            continue
        value = str(row[position])
        if value not in {"-", "nan", ""}:
            letters.append(value)
    return "".join(letters)


def load_scheme(chain: str):
    frames = {}
    for scheme, files in SCHEMES.items():
        path = RESULTS / files[chain]
        frame = pd.read_csv(path)
        positions = IMGT_POS if scheme == "imgt" else (
            KABAT_CHOTHIA_VH_POS if chain == "vh" else KABAT_CHOTHIA_VL_POS)
        frame = frame.set_index("Id")
        frame["cdr3"] = frame.apply(lambda row: cdr3(row, positions), axis=1)
        frames[scheme] = frame
    return frames


def compare(chain: str):
    frames = load_scheme(chain)
    base = frames["imgt"]
    rows = []
    for identifier, imgt_row in base.iterrows():
        # IDs are like 2IPU_VH / 2IPU_VL.
        record = {"id": identifier}
        for scheme, frame in frames.items():
            if identifier not in frame.index:
                record[f"{scheme}_missing"] = True
                continue
            row = frame.loc[identifier]
            record[f"{scheme}_v_gene"] = row.get("v_gene")
            record[f"{scheme}_j_gene"] = row.get("j_gene")
            record[f"{scheme}_species"] = row.get("identity_species")
            record[f"{scheme}_cdr3"] = row.get("cdr3")
            record[f"{scheme}_cdr3_length"] = len(str(row.get("cdr3")))
        rows.append(record)

    vh_mismatch = sum(
        1 for row in rows
        if len({row.get("imgt_v_gene"), row.get("kabat_v_gene"), row.get("chothia_v_gene")}) > 1)
    jh_mismatch = sum(
        1 for row in rows
        if len({row.get("imgt_j_gene"), row.get("kabat_j_gene"), row.get("chothia_j_gene")}) > 1)
    species_mismatch = sum(
        1 for row in rows
        if len({row.get("imgt_species"), row.get("kabat_species"), row.get("chothia_species")}) > 1)
    cdr3_similarity = []
    for row in rows:
        imgt = str(row.get("imgt_cdr3", ""))
        kabat = str(row.get("kabat_cdr3", ""))
        if imgt and kabat:
            cdr3_similarity.append(SequenceMatcher(None, imgt, kabat).ratio())
    return {
        "chain": chain,
        "n": len(rows),
        "v_gene_mismatches": vh_mismatch,
        "j_gene_mismatches": jh_mismatch,
        "species_mismatches": species_mismatch,
        "cdr3_imgt_vs_kabat_similarity_median": (
            float(pd.Series(cdr3_similarity).median()) if cdr3_similarity else None
        ),
        "rows": rows,
    }


def main():
    output = {
        "schema_version": 1,
        "created": date.today().isoformat(),
        "analysis": "anarci_scheme_sensitivity",
        "classification": "sensitivity_analysis_not_confirmatory",
        "vh": compare("vh"),
        "vl": compare("vl"),
        "conclusion": (
            "V/J germline assignments are compared across IMGT, Kabat and Chothia. "
            "Identical V/J calls mean the lineage clusters do not depend on the "
            "numbering scheme; CDR3 boundaries differ by scheme by construction."
        ),
        "sources": {
            scheme: {chain: str(RESULTS / files[chain]) for chain in files}
            for scheme, files in SCHEMES.items()
        },
    }
    path = RESULTS / "anarci_scheme_comparison.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "vh": {key: output["vh"][key] for key in (
            "n", "v_gene_mismatches", "j_gene_mismatches", "species_mismatches",
            "cdr3_imgt_vs_kabat_similarity_median")},
        "vl": {key: output["vl"][key] for key in (
            "n", "v_gene_mismatches", "j_gene_mismatches", "species_mismatches",
            "cdr3_imgt_vs_kabat_similarity_median")},
    }
    print(json.dumps(summary, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
