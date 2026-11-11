#!/usr/bin/env python3
"""Build the Aβ N-terminal curation table for the CSN expansion.

Read-only. Collects, for each candidate PDB entry, the SAbDab metadata, the
antibody heavy/light chains, and every short peptide chain. No RMSD and no
model inference: this table is the input to manual curation and to the
preregistration that must be frozen before any conformer analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from datetime import date
from pathlib import Path

import gemmi
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "data" / "sabdab2_current" / "all-summary.csv"
DEFAULT_ARCHIVE = ROOT / "data" / "sabdab2_current" / "splits.tar.gz"
DEFAULT_EXTRA = ROOT / "data" / "rcsb_abeta_extra"
DEFAULT_POOL = ROOT / "data" / "sabdab2_abag_project_pool_v2" / "cif"
CANDIDATES = [
    "2IPU", "2R0W", "3BAE", "3BKJ", "3IFL", "3IFN", "3IFO", "3IFP",
    "4F37", "4HIX", "4M1C", "4OJF", "4ONF", "4ONG", "5CSZ", "5W3P", "6CO3",
]
AA = set("ACDEFGHIKLMNPQRSTVWY")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def chain_sequence(chain) -> str:
    letters = []
    for residue in chain:
        info = gemmi.find_tabulated_residue(residue.name)
        code = info.one_letter_code.upper() if info else "X"
        if code in AA:
            letters.append(code)
        elif code == "X":
            letters.append("X")
    return "".join(letters)


def parse_structure(text: str):
    structure = gemmi.read_structure_string(text)
    if len(structure) == 0:
        return {}
    return {chain.name: chain_sequence(chain) for chain in structure[0]}


def load_archive_map(archive: Path, wanted: set[str]):
    members = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".cif"):
                continue
            match = re.search(r"(pdb_[0-9a-z]{8})", member.name)
            if not match:
                continue
            pdb4 = match.group(1)[-4:].upper()
            if pdb4 in wanted:
                text = tar.extractfile(member).read().decode("utf-8", errors="replace")
                members[pdb4] = (member.name, parse_structure(text))
    return members


def load_extra(extra_dir: Path, wanted: set[str]):
    found = {}
    for pdb in wanted:
        path = extra_dir / f"{pdb}.cif"
        if path.exists():
            found[pdb] = (str(path.relative_to(ROOT)), parse_structure(
                path.read_text(encoding="utf-8", errors="replace")))
    return found


def load_pool(pool_dir: Path, wanted: set[str]):
    found = {}
    for pdb in wanted:
        matches = sorted(pool_dir.glob(f"pdb_0000{pdb.lower()}_*.cif"))
        if not matches:
            continue
        path = matches[0]
        found[pdb] = (str(path.relative_to(ROOT)), parse_structure(
            path.read_text(encoding="utf-8", errors="replace")))
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--extra", type=Path, default=DEFAULT_EXTRA)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "csn_abeta_v1")
    args = parser.parse_args(argv)

    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(CANDIDATES)
    summary = pd.read_csv(args.summary, low_memory=False)
    summary["pdb4"] = summary["PDB"].str[-4:].str.upper()
    metadata = summary[summary["pdb4"].isin(wanted)].drop_duplicates("pdb4").set_index("pdb4")

    structures = {}
    structures.update(load_archive_map(args.archive, wanted))
    structures.update(load_extra(args.extra, wanted))
    structures.update(load_pool(args.pool, wanted))

    rows = []
    for pdb in CANDIDATES:
        if pdb not in structures:
            rows.append({"pdb": pdb, "source": "missing", "curation": "not_found"})
            continue
        source, chains = structures[pdb]
        h_chain = next((name for name, sequence in chains.items()
                        if name == "H" and len(sequence) > 50), None)
        l_chain = next((name for name, sequence in chains.items()
                        if name == "L" and len(sequence) > 50), None)
        if h_chain is None:
            long = sorted(((name, seq) for name, seq in chains.items() if len(seq) > 85),
                          key=lambda item: len(item[1]), reverse=True)
            h_chain = long[0][0] if long else None
            l_chain = long[1][0] if len(long) > 1 else None
        peptides = sorted(
            ((name, seq) for name, seq in chains.items()
             if name not in {h_chain, l_chain} and 5 <= len(seq) <= 50),
            key=lambda item: (-len(item[1]), item[0]),
        )
        meta = metadata.loc[pdb] if pdb in metadata.index else {}
        rows.append({
            "pdb": pdb,
            "source": source,
            "date": meta.get("date", ""),
            "antigen_name": meta.get("antigen_name", ""),
            "antigen_type": meta.get("antigen_type", ""),
            "method": meta.get("method", ""),
            "resolution": meta.get("resolution", ""),
            "heavy_species": meta.get("heavy_species", ""),
            "light_species": meta.get("light_species", ""),
            "heavy_subclass": meta.get("heavy_subclass", ""),
            "light_subclass": meta.get("light_subclass", ""),
            "heavy_chain": h_chain or "",
            "light_chain": l_chain or "",
            "VH": chains.get(h_chain, "")[:125] if h_chain else "",
            "VL": chains.get(l_chain, "")[:115] if l_chain else "",
            "peptide_chains": "|".join(name for name, _ in peptides),
            "peptide_sequences": "|".join(seq for _, seq in peptides),
            "contains_DAEFRH": any("DAEFRH" in seq for _, seq in peptides),
            "curation": "",
        })

    table = pd.DataFrame(rows)
    csv_path = out_dir / "abeta_entries.csv"
    table.to_csv(csv_path, index=False)
    manifest = {
        "schema_version": 1,
        "created": date.today().isoformat(),
        "candidates": CANDIDATES,
        "summary_sha256": sha256(args.summary),
        "archive_sha256": sha256(args.archive),
        "entries_found": int((table["source"] != "missing").sum()),
        "entries_missing": int((table["source"] == "missing").sum()),
        "entries_with_DAEFRH": int(table["contains_DAEFRH"].sum()),
        "output": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
        "output_sha256": sha256(csv_path),
        "note": "No RMSD or model score is computed here; curation must be frozen before conformer analysis.",
    }
    manifest_path = out_dir / "abeta_entries_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
