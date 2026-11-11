#!/usr/bin/env python3
"""Stream the SAbDab2 CIF archive and inventory peptide-like chains.

Read-only. This is the first step of the CSN expansion protocol: it produces a
peptide-chain table (PDB, chain, sequence, length) that can be clustered by
exact/near sequence and then by antibody lineage. It does not call any model
and does not write to frozen artifacts.

Example:
    python scripts/scan_peptide_chains_v1.py \
        --out results/csn_expansion_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from collections import Counter
from datetime import date
from pathlib import Path

import gemmi
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "data" / "sabdab2_current" / "splits.tar.gz"
PDB_RE = re.compile(r"(pdb_[0-9a-z]{8})")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


AA_LETTERS = set("ACDEFGHIKLMNPQRSTVWY")


def chain_sequence(chain) -> str:
    """Return only amino-acid residues; drop waters, ions and other heteroatoms."""
    letters = []
    for residue in chain:
        info = gemmi.find_tabulated_residue(residue.name)
        code = info.one_letter_code.upper() if info else "X"
        if code in AA_LETTERS:
            letters.append(code)
        elif code == "X":
            letters.append("X")
    return "".join(letters)


def scan(archive: Path, min_len: int, max_len: int):
    rows = []
    failures = []
    n_members = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".cif"):
                continue
            n_members += 1
            match = PDB_RE.search(member.name)
            pdb_id = match.group(1) if match else ""
            try:
                text = tar.extractfile(member).read().decode("utf-8", errors="replace")
                structure = gemmi.read_structure_string(text)
                if len(structure) == 0:
                    failures.append({"member": member.name, "reason": "no_model"})
                    continue
                model = structure[0]
                names = [chain.name for chain in model]
                has_heavy = "H" in names
                has_light = "L" in names
                if not has_heavy:
                    continue
                for chain in model:
                    if chain.name in {"H", "L"}:
                        continue
                    sequence = chain_sequence(chain)
                    if not (min_len <= len(sequence) <= max_len):
                        continue
                    if sequence.count("X") > max(1, len(sequence) // 5):
                        continue
                    rows.append({
                        "pdb_id": pdb_id,
                        "member": member.name,
                        "peptide_chain": chain.name,
                        "length": len(sequence),
                        "sequence": sequence,
                        "has_light": has_light,
                        "n_chains": len(names),
                    })
            except Exception as exc:  # noqa: BLE001 - per-member fail-closed
                failures.append({"member": member.name, "reason": type(exc).__name__})
    return pd.DataFrame(rows), n_members, failures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "csn_expansion_v1")
    parser.add_argument("--min-len", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=50)
    args = parser.parse_args(argv)

    archive = args.archive if args.archive.is_absolute() else ROOT / args.archive
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    if not archive.exists():
        raise FileNotFoundError(archive)

    table, n_members, failures = scan(archive, args.min_len, args.max_len)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "peptide_inventory.csv"
    table.to_csv(csv_path, index=False)

    sequence_counts = table["sequence"].value_counts()
    repeated = sequence_counts[sequence_counts >= 2]
    pdb_counts = table.groupby("sequence")["pdb_id"].nunique()
    manifest = {
        "schema_version": 1,
        "created": date.today().isoformat(),
        "archive": str(archive.relative_to(ROOT)).replace("\\", "/"),
        "archive_sha256": sha256(archive),
        "filters": {
            "min_len": args.min_len,
            "max_len": args.max_len,
            "require_heavy_chain": True,
            "max_x_fraction": 0.2,
        },
        "counts": {
            "cif_members": n_members,
            "peptide_rows": int(len(table)),
            "unique_pdb": int(table["pdb_id"].nunique()),
            "unique_sequences": int(table["sequence"].nunique()),
            "sequences_in_multiple_pdbs": int((pdb_counts >= 2).sum()),
            "max_pdbs_per_sequence": int(pdb_counts.max()) if len(pdb_counts) else 0,
            "rows_in_repeated_sequences": int(repeated.sum()) if len(repeated) else 0,
            "parse_failures": len(failures),
        },
        "top_repeated_sequences": [
            {"sequence": seq, "rows": int(count), "pdbs": int(pdb_counts[seq])}
            for seq, count in repeated.head(20).items()
        ],
        "output": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
        "output_sha256": sha256(csv_path),
        "failures": failures[:50],
        "note": (
            "Local archive lower bound only; the archive covers 4,530/5,695 "
            "pre-2021 PDBs. Lineage independence still requires ANARCI-based "
            "clustering and manual curation."
        ),
    }
    manifest_path = out_dir / "peptide_inventory_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(json.dumps(manifest["counts"], indent=2))
    print(f"wrote {csv_path} ({csv_path.stat().st_size:,} bytes)")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
