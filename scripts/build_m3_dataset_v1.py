#!/usr/bin/env python3
"""Build a ProteinMPNN-format training set from pre-2021 SAbDab CIFs.

Read-only over the source archive; writes only to the requested output
directory. The output layout matches ProteinMPNN/training/training.py:

  <out>/list.csv
  <out>/valid_clusters.txt
  <out>/test_clusters.txt
  <out>/pdb/<two-char>/<pdbid>.pt
  <out>/pdb/<two-char>/<pdbid>_<chain>.pt

Only N, CA, C and O atoms are populated (the training featurizer uses the
first four atom slots); the remaining 10 slots are zero/masked.
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
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
AA = set("ACDEFGHIKLMNPQRSTVWY")
BACKBONE = ("N", "CA", "C", "O")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def chain_arrays(chain):
    sequence = []
    coords = []
    for residue in chain:
        info = gemmi.find_tabulated_residue(residue.name)
        code = info.one_letter_code.upper() if info else ""
        if code not in AA:
            return None
        atoms = {}
        for atom in residue:
            # Gemmi uses "\x00" for a blank altloc in some CIFs.
            if atom.name in BACKBONE and atom.altloc in ("", "A", "\x00"):
                atoms.setdefault(atom.name, atom.pos)
        if len(atoms) != 4:
            return None
        sequence.append(code)
        coords.append([[atoms[name].x, atoms[name].y, atoms[name].z]
                       for name in BACKBONE])
    if not sequence:
        return None
    xyz = np.zeros((len(sequence), 14, 3), dtype=np.float32)
    mask = np.zeros((len(sequence), 14), dtype=bool)
    xyz[:, :4, :] = np.asarray(coords, dtype=np.float32)
    mask[:, :4] = True
    return {
        "seq": "".join(sequence),
        "xyz": torch.from_numpy(xyz),
        "mask": torch.from_numpy(mask),
        "bfac": torch.zeros((len(sequence), 14), dtype=torch.float32),
        "occ": torch.ones((len(sequence), 14), dtype=torch.float32),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory", type=Path,
        default=ROOT / "results" / "controlled_replication_v1" / "pre2021_inventory.csv")
    parser.add_argument(
        "--archive", type=Path,
        default=ROOT / "data" / "sabdab2_current" / "splits.tar.gz")
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "data" / "controlled_replication_v1" / "m3_dataset")
    parser.add_argument("--max-pdb", type=int, default=None)
    parser.add_argument("--min-length", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=500)
    args = parser.parse_args(argv)

    inventory = pd.read_csv(args.inventory, low_memory=False)
    inventory = inventory.drop_duplicates("PDB").reset_index(drop=True)
    inventory["pdb4"] = inventory["PDB"].str[-4:].str.lower()
    if args.max_pdb is not None:
        inventory = inventory.head(args.max_pdb)
    wanted = dict(zip(inventory["pdb4"], inventory.index))

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "pdb").mkdir(exist_ok=True)
    rows = []
    written = []
    skipped = []

    with tarfile.open(args.archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".cif"):
                continue
            match = re.search(r"(pdb_[0-9a-z]{8})", member.name)
            if not match:
                continue
            pdb4 = match.group(1)[-4:].lower()
            if pdb4 not in wanted or pdb4 in written:
                continue
            text = tar.extractfile(member).read().decode("utf-8", errors="replace")
            structure = gemmi.read_structure_string(text)
            if len(structure) == 0:
                skipped.append({"pdb": pdb4, "reason": "no_model"})
                continue
            model = structure[0]
            row = inventory.iloc[wanted[pdb4]]
            heavy_id = str(row.get("Hchain", "")).strip()
            light_id = str(row.get("Lchain", "")).strip()
            candidates = {}
            for chain in model:
                arrays = chain_arrays(chain)
                if arrays is None:
                    continue
                length = len(arrays["seq"])
                if not (args.min_length <= length <= args.max_length):
                    continue
                candidates[chain.name] = arrays
            if not candidates:
                skipped.append({"pdb": pdb4, "reason": "no_chain"})
                continue
            # Chain names in the CIF may be original (A/B) or normalized (H/L).
            if heavy_id in candidates:
                selected = [heavy_id]
            elif "H" in candidates:
                selected = ["H"]
            else:
                selected = [max(candidates, key=lambda name: len(candidates[name]["seq"]))]
            light = light_id if light_id in candidates else (
                "L" if "L" in candidates else None)
            if light and light not in selected:
                selected.append(light)
            elif len(selected) == 1:
                others = [name for name in candidates if name not in selected]
                if others:
                    selected.append(
                        max(others, key=lambda name: len(candidates[name]["seq"])))
            chains = {name: candidates[name] for name in selected}
            keys = sorted(chains)
            folder = out / "pdb" / pdb4[1:3]
            folder.mkdir(parents=True, exist_ok=True)
            for name, arrays in chains.items():
                torch.save(arrays, folder / f"{pdb4}_{name}.pt")
            row = inventory.iloc[wanted[pdb4]]
            metadata = {
                "method": str(row.get("method", "XRAY")),
                "date": str(row.get("date", "")),
                "resolution": float(row.get("resolution_numeric", 0.0) or 0.0),
                "chains": keys,
                "seq": [[chains[name]["seq"], chains[name]["seq"]] for name in keys],
                "asmb_ids": ["1"],
                "asmb_details": ["author"],
                "asmb_method": ["PISA"],
                "asmb_chains": [",".join(keys)],
                "asmb_xform0": torch.eye(4).unsqueeze(0),
                "tm": torch.zeros((len(keys), len(keys), 3)),
            }
            torch.save(metadata, folder / f"{pdb4}.pt")
            cluster = int(wanted[pdb4])
            for name, arrays in chains.items():
                sequence = arrays["seq"]
                rows.append({
                    "CHAINID": f"{pdb4}_{name}",
                    "DEPOSITION": metadata["date"],
                    "RESOLUTION": metadata["resolution"],
                    "HASH": hashlib.sha1(sequence.encode()).hexdigest()[:6],
                    "CLUSTER": cluster,
                    "SEQUENCE": sequence,
                })
            written.append(pdb4)

    pd.DataFrame(rows).to_csv(out / "list.csv", index=False)
    clusters = sorted({row["CLUSTER"] for row in rows})
    valid = [cluster for cluster in clusters if cluster % 10 == 0]
    test = [cluster for cluster in clusters if cluster % 10 == 5]
    (out / "valid_clusters.txt").write_text("\n".join(map(str, valid)) + "\n")
    (out / "test_clusters.txt").write_text("\n".join(map(str, test)) + "\n")
    manifest = {
        "schema_version": 1,
        "created": date.today().isoformat(),
        "inventory": str(args.inventory.relative_to(ROOT)).replace("\\", "/"),
        "inventory_sha256": sha256(args.inventory),
        "archive_sha256": sha256(args.archive),
        "n_pdb_written": len(written),
        "n_chains": len(rows),
        "n_clusters": len(clusters),
        "n_valid_clusters": len(valid),
        "n_test_clusters": len(test),
        "skipped": skipped,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
