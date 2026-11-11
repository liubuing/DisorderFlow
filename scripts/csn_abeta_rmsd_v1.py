#!/usr/bin/env python3
"""Preregistered CSN Aβ N-terminal peptide-core RMSD analysis.

The cohort, core, tiers and decision rules live in the preregistration JSON
that this script reads. The script must not be run before that file is frozen.
It computes Cα RMSD after a row-convention Kabsch superposition (the corrected
implementation used by CSN C0), reports the same-entry NCS floor, and writes
pair-level and summary artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "sabdab2_current" / "splits.tar.gz"
AA = set("ACDEFGHIKLMNPQRSTVWY")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def kabsch_row(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Row-convention Kabsch; a perfect rigid motion returns zero RMSD."""
    mc = mobile - mobile.mean(axis=0)
    rc = reference - reference.mean(axis=0)
    u, _s, vt = np.linalg.svd(mc.T @ rc)
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, d]) @ vt
    return mc @ rotation + reference.mean(axis=0)


def rmsd(mobile: np.ndarray, reference: np.ndarray) -> float:
    aligned = kabsch_row(mobile, reference)
    return float(np.sqrt(np.mean(np.sum((aligned - reference) ** 2, axis=1))))


def chain_residues(chain):
    residues = []
    for residue in chain:
        info = gemmi.find_tabulated_residue(residue.name)
        code = info.one_letter_code.upper() if info else "X"
        if code in AA or code == "X":
            residues.append((residue, code))
    return residues


def ca_coords(residues):
    coords = []
    for residue, _code in residues:
        ca = None
        for atom in residue:
            if atom.name == "CA" and atom.altloc in ("", "A"):
                ca = atom.pos
                break
        if ca is None:
            for atom in residue:
                if atom.name == "CA":
                    ca = atom.pos
                    break
        if ca is None:
            return None
        coords.append([ca.x, ca.y, ca.z])
    return np.asarray(coords, dtype=float)


def core_coords(chain, core: str):
    residues = chain_residues(chain)
    sequence = "".join(code for _residue, code in residues)
    index = sequence.find(core)
    if index < 0:
        return None, sequence
    subset = residues[index:index + len(core)]
    if len(subset) != len(core):
        return None, sequence
    coords = ca_coords(subset)
    return coords, sequence


def load_entry_chains(source_value: str, core: str):
    if source_value.startswith("splits_final/"):
        with tarfile.open(ARCHIVE, "r:gz") as tar:
            text = tar.extractfile(source_value).read().decode("utf-8", errors="replace")
    else:
        source = Path(source_value.replace("\\", "/"))
        if not source.is_absolute():
            source = ROOT / source
        text = source.read_text(encoding="utf-8", errors="replace")
    structure = gemmi.read_structure_string(text)
    if len(structure) == 0:
        return []
    chains = []
    for chain in structure[0]:
        coords, sequence = core_coords(chain, core)
        if coords is None:
            continue
        chains.append({"chain": chain.name, "sequence": sequence, "coords": coords})
    return chains


def build_records(entries: pd.DataFrame, cohort: dict, core: str):
    records = []
    missing = []
    for lineage, pdbs in cohort.items():
        for pdb in pdbs:
            row = entries[entries["pdb"] == pdb]
            if row.empty:
                missing.append({"pdb": pdb, "reason": "not_in_entry_table"})
                continue
            source_value = row.iloc[0]["source"]
            if not isinstance(source_value, str) or source_value in ("", "missing"):
                missing.append({"pdb": pdb, "reason": "no_source"})
                continue
            try:
                chains = load_entry_chains(source_value, core)
            except (FileNotFoundError, KeyError, OSError):
                missing.append({"pdb": pdb, "reason": "source_missing"})
                continue
            if not chains:
                missing.append({"pdb": pdb, "reason": "no_core_chain"})
                continue
            for chain in chains:
                records.append({
                    "pdb": pdb,
                    "lineage": lineage,
                    "chain": chain["chain"],
                    "sequence": chain["sequence"],
                    "coords": chain["coords"],
                })
    return records, missing


def collect_pairs(records):
    pairs = []
    by_entry = {}
    for record in records:
        by_entry.setdefault(record["pdb"], []).append(record)
    for pdb, chains in by_entry.items():
        for first, second in combinations(chains, 2):
            pairs.append({
                "pdb_i": pdb, "chain_i": first["chain"], "lineage_i": first["lineage"],
                "pdb_j": pdb, "chain_j": second["chain"], "lineage_j": second["lineage"],
                "pair_type": "ncs", "rmsd": rmsd(first["coords"], second["coords"]),
            })
    for first, second in combinations(records, 2):
        if first["pdb"] == second["pdb"]:
            continue
        pair_type = "within_lineage" if first["lineage"] == second["lineage"] else "cross_lineage"
        pairs.append({
            "pdb_i": first["pdb"], "chain_i": first["chain"], "lineage_i": first["lineage"],
            "pdb_j": second["pdb"], "chain_j": second["chain"], "lineage_j": second["lineage"],
            "pair_type": pair_type, "rmsd": rmsd(first["coords"], second["coords"]),
        })
    return pairs


def entry_pair_values(pairs):
    """Aggregate chain pairs to one value per entry pair."""
    grouped = {}
    for pair in pairs:
        key = tuple(sorted((pair["pdb_i"], pair["pdb_j"])))
        grouped.setdefault(key, []).append(pair["rmsd"])
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def bootstrap_lineage_ci(records, pairs, n_boot=10000, seed=20261111):
    """Resample lineages, then recompute the median cross-lineage entry-pair RMSD."""
    lineage_values = {}
    for pair in pairs:
        if pair["pair_type"] != "cross_lineage":
            continue
        key = tuple(sorted((pair["lineage_i"], pair["lineage_j"])))
        lineage_values.setdefault(key, []).append(pair["rmsd"])
    if not lineage_values:
        return None
    lineages = sorted({key[0] for key in lineage_values} | {key[1] for key in lineage_values})
    rng = np.random.default_rng(seed)
    medians = []
    for _ in range(n_boot):
        sample = list(rng.choice(lineages, size=len(lineages), replace=True))
        values = []
        for i, j in combinations(range(len(sample)), 2):
            key = tuple(sorted((sample[i], sample[j])))
            values.extend(lineage_values.get(key, []))
        if values:
            medians.append(float(np.median(values)))
    if not medians:
        return None
    return {
        "n_bootstrap": n_boot,
        "seed": seed,
        "median_ci95": [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))],
    }


def summarize(records, pairs, missing, core: str, tiers: dict):
    ncs = [p["rmsd"] for p in pairs if p["pair_type"] == "ncs"]
    within = [p["rmsd"] for p in pairs if p["pair_type"] == "within_lineage"]
    cross = [p["rmsd"] for p in pairs if p["pair_type"] == "cross_lineage"]
    cross_entry = entry_pair_values([p for p in pairs if p["pair_type"] == "cross_lineage"])
    tier_same = tiers["same_bound_conformer_max"]
    tier_related = tiers["related_max"]
    summary = {
        "core": core,
        "n_entries": len({r["pdb"] for r in records}),
        "n_lineages": len({r["lineage"] for r in records}),
        "n_chains": len(records),
        "n_pairs": len(pairs),
        "ncs": {
            "n": len(ncs),
            "mean": float(np.mean(ncs)) if ncs else None,
            "median": float(np.median(ncs)) if ncs else None,
            "max": float(np.max(ncs)) if ncs else None,
        },
        "within_lineage": {
            "n": len(within),
            "mean": float(np.mean(within)) if within else None,
            "median": float(np.median(within)) if within else None,
        },
        "cross_lineage": {
            "n": len(cross),
            "mean": float(np.mean(cross)) if cross else None,
            "median": float(np.median(cross)) if cross else None,
            "entry_pair_median": float(np.median(list(cross_entry.values()))) if cross_entry else None,
            "entry_pair_mean": float(np.mean(list(cross_entry.values()))) if cross_entry else None,
        },
        "cross_lineage_tiers": {
            "same_bound_conformer": int(sum(1 for value in cross if value <= tier_same)),
            "related": int(sum(1 for value in cross if tier_same < value <= tier_related)),
            "different": int(sum(1 for value in cross if value > tier_related)),
        },
        "missing": missing,
    }
    return summary


def decide(summary, tiers: dict):
    ncs_mean = summary["ncs"]["mean"]
    cross_median = summary["cross_lineage"]["entry_pair_median"]
    if ncs_mean is None or ncs_mean > tiers["ncs_floor_max"]:
        return "measurement_floor_invalid"
    if cross_median is None:
        return "insufficient_pairs"
    if cross_median <= tiers["same_bound_conformer_max"]:
        return "shared_bound_conformer"
    if cross_median <= tiers["related_max"]:
        return "related_conformers"
    return "different_conformers"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path,
                        default=ROOT / "results" / "csn_abeta_v1" / "preregistration.json")
    parser.add_argument("--entries", type=Path,
                        default=ROOT / "results" / "csn_abeta_v1" / "abeta_entries.csv")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results" / "csn_abeta_v1")
    args = parser.parse_args(argv)

    prereg_path = args.prereg if args.prereg.is_absolute() else ROOT / args.prereg
    entries_path = args.entries if args.entries.is_absolute() else ROOT / args.entries
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if prereg.get("status") != "frozen_before_rmsd":
        raise RuntimeError("Preregistration is not frozen; refusing to compute RMSD")
    entries = pd.read_csv(entries_path)
    tiers = prereg["tiers"]
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}
    for label, core in (
            ("primary", prereg["core_primary"]),
            ("secondary", prereg["core_secondary"])):
        cohort = prereg["cohort"] if label == "primary" else prereg["cohort_secondary"]
        records, missing = build_records(entries, cohort, core)
        pairs = collect_pairs(records)
        summary = summarize(records, pairs, missing, core, tiers)
        summary["verdict"] = decide(summary, tiers)
        summary["bootstrap"] = bootstrap_lineage_ci(records, pairs)
        summary["classification"] = "development_descriptive_not_confirmatory_claim"
        summary["prereg_sha256"] = sha256(prereg_path)
        pairs_path = out_dir / f"rmsd_pairs_{label}.csv"
        pd.DataFrame([{k: v for k, v in pair.items() if k != "coords"} for pair in pairs]).to_csv(
            pairs_path, index=False)
        summary_path = out_dir / f"summary_{label}.json"
        summary["pairs_csv"] = str(pairs_path.relative_to(ROOT)).replace("\\", "/")
        summary["pairs_csv_sha256"] = sha256(pairs_path)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8")
        outputs[label] = summary

    completion = {
        "schema_version": 1,
        "created": datetime.now(timezone.utc).isoformat(),
        "prereg": str(prereg_path.relative_to(ROOT)).replace("\\", "/"),
        "prereg_sha256": sha256(prereg_path),
        "entries_sha256": sha256(entries_path),
        "script_sha256": sha256(Path(__file__)),
        "primary_verdict": outputs["primary"]["verdict"],
        "secondary_verdict": outputs["secondary"]["verdict"],
        "outputs": {label: summary["pairs_csv"] for label, summary in outputs.items()},
    }
    (out_dir / "completion.json").write_text(
        json.dumps(completion, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "primary": outputs["primary"]["verdict"],
        "secondary": outputs["secondary"]["verdict"],
        "primary_summary": outputs["primary"]["cross_lineage"],
        "secondary_summary": outputs["secondary"]["cross_lineage"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
