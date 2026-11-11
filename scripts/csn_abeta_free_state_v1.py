#!/usr/bin/env python3
"""Compare the bound Abeta N-terminal AEFRH conformer to public free-state NMR ensembles.

Read-only descriptive analysis. The free-state ensembles are public NMR models
(1AMC, 1BJB, 1BJC, 1ZE7, 1ZE9, 2BP4); they are not Boltzmann-weighted and some
are not pure aqueous (1ZE9 has zinc, 2BP4 has TFE). The script reports the
distribution of the best RMSD from each free-state model to the bound tight
cluster and to the bound outlier lineages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from itertools import combinations
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from csn_abeta_rmsd_v1 import (  # noqa: E402
    build_records,
    core_coords,
    rmsd,
    sha256 as _sha256,
)

ENTRIES = ROOT / "results" / "csn_abeta_v1" / "abeta_entries.csv"
FREE_DIR = ROOT / "data" / "rcsb_abeta_free_state"
OUT = ROOT / "results" / "csn_abeta_v1"
CORE = "AEFRH"
TIGHT_LINEAGES = {"L1", "L2", "L3", "L4", "L5", "L7"}
LINEAGE_REFS = {
    "L1": ["2IPU"], "L2": ["3BAE"], "L3": ["3IFN"], "L4": ["3IFO"],
    "L5": ["3IFP"], "L6": ["4HIX"], "L7": ["5CSZ"], "L8": ["5W3P"],
    "L9": ["6CO3"],
}
NMR_ENTRIES = {
    "1AMC": {"condition": "aqueous, Abeta 1-28", "primary": True},
    "1BJB": {"condition": "aqueous, Abeta[E16] 1-28", "primary": True},
    "1BJC": {"condition": "aqueous, Abeta[F16] 1-28", "primary": True},
    "1ZE7": {"condition": "aqueous, Abeta 1-16", "primary": True},
    "1ZE9": {"condition": "zinc-bound, Abeta 1-16", "primary": False},
    "2BP4": {"condition": "TFE-water 80%, Abeta 1-16", "primary": False},
}


def sha256(path: Path) -> str:
    return _sha256(path)


def valid_models(path: Path, core: str):
    structure = gemmi.read_structure_string(
        path.read_text(encoding="utf-8", errors="replace"))
    models = []
    for model in structure:
        for chain in model:
            coords, sequence = core_coords(chain, core)
            if coords is not None:
                models.append({"chain": chain.name, "sequence": sequence, "coords": coords})
                break
    return models


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entries", type=Path, default=ENTRIES)
    parser.add_argument("--free-dir", type=Path, default=FREE_DIR)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    entries = pd.read_csv(args.entries)
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    references = {}
    for lineage, pdbs in LINEAGE_REFS.items():
        records, missing = build_records(entries, {lineage: pdbs}, CORE)
        if missing:
            raise RuntimeError(f"Missing reference records: {missing}")
        references[lineage] = records
    tight_refs = [record for lineage, records in references.items()
                  if lineage in TIGHT_LINEAGES for record in records]
    if not tight_refs:
        raise RuntimeError("No tight-cluster references")

    rows = []
    per_entry = {}
    for pdb, meta in NMR_ENTRIES.items():
        path = args.free_dir / f"{pdb}.cif"
        if not path.exists():
            per_entry[pdb] = {"status": "missing"}
            continue
        models = valid_models(path, CORE)
        model_rows = []
        for index, model in enumerate(models):
            tight = [rmsd(model["coords"], reference["coords"]) for reference in tight_refs]
            best_tight = min(tight)
            best_ref = tight_refs[int(np.argmin(tight))]
            row = {
                "nmr_entry": pdb,
                "model": index + 1,
                "chain": model["chain"],
                "sequence": model["sequence"],
                "min_tight_rmsd": best_tight,
                "median_tight_rmsd": float(np.median(tight)),
                "closest_lineage": best_ref["lineage"],
                "closest_pdb": best_ref["pdb"],
                "min_L6_rmsd": min(rmsd(model["coords"], ref["coords"])
                                   for ref in references["L6"]),
                "min_L8_rmsd": min(rmsd(model["coords"], ref["coords"])
                                   for ref in references["L8"]),
                "min_L9_rmsd": min(rmsd(model["coords"], ref["coords"])
                                   for ref in references["L9"]),
            }
            model_rows.append(row)
            rows.append(row)
        if model_rows:
            within = [rmsd(first["coords"], second["coords"])
                      for first, second in combinations(models, 2)]
            best = np.array([row["min_tight_rmsd"] for row in model_rows])
            per_entry[pdb] = {
                "condition": meta["condition"],
                "primary": meta["primary"],
                "n_models": len(models),
                "n_valid": len(model_rows),
                "best_tight_median": float(np.median(best)),
                "best_tight_min": float(np.min(best)),
                "best_tight_max": float(np.max(best)),
                "fraction_within_0.5": float(np.mean(best <= 0.5)),
                "fraction_within_1.0": float(np.mean(best <= 1.0)),
                "fraction_within_1.5": float(np.mean(best <= 1.5)),
                "closest_lineage_counts": pd.Series(
                    [row["closest_lineage"] for row in model_rows]).value_counts().to_dict(),
                "free_ensemble_pairwise_median": float(np.median(within)) if within else None,
            }
        else:
            per_entry[pdb] = {
                "condition": meta["condition"],
                "primary": meta["primary"],
                "n_models": len(models),
                "n_valid": 0,
                "status": "core_not_resolved",
            }

    table = pd.DataFrame(rows)
    csv_path = out_dir / "free_state_models.csv"
    table.to_csv(csv_path, index=False)

    primary = table[table["nmr_entry"].isin(
        [pdb for pdb, meta in NMR_ENTRIES.items() if meta["primary"]])]
    aggregate = {
        "n_models": int(len(primary)),
        "best_tight_median": float(primary["min_tight_rmsd"].median()) if len(primary) else None,
        "fraction_within_0.5": float((primary["min_tight_rmsd"] <= 0.5).mean()) if len(primary) else None,
        "fraction_within_1.0": float((primary["min_tight_rmsd"] <= 1.0).mean()) if len(primary) else None,
        "fraction_within_1.5": float((primary["min_tight_rmsd"] <= 1.5).mean()) if len(primary) else None,
        "closest_lineage_counts": primary["closest_lineage"].value_counts().to_dict() if len(primary) else {},
    }
    payload = {
        "schema_version": 1,
        "created": date.today().isoformat(),
        "analysis": "bound_vs_free_state",
        "classification": "development_descriptive_not_confirmatory_claim",
        "core": CORE,
        "tight_lineages": sorted(TIGHT_LINEAGES),
        "references": {lineage: [record["pdb"] for record in records]
                       for lineage, records in references.items()},
        "free_state_entries": per_entry,
        "primary_aggregate": aggregate,
        "caveats": [
            "NMR ensembles are not Boltzmann-weighted.",
            "1ZE9 contains zinc; 2BP4 contains 80% TFE; both are sensitivity arms.",
            "1AMC/1BJB/1BJC are Abeta 1-28; 1ZE7 is Abeta 1-16.",
            "The bound reference is the AEFRH core on the crystal peptide; terminal residues are excluded.",
            "This analysis tests whether the bound conformer exists in the free ensemble, not whether binding selects it.",
        ],
        "entries_csv": str(args.entries.relative_to(ROOT)).replace("\\", "/"),
        "entries_csv_sha256": sha256(args.entries),
        "output_csv": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
        "output_csv_sha256": sha256(csv_path),
    }
    path = out_dir / "free_state_summary.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
