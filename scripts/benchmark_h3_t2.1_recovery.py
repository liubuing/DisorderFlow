#!/usr/bin/env python
"""Orchestrate T2.1 v2 per-structure benchmark across sealed-final records."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_h3_epitope_delta import read_record, write_record_backbone  # noqa: E402
from generate_h3_peptide_t21 import generate_t2_1  # noqa: E402
from t21_statistics import aggregate  # noqa: E402


def selected_hash(records):
    value = "\n".join(sorted(r["id"] for r in records))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def benchmark_record(config, audit_record, lmdb_path, out_dir):
    record = read_record(lmdb_path, audit_record["id"])
    rid = record["id"]
    cache = out_dir / "work" / rid / "record_result.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    record_dir = out_dir / "work" / rid
    deposited = record_dir / "deposited_complex.pdb"
    manifest = write_record_backbone(record, deposited, include_antigen=True)
    antibody_chains = [c for c in ("H", "L") if c in manifest]

    from Bio.PDB import PDBParser
    p = PDBParser(QUIET=True)
    model = p.get_structure("complex", str(deposited))[0]

    gen_cfg = {
        "chains": {"antibody": antibody_chains, "peptide": "P"},
        "sampling": config["sampling"],
        "torsion_perturbation": config["torsion_perturbation"],
    }
    audit = generate_t2_1(deposited, record_dir / "t2.1", gen_cfg, rid,
                          model, antibody_chains, "P")

    if not audit["torsion_hit_tier"]:
        result = {"id": rid, "valid_t2_1": False, "torsion_hit_tier": False, "t2_1_audit": audit}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
        return result

    arms = audit.get("arms", {})
    sc = arms.get("supplied_contacts", {})
    replicas = sc.get("replicas", [])
    accepted = [r for r in replicas if r.get("accepted")]
    req = config["ensemble_requirements"]

    valid = bool(
        len(accepted) >= int(req["minimum_accepted_replicas"])
        and sc.get("acceptance_fraction", 0) >= float(req["minimum_acceptance_fraction"]))

    held_out = [r.get("final_metrics", {}).get("held_out_contact_retention", 0)
                for r in accepted]
    rmsd_final = [r.get("final_metrics", {}).get("peptide_backbone_rmsd", 0)
                  for r in accepted]
    rmsd_init = audit.get("initial_metrics", {}).get("peptide_backbone_rmsd", 0)
    rmsd_rec = [rmsd_init - v for v in rmsd_final] if rmsd_final else []

    result = {
        "id": rid,
        "valid_t2_1": valid,
        "torsion_hit_tier": audit["torsion_hit_tier"],
        "torsion_scale_degrees": audit.get("torsion_scale_degrees"),
        "initial_peptide_rmsd": audit.get("initial_peptide_rmsd"),
        "t2_1_audit": audit,
    }
    if valid and held_out:
        m = {
            "mean_held_out_contact_recovery": float(np.mean(held_out)),
            "mean_rmsd_recovery_angstrom": float(np.mean(rmsd_rec)),
            "supplied_acceptance_fraction": sc.get("acceptance_fraction"),
        }
        for arm_name in ("all_contacts", "random_restraints", "null_structural"):
            ad = arms.get(arm_name, {})
            ar = ad.get("replicas", [])
            aa = [r.get("final_metrics", {}).get("held_out_contact_retention", 0)
                  for r in ar if r.get("accepted")]
            if aa:
                m[f"{arm_name}_held_out_contact"] = float(np.mean(aa))
        result["metrics"] = m

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    return result


def select_structures(audit, config):
    mode = config.get("selection_mode", "per_structure")
    records = audit["records"][config["benchmark_split"]]
    if mode == "per_structure":
        return list(records)
    selected = {}
    axis = config.get("inference_axis", "official_antigen_cluster")
    for r in sorted(records, key=lambda x: x["id"]):
        vals = r["axis_values"].get(axis, [])
        unit = vals[0] if vals else r["axis_values"]["pdb_id"][0]
        selected.setdefault(unit, r)
    return list(selected.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",
                        default=str(ROOT / "configs/benchmarks/peptide_h3_t2.1_v2_temporal_final.yml"))
    parser.add_argument("--out-dir",
                        default=str(ROOT / "results/publication/h3_t2.1_v2_temporal_final"))
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    audit = json.loads((ROOT / config["split_audit"]).read_text(encoding="utf-8"))
    selected = select_structures(audit, config)
    if args.max_records is not None:
        selected = selected[:args.max_records]

    print(f"T2.1 v2: {len(selected)} structures selected", flush=True)
    out_dir = Path(args.out_dir)
    lmdb_path = ROOT / config["lmdb"]

    results = []
    for i, rec in enumerate(selected, 1):
        results.append(benchmark_record(config, rec, lmdb_path, out_dir))
        v = results[-1].get("valid_t2_1")
        t = results[-1].get("torsion_hit_tier")
        print(f"  {i}/{len(selected)} {rec['id']} valid={v} torsion={t}", flush=True)

    cluster_by_id = {
        record["id"]: record["axis_values"]["official_antigen_cluster"][0]
        for record in audit["records"][config["benchmark_split"]]
    }
    agg = aggregate(results, config, cluster_by_id)
    output = {
        "schema_version": 1,
        "status": "t2.1_v2_per_structure_final",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "selected_ids_sha256": selected_hash(selected),
        "aggregate": agg,
        "results": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps({"aggregate": agg, "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
