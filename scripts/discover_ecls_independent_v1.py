"""Model-free 2025-2026 structural cohort feasibility with conservative exclusions."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_ecls_pae_joint import digest, read, write
from scripts.build.acquire_rcsb_candidate_interface_extension import acquire
from scripts.build.discover_rcsb_candidate_interface_extension import discover
from scripts.build.materialize_rcsb_candidate_interface_extension import materialize

OUT = ROOT / "data/ecls_independent_feasibility_v1"


def freeze(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Refusing changed frozen artifact: {path}")
    else:
        write(path, value)


def prepare():
    query = read(ROOT / "configs/candidate_interface_extension_rcsb_query_v2.json")
    for node in query["query"]["nodes"]:
        if node.get("parameters", {}).get("attribute") == "rcsb_accession_info.initial_release_date":
            node["parameters"]["value"].update({"from": "2025-01-01", "to": "2026-09-16"})
    freeze(OUT / "query.json", query)
    base = ROOT / "data/pae_surrogate_external_v4/reference_union.json"
    ref = read(base)
    inputs = {base.relative_to(ROOT).as_posix(): digest(base)}
    exposed = set(ref["exact_exposed_pdb_ids"])
    # Inherit exact exclusion even for metadata viewed in earlier work.
    for path in sorted((ROOT / "data/candidate_interface_external_calibration_v1").glob("snapshot*/entries.json")) + [ROOT / "data/pae_surrogate_external_v4/snapshot/entries.json"]:
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
        exposed.update(e["rcsb_id"].lower() for e in read(path)["entries"])
    recent = ROOT / "data/pae_surrogate_external_v4/structures/structural_manifest.json"
    inputs[recent.relative_to(ROOT).as_posix()] = digest(recent)
    for r in read(recent)["records"]:
        ref["records"].append({**r, "reference_id": "ecls_prior_v4_" + r["instance"]})
        exposed.add(r["pdb_id"].lower())
    ref["exact_exposed_pdb_ids"] = sorted(exposed)
    ref["new_exposure_inputs_ecls_v1"] = inputs
    ref["classification"] = "exclusion_reference_for_ecls_independent_feasibility"
    freeze(OUT / "reference_union.json", ref)
    protocol = {"classification": "model_free_feasibility_not_validation",
        "discovery_contract": {"query_config_sha256": digest(OUT / "query.json")},
        "release_window": ["2025-01-01", "2026-09-16"],
        "scope": "paired_HL_antibody_peptide_5_to_50_residues_existing_structural_eligibility",
        "reference_sha256": digest(OUT / "reference_union.json"),
        "minimum_components_for_feasibility": 12,
        "minimum_is_not_power_analysis": True,
        "five_axis_identity": {"vh": .9, "vl": .9, "paired_cdr": .7, "h3": .5, "antigen": .3},
        "coverage": .8, "include_viral": True,
        "selection": "all_metadata_eligible_unexposed_entries_then_geometry_and_five_axis_isolation",
        "pretraining_independence": "must_audit_separately_not_established_by_project_exclusion",
        "no_model_scoring": True, "no_AF2_labels": True, "no_frozen_final_rerun": True,
        "no_threshold_relaxation_after_counts": True,
        "analysis_before_future_scoring": "freeze_new_evaluation_protocol_and_power_plan_before_any_surviving_candidate_scoring"}
    freeze(OUT / "protocol.json", protocol)
    return ref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "acquire", "materialize"), default="prepare")
    args = parser.parse_args()
    ref = prepare()
    if args.stage == "prepare":
        print(f"Prepared {len(ref['records'])} sequence references, {len(ref['exact_exposed_pdb_ids'])} exact exclusions")
        return
    if not (OUT / "snapshot/acquisition.json").exists():
        acquire(OUT / "query.json", OUT / "protocol.json", OUT / "snapshot")
    if not (OUT / "discovery.json").exists():
        discover(OUT / "snapshot", OUT / "discovery.json", include_viral=True)
    discovery = read(OUT / "discovery.json")
    filtered = copy.deepcopy(discovery)
    exposed = set(ref["exact_exposed_pdb_ids"])
    removed = []
    for entry in filtered["entries"]:
        if entry["status"] == "candidate" and entry["pdb_id"].lower() in exposed:
            entry["status"] = "excluded_prior_metadata_exposure"
            removed.append(entry["pdb_id"])
    filtered["prior_exposure_filter"] = {"reference_sha256": digest(OUT / "reference_union.json"), "removed_candidates": removed}
    filtered["counts_after_exact_exclusion"] = {"remaining_candidates": sum(e["status"] == "candidate" for e in filtered["entries"]), "excluded_prior_candidates": len(removed)}
    freeze(OUT / "discovery_unexposed.json", filtered)
    print(json.dumps({"discovery": discovery["counts"], "exact_exclusion": filtered["counts_after_exact_exclusion"]}, indent=2), flush=True)
    if args.stage == "materialize" and not (OUT / "structures/structural_manifest.json").exists():
        materialize(OUT / "discovery_unexposed.json", OUT / "structures")


if __name__ == "__main__":
    main()
