#!/usr/bin/env python
"""Run five-axis isolation for the retrospective successor-v3 exploratory cohort."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_successor_v3_isolation import (  # noqa: E402
    AXES,
    COVERAGE,
    COVERAGE_MODE,
    MINIMUM_COMPONENTS,
    connected_components,
    exact_pdb_exclusion,
    identity_fraction,
    normalize_pdb,
    select_donor,
    select_representative,
    set_sha256,
    sha256_file,
    validate_candidates,
    write_fasta,
    write_json_once,
)


DEFAULT_MMSEQS = "/home/liubuing/.local/mmseqs2/usr/bin/mmseqs-avx2"
DEFAULT_LIBRARY = "/home/liubuing/.local/mmseqs2/usr/lib/x86_64-linux-gnu"


def wsl_path(path):
    resolved = Path(path).resolve().as_posix()
    if len(resolved) >= 3 and resolved[1:3] == ":/":
        return f"/mnt/{resolved[0].lower()}/{resolved[3:]}"
    return resolved


def wsl_command(mmseqs, arguments, library=DEFAULT_LIBRARY):
    shell = "LD_LIBRARY_PATH=" + library + " " + " ".join(
        [mmseqs] + [str(value) for value in arguments])
    return ["wsl.exe", "-d", "Debian", "--", "bash", "-lc", shell]


def mmseqs_version(mmseqs):
    command = wsl_command(mmseqs, ["version"])
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"Cannot obtain MMseqs2 version: {completed.stderr[-2000:]}")
    return (completed.stdout or completed.stderr).strip(), command


def run_search(mmseqs, query, target, output, work, threshold, threads):
    arguments = [
        "easy-search", wsl_path(query), wsl_path(target), wsl_path(output),
        wsl_path(work), "--min-seq-id", threshold, "-c", COVERAGE,
        "--cov-mode", COVERAGE_MODE,
        "--format-output", "query,target,fident,qcov,tcov,evalue",
        "--threads", threads,
    ]
    command = wsl_command(mmseqs, arguments)
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"MMseqs2 failed: {completed.stderr[-2000:]}")
    hits = []
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            query_id, target_id, identity, qcov, tcov, evalue = line.split("\t")
            hits.append({
                "query": query_id,
                "target": target_id,
                "identity": float(identity),
                "query_coverage": float(qcov),
                "target_coverage": float(tcov),
                "evalue": float(evalue),
            })
    return hits, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mmseqs-wsl", default=DEFAULT_MMSEQS)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    candidate_path = ROOT / args.candidate_manifest
    reference_path = ROOT / args.reference_manifest
    out_path = ROOT / args.out
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite exploratory audit: {out_path}")

    candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
    if candidate_payload.get("classification") != "model_free_structural_eligibility":
        raise ValueError("Candidate manifest is not model-free structural eligibility")
    if reference_payload.get("status") != "frozen_successor_v3_confirmatory_reference_union":
        raise ValueError("Reference manifest is not the frozen successor-v3 union")
    candidates, references = candidate_payload["records"], reference_payload["records"]
    validate_candidates(candidates)
    exact_eligible, exact_excluded = exact_pdb_exclusion(
        candidates, reference_payload["exact_exposed_pdb_ids"])
    version, version_command = mmseqs_version(args.mmseqs_wsl)

    reference_hits, self_hits, commands = {}, {}, []
    with tempfile.TemporaryDirectory(
            prefix="successor_v3_exploratory_", ignore_cleanup_errors=True) as name:
        temporary = Path(name)
        for axis, (field, threshold) in AXES.items():
            query = temporary / f"candidate_{axis}.fasta"
            target = temporary / f"reference_{axis}.fasta"
            write_fasta(query, exact_eligible, field, "instance")
            n_references = write_fasta(target, references, field, "reference_id")
            if n_references and exact_eligible:
                reference_hits[axis], command = run_search(
                    args.mmseqs_wsl, query, target,
                    temporary / f"reference_{axis}.tsv",
                    temporary / f"reference_work_{axis}", threshold, args.threads)
                commands.append(command)
            else:
                reference_hits[axis] = []
            if exact_eligible:
                self_hits[axis], command = run_search(
                    args.mmseqs_wsl, query, query,
                    temporary / f"candidate_{axis}.tsv",
                    temporary / f"candidate_work_{axis}", threshold, args.threads)
                commands.append(command)
            else:
                self_hits[axis] = []

    failed_by_id, best_hits = {}, {}
    for axis, hits in reference_hits.items():
        for hit in sorted(hits, key=lambda row: (
                row["query"], -identity_fraction(row["identity"]), row["target"])):
            failed_by_id.setdefault(hit["query"], set()).add(axis)
            if len(best_hits.setdefault((hit["query"], axis), [])) < 3:
                best_hits[(hit["query"], axis)].append(hit)
    independent = [row for row in exact_eligible if row["instance"] not in failed_by_id]
    components = connected_components(
        [row["instance"] for row in independent], self_hits) if independent else []
    by_id = {row["instance"]: row for row in independent}
    component_by_id = {
        member: f"SV3E{index:03d}"
        for index, members in enumerate(components, 1) for member in members
    }
    representatives = [
        select_representative([by_id[value] for value in members]) for members in components]
    frozen_components = []
    for index, (members, representative) in enumerate(zip(components, representatives), 1):
        donor = (select_donor(representative, representatives, component_by_id)
                 if len(components) >= 2 else None)
        frozen_components.append({
            "component_id": f"SV3E{index:03d}",
            "members": members,
            "representative_id": representative["instance"],
            "donor_id": donor["instance"] if donor else None,
            "donor_component_id": component_by_id[donor["instance"]] if donor else None,
        })

    exact_ids = {row["instance"] for row in exact_excluded}
    audit_rows = []
    for row in sorted(candidates, key=lambda item: item["instance"]):
        axes = sorted(failed_by_id.get(row["instance"], set()))
        if row["instance"] in exact_ids:
            axes.insert(0, "pdb_id")
        audit_rows.append({
            "instance": row["instance"],
            "pdb_id": normalize_pdb(row["pdb_id"]),
            "independent": not axes,
            "failed_axes": axes,
            "best_reference_hits": {
                axis: best_hits.get((row["instance"], axis), []) for axis in AXES},
        })
    gate_passed = len(components) >= MINIMUM_COMPONENTS
    report = {
        "schema_version": 1,
        "status": ("exploratory_model_free_isolation_ready" if gate_passed
                   else "exploratory_model_free_isolation_insufficient"),
        "classification": "retrospective_exploratory; not confirmatory or external",
        "claim_boundary": "five-axis sequence isolation only; model-free exploratory panel",
        "inputs": {
            "candidate_manifest": args.candidate_manifest.as_posix(),
            "candidate_manifest_sha256": sha256_file(candidate_path),
            "reference_manifest": args.reference_manifest.as_posix(),
            "reference_manifest_sha256": sha256_file(reference_path),
        },
        "normalized_sets_sha256": {
            "candidate_ids": set_sha256(row["instance"] for row in candidates),
            "candidate_pdb_ids": set_sha256(normalize_pdb(row["pdb_id"])
                                             for row in candidates),
            "independent_ids": set_sha256(row["instance"] for row in independent),
            "representative_ids": set_sha256(row["instance"] for row in representatives),
        },
        "thresholds": {
            axis: {"minimum_identity": threshold, "minimum_coverage": COVERAGE,
                   "coverage_mode": COVERAGE_MODE}
            for axis, (_field, threshold) in AXES.items()
        },
        "mmseqs": {"version": version, "backend": "WSL2 Debian",
                   "version_command": version_command},
        "counts": {
            "candidates": len(candidates),
            "exact_pdb_excluded": len(exact_excluded),
            "reference_homology_excluded": len(failed_by_id),
            "independent": len(independent),
            "components": len(components),
            "representatives": len(representatives),
        },
        "excluded_by_axis": dict(sorted(Counter(
            axis for axes in failed_by_id.values() for axis in axes).items())),
        "minimum_required_components": MINIMUM_COMPONENTS,
        "gate_passed": gate_passed,
        "decision": ("eligible_for_exploratory_evaluation" if gate_passed
                     else "insufficient_independent_components; evaluation_not_run"),
        "components": frozen_components,
        "representatives": representatives,
        "audit": audit_rows,
    }
    write_json_once(out_path, report)
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
