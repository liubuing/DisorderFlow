"""Build a current ECLS review archive without modifying frozen evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREPARED = "release/ecls_v1/prepared"
ARCHIVE = "dist/disorderflow-ecls-v1-reviewer-20260928.zip"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def portable_copy(node, changes, location="$"):
    """Normalize execution paths only. Record locations and hashes, never private paths."""
    if isinstance(node, dict):
        return {k: portable_copy(v, changes, f"{location}.{k}") for k, v in node.items()}
    if isinstance(node, list):
        return [portable_copy(v, changes, f"{location}[{i}]") for i, v in enumerate(node)]
    if isinstance(node, str):
        normalized = re.sub(r"/+", "/", node.replace("\\", "/"))
        if re.match(r"^(?:[A-Za-z]:/|/mnt/[a-z]/|/home/|/Users/)", normalized):
            marker = "/biological/DisorderFlow/"
            if marker.lower() in normalized.lower():
                index = normalized.lower().index(marker.lower())
                replacement = normalized[index + len(marker):]
            elif normalized.lower().endswith(("/python.exe", "/python", "/python3")):
                replacement = "python"
            else:
                replacement = "external_path/" + normalized.rsplit("/", 1)[-1]
            changes.append({"field": location, "original_string_sha256": hashlib.sha256(node.encode()).hexdigest(),
                            "replacement": replacement})
            return replacement
    return node


def build(root=ROOT):
    root = Path(root).resolve()
    prepared = root / PREPARED
    prepared.mkdir(parents=True, exist_ok=True)
    line = json.loads((root / "release/ecls_v1/publication_line.json").read_text(encoding="utf-8"))
    decision_path = root / line["primary_decision"]
    decision = json.loads(decision_path.read_text())
    if digest(root / line["primary_results"]) != decision["source_result_sha256"]:
        raise ValueError("Original final result no longer matches frozen decision")
    ledger = {"schema_version": 1, "scope": "derived portable copies; numerical values unchanged",
              "sources": {}}
    data_sources = {
        "results/publication/h3_ecls_temporal_final_v1/results.json": "temporal_results_portable.json",
        "results/publication/h3_ecls_adaptation_v1/results.json": "adaptation_results_portable.json",
        "results/publication/h3_candidate_reranking_dev_v1/results.json": "candidate_results.json",
        "results/publication/h3_generator_calibration_v1/analysis.json": "calibration_analysis.json",
        "results/publication/h3_ecls_statistical_summary_v1/analysis.json": "statistical_sensitivity.json",
    }
    for source, name in data_sources.items():
        changes = []
        original = json.loads((root / source).read_text(encoding="utf-8"))
        public = portable_copy(original, changes)
        write_json(prepared / name, public)
        ledger["sources"][source] = {"original_sha256": digest(root / source),
            "public_path": "evidence/" + name, "public_sha256": digest(prepared / name),
            "path_changes": changes, "scientific_values_unchanged": True}
    audit_path = root / "data/peptide_h3_publication_split_v4/audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    ids = {r["id"] for r in json.loads((prepared / "adaptation_results_portable.json").read_text())["results"]}
    assignments = {}
    for row in audit["records"]["adaptation"]:
        if row["id"] in ids:
            values = row["axis_values"].get("official_antigen_cluster") or row["axis_values"]["pdb_id"]
            assignments[row["id"]] = values[0]
    if set(assignments) != ids or len(set(assignments.values())) != 46:
        raise ValueError("Adaptation assignments do not match 46 independent inference units")
    write_json(prepared / "adaptation_cluster_assignments.json", assignments)
    ledger["sources"]["data/peptide_h3_publication_split_v4/audit.json"] = {
        "original_sha256": digest(audit_path), "public_path": "evidence/adaptation_cluster_assignments.json",
        "public_sha256": digest(prepared / "adaptation_cluster_assignments.json"),
        "transformation": "selected adaptation record IDs and their frozen inference-unit assignments"}
    write_json(prepared / "SOURCE_PROVENANCE.json", ledger)
    mapping = {
        "release/ecls_v1/PUBLIC_ARCHIVE_README.md": "README.md",
        "LICENSE.md": "LICENSE.md",
        "release/ecls_v1/LICENSE_SCOPE.md": "LICENSE_SCOPE.md",
        "publication/MANUSCRIPT_DRAFT.md": "manuscript/ECLS_MANUSCRIPT.md",
        "publication/ECLS_MANUSCRIPT.pdf": "manuscript/ECLS_MANUSCRIPT.pdf",
        "publication/ECLS_SUPPLEMENT.md": "manuscript/ECLS_SUPPLEMENT.md",
        "publication/ECLS_SUPPLEMENT.pdf": "manuscript/ECLS_SUPPLEMENT.pdf",
        "publication/figures/ecls_study_design.png": "manuscript/figures/ecls_study_design.png",
        "publication/figures/ecls_study_design.pdf": "manuscript/figures/ecls_study_design.pdf",
        "publication/figures/ecls_study_design.svg": "manuscript/figures/ecls_study_design.svg",
        "publication/figures/ecls_frozen_evidence.png": "manuscript/figures/ecls_frozen_evidence.png",
        "publication/figures/ecls_frozen_evidence.pdf": "manuscript/figures/ecls_frozen_evidence.pdf",
        "publication/figures/ecls_frozen_evidence.svg": "manuscript/figures/ecls_frozen_evidence.svg",
        "release/ecls_v1/figure_source_data.json": "figure_source_data.json",
        "scripts/plot_ecls_narrative.py": "plot_ecls_narrative.py",
        line["primary_decision"]: "evidence/final_decision_original.json",
        "publication/ECLS_SCOPE_FREEZE.yml": "evidence/scope_original.yml",
        "configs/benchmarks/peptide_h3_ecls_adaptation_v1.yml": "protocols/adaptation_original.yml",
        "configs/benchmarks/peptide_h3_ecls_temporal_final_v1.yml": "protocols/temporal_original.yml",
        "configs/benchmarks/peptide_h3_generator_calibration_v1.yml": "protocols/calibration_original.yml",
        "scripts/verify_ecls_saved_results.py": "verify_ecls_saved_results.py",
        "release/ecls_v1/REFERENCE_AUDIT.md": "REFERENCE_AUDIT.md",
        "release/ecls_v1/CLAIM_EVIDENCE_MAP.md": "CLAIM_EVIDENCE_MAP.md",
        "release/ecls_v1/primary_results.csv": "primary_results.csv",
        PREPARED + "/SOURCE_PROVENANCE.json": "SOURCE_PROVENANCE.json",
    }
    for name in [*data_sources.values(), "adaptation_cluster_assignments.json"]:
        mapping[PREPARED + "/" + name] = "evidence/" + name
    files = []
    for source, name in mapping.items():
        path = root / source
        if not path.is_file():
            raise FileNotFoundError(source)
        files.append({"path": name, "bytes": path.stat().st_size, "sha256": digest(path)})
    manifest = {"schema_version": 1, "release_id": "disorderflow-ecls-v1",
                "manuscript_title": line["manuscript_title"],
                "status": "author_review_candidate_not_published", "files": files}
    write_json(prepared / "MANIFEST.json", manifest)
    archive = root / ARCHIVE
    archive.parent.mkdir(parents=True, exist_ok=True)
    # Never package old reviewer ZIPs, checkpoints or broad working directories.
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for source, name in sorted(mapping.items(), key=lambda row: row[1]):
            info = zipfile.ZipInfo(name, (2026, 9, 28, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, (root / source).read_bytes())
        info = zipfile.ZipInfo("MANIFEST.json", (2026, 9, 28, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, (prepared / "MANIFEST.json").read_bytes())
    summary = {"schema_version": 1, "status": "local_author_review_candidate",
               "archive": ARCHIVE, "sha256": digest(archive), "bytes": archive.stat().st_size,
               "files": len(files) + 1, "original_final_sha256": decision["source_result_sha256"],
               "path_strings_normalized": sum(len(v.get("path_changes", [])) for v in ledger["sources"].values()),
               "model_forward_performed": False, "remote_publication_confirmed": False}
    write_json(root / "release/ecls_v1/submission_package_manifest.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
