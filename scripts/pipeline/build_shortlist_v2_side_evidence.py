#!/usr/bin/env python
"""Build shortlist v2 evidence package and Fv fold side-check inputs."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build shortlist v2 side-evidence package")
    parser.add_argument("--shortlist", required=True, help="synthesis_candidate_shortlist.csv")
    parser.add_argument("--developability", required=True, help="shortlist_developability.csv")
    parser.add_argument("--variable-evidence", required=True, help="variable_region_construct_evidence.csv")
    parser.add_argument("--variable-fasta", required=True, help="shortlist_variable_regions.fasta")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_variable_fasta(path):
    records = defaultdict(dict)
    header = None
    seq = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    store(records, header, "".join(seq))
                header = line[1:]
                seq = []
            else:
                seq.append(line)
    if header:
        store(records, header, "".join(seq))
    return records


def store(records, header, seq):
    parts = header.split("|")
    if len(parts) < 2:
        return
    construct, chain_type = parts[0], parts[1]
    if chain_type in ("heavy", "light"):
        records[construct][chain_type] = seq


def tool_status():
    try:
        alphafold_common = importlib.util.find_spec("alphafold.common") is not None
    except ModuleNotFoundError:
        alphafold_common = False
    cf_python = Path(r"C:\cf\Scripts\python.exe")
    cf_alphafold_common = False
    if cf_python.exists():
        try:
            probe = subprocess.run(
                [str(cf_python), "-c", "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('alphafold.common') else 1)"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            cf_alphafold_common = probe.returncode == 0
        except Exception:
            cf_alphafold_common = False
    colabfold_path = first_existing([
        Path(r"C:\cf\Scripts\colabfold_batch.exe"),
        shutil.which("colabfold_batch"),
    ])
    colabfold_runnable = False
    colabfold_error = "not_found"
    if colabfold_path:
        try:
            probe = subprocess.run(
                [str(colabfold_path), "--help"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            colabfold_runnable = probe.returncode == 0
            if not colabfold_runnable:
                colabfold_error = (probe.stderr or probe.stdout)[-500:]
            else:
                colabfold_error = ""
        except Exception as exc:
            colabfold_error = str(exc)
    anarci_script = first_existing([Path(r"C:\cf\Scripts\ANARCI"), shutil.which("ANARCI")])
    hmmscan_path = first_existing([Path(r"C:\cf\Scripts\hmmscan.exe"), shutil.which("hmmscan")])
    hmmscan_runnable = False
    hmmscan_error = "not_found"
    if hmmscan_path:
        try:
            probe = subprocess.run(
                [str(hmmscan_path), "-h"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            hmmscan_runnable = probe.returncode == 0
            hmmscan_error = "" if hmmscan_runnable else (probe.stderr or probe.stdout)[-500:]
        except Exception as exc:
            hmmscan_error = str(exc)
    return {
        "ANARCI": bool(anarci_script),
        "ANARCI_runnable": bool(anarci_script) and hmmscan_runnable,
        "anarci": bool(shutil.which("anarci")),
        "igblastn": bool(shutil.which("igblastn")),
        "igblastp": bool(shutil.which("igblastp")),
        "hmmscan": bool(hmmscan_path),
        "hmmscan_runnable": hmmscan_runnable,
        "hmmscan_error": hmmscan_error,
        "preferred_colabfold_batch": str(colabfold_path) if colabfold_path else "",
        "colabfold_batch": bool(colabfold_path),
        "current_python_alphafold_common_importable": alphafold_common,
        "cf_python_alphafold_common_importable": cf_alphafold_common,
        "colabfold_batch_runnable": colabfold_runnable,
        "colabfold_batch_error": colabfold_error,
        "notes": [
            "ANARCI/IgBLAST are required for synthesis-grade numbering, not for this heuristic triage.",
            "ColabFold requires a runnable colabfold_batch plus AlphaFold package dependencies.",
        ],
}


def first_existing(candidates):
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if p.exists():
            return p
    return None


def evidence_status(dev, var):
    if dev.get("developability_status") != "developability_pass":
        return "developability_review"
    if dev.get("introduced_or_candidate_specific_flags"):
        return "candidate_specific_developability_review"
    if var.get("variable_region_status") == "variable_region_review":
        return "variable_region_developability_review"
    if var.get("variable_region_status") == "boundary_review":
        return "numbering_boundary_review"
    return "ready_for_fv_fold_sidecheck"


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    shortlist = load_csv(args.shortlist)
    dev_by_id = {r["construct_id"]: r for r in load_csv(args.developability)}
    var_by_id = {r["construct_id"]: r for r in load_csv(args.variable_evidence)}
    variable_fasta = load_variable_fasta(args.variable_fasta)

    rows = []
    primary_fasta = []
    review_fasta = []
    for r in shortlist:
        cid = r["construct_id"]
        dev = dev_by_id.get(cid, {})
        var = var_by_id.get(cid, {})
        status = evidence_status(dev, var)
        row = {
            **r,
            "developability_status": dev.get("developability_status", "missing"),
            "inherited_flags": dev.get("inherited_flags", ""),
            "candidate_specific_flags": dev.get("introduced_or_candidate_specific_flags", ""),
            "variable_region_status": var.get("variable_region_status", "missing"),
            "variable_region_unresolved_flags": var.get("variable_region_unresolved_flags", ""),
            "shortlist_v2_status": status,
            "fold_sidecheck_status": "pending_not_run",
            "numbering_status": "heuristic_j_motif_boundary_only",
        }
        rows.append(row)
        heavy = variable_fasta.get(cid, {}).get("heavy", "")
        light = variable_fasta.get(cid, {}).get("light", "")
        if heavy and light:
            entry = [f">{cid}|Fv_HL_pair|status={status}", f"{heavy}:{light}"]
            if status == "ready_for_fv_fold_sidecheck":
                primary_fasta.extend(entry)
            else:
                review_fasta.extend(entry)

    rows.sort(key=lambda r: (r["shortlist_v2_status"] != "ready_for_fv_fold_sidecheck", int(r["shortlist_rank"])))
    for i, r in enumerate(rows, 1):
        r["shortlist_v2_rank"] = i

    fields = [
        "shortlist_v2_rank", "shortlist_rank", "construct_id", "reference_pdb", "candidate_id",
        "n_mutations", "candidate_score", "contact_retention", "shortlist_v2_status",
        "numbering_status", "fold_sidecheck_status", "developability_status", "inherited_flags",
        "candidate_specific_flags", "variable_region_status", "variable_region_unresolved_flags",
        "applied_mutations", "required_side_evidence",
    ]
    with open(out_dir / "synthesis_candidate_shortlist_v2.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    write_text(out_dir / "fv_fold_queue_primary.fasta", primary_fasta)
    write_text(out_dir / "fv_fold_queue_review.fasta", review_fasta)

    summary = {
        "input": len(rows),
        "ready_for_fv_fold_sidecheck": sum(1 for r in rows if r["shortlist_v2_status"] == "ready_for_fv_fold_sidecheck"),
        "review": sum(1 for r in rows if r["shortlist_v2_status"] != "ready_for_fv_fold_sidecheck"),
        "primary_fold_queue_pairs": len(primary_fasta) // 2,
        "review_fold_queue_pairs": len(review_fasta) // 2,
        "tool_status": tool_status(),
    }
    with open(out_dir / "shortlist_v2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "sidecheck_tool_status.json", "w", encoding="utf-8") as f:
        json.dump(summary["tool_status"], f, indent=2)
    write_commands(out_dir / "fold_sidecheck_commands.md")
    write_report(out_dir / "shortlist_v2_report.md", summary, rows)

    print(f"Wrote shortlist v2 side-evidence package to {out_dir}")
    print(
        f"Input={summary['input']} ready={summary['ready_for_fv_fold_sidecheck']} "
        f"review={summary['review']}"
    )


def write_text(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines) + "\n")


def write_commands(path):
    colabfold = r"C:\cf\Scripts\colabfold_batch.exe"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Fv Fold Side-Check Commands\n\n")
        f.write("Primary queue command using the detected C:\\cf ColabFold environment:\n\n")
        f.write("```bash\n")
        f.write(f"{colabfold} outputs/synthesis_candidate_shortlist_v5_nglyco_rescued/fv_fold_queue_primary.fasta outputs/synthesis_candidate_shortlist_v5_nglyco_rescued/colabfold_fv_primary --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 3 --rank auto\n")
        f.write("```\n\n")
        f.write("CPU smoke command for one paired Fv, useful before a long full queue run:\n\n")
        f.write("```bash\n")
        f.write(f"{colabfold} outputs/synthesis_candidate_shortlist_v5_nglyco_rescued/fv_fold_smoke_1.fasta outputs/synthesis_candidate_shortlist_v5_nglyco_rescued/colabfold_smoke_1 --model-type alphafold2_multimer_v3 --msa-mode single_sequence --num-models 1 --num-recycle 1 --rank auto --overwrite-existing-results\n")
        f.write("```\n")


def write_report(path, summary, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Synthesis Candidate Shortlist v2\n\n")
        f.write(
            f"Input: {summary['input']}; ready for Fv fold side-check: "
            f"{summary['ready_for_fv_fold_sidecheck']}; review: {summary['review']}\n\n"
        )
        f.write("## Tool Status\n\n")
        for k, v in summary["tool_status"].items():
            if k != "notes":
                f.write(f"- {k}: {v}\n")
        f.write("\n## Candidates\n\n")
        f.write("| v2 Rank | Construct | Status | Contact Retention | Inherited Flags | Variable Unresolved Flags |\n")
        f.write("|---:|---|---|---:|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['shortlist_v2_rank']} | {r['construct_id']} | {r['shortlist_v2_status']} | "
                f"{r['contact_retention']} | {r['inherited_flags']} | {r['variable_region_unresolved_flags']} |\n"
            )
        f.write("\nStatus remains draft_not_synthesis_ready until ANARCI/Chothia numbering and Fv fold side-checks pass.\n")


if __name__ == "__main__":
    main()
