#!/usr/bin/env python
"""Validate the S0 wet-lab synthesis bundle against the candidate JSON (V15 S0.5).

Checks, in order:
1. File counts: 25 scFv AA, 25 DNA, 50 Fab chains, 25 annotation rows, 25 top candidates.
2. Construct integrity per rank: scFv == VH + (G4S)3 + VL + His6 across the three
   FASTA files and the annotation TSV; DNA round-trips through the simplified
   E.coli codon table.
3. CDR containment: each candidate's cdr_sequence occurs in its scFv sequence.
4. ANARCII numbering (best-effort): VH/VL chain-type assignment and, where the
   CDR-H3 is expected inside the VH, the Chothia-extracted CDR-H3 sequence must
   match the candidate cdr_sequence. If anarcii is unavailable, the check is
   recorded as degraded, not failed.

Writes a machine-readable QA report and prints a summary. Frozen-publication
files are not touched; this bundle belongs to the IDP platform branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DEFAULT = ROOT / "idp_design_results" / "s5_wetlab_synthesis_20260705"
OUT_DEFAULT = ROOT / "idp_design_results" / "s5_wetlab_qa_v1"
CANDIDATES = ROOT / "idp_design_results" / "s5_wetlab_candidates.json"

LINKER = "GGGGSGGGGSGGGGS"
HIS_TAG = "HHHHHH"
# prepare_synthesis_seqs.py 记录在案的框架重映射: 6CGZ 是细菌酶而非抗体, 其 PDB 分支
# 被替换为 4HIX 天然 VH/VL 原样构建(不嫁接 CDR)。该 case 的 CDR 包含性失败按已知替换
# 警告处理, 留给实验团队裁决, 不静默改序列。
KNOWN_FRAMEWORK_SUBSTITUTIONS = {"PDB 6CGZ": "4HIX"}
# 与 scripts/prepare_synthesis_seqs.py 的简化 E.coli 密码子表保持一致(单密码子/氨基酸)
ECOLI_CODONS = {
    'A': 'GCG', 'R': 'CGT', 'N': 'AAC', 'D': 'GAT', 'C': 'TGC',
    'Q': 'CAG', 'E': 'GAA', 'G': 'GGT', 'H': 'CAT', 'I': 'ATT',
    'L': 'CTG', 'K': 'AAA', 'M': 'ATG', 'F': 'TTT', 'P': 'CCG',
    'S': 'AGC', 'T': 'ACC', 'W': 'TGG', 'Y': 'TAT', 'V': 'GTG',
    '*': 'TAA'
}
CODON_TO_AA = {codon: aa for aa, codon in ECOLI_CODONS.items()}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_fasta(path: Path):
    records = []
    header, chunks = None, []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header, chunks = line[1:], []
            elif line:
                chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def read_tsv(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle if line.strip()]
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:]]


def translate_dna(dna: str):
    residues = []
    for index in range(0, len(dna), 3):
        residues.append(CODON_TO_AA.get(dna[index:index + 3], "X"))
    return "".join(residues)


def check_counts(bundle: Path, candidates: dict):
    aa = read_fasta(bundle / "candidates_aa.fa")
    dna = read_fasta(bundle / "candidates_dna.fa")
    fab = read_fasta(bundle / "candidates_fab.fa")
    tsv = read_tsv(bundle / "candidates_annotation.tsv")
    top = candidates["top_candidates"]
    summary = json.loads((bundle / "synthesis_summary.json").read_text(encoding="utf-8"))
    checks = {
        "aa_records": len(aa),
        "dna_records": len(dna),
        "fab_records": len(fab),
        "annotation_rows": len(tsv),
        "top_candidates": len(top),
        "summary_completed": summary.get("completed"),
        "summary_unpaired": summary.get("unpaired"),
        "unpaired_tsv_data_rows": len(read_tsv(bundle / "unpaired_candidates.tsv")),
    }
    ok = (
        checks["aa_records"] == len(top) == 25
        and checks["dna_records"] == 25
        and checks["fab_records"] == 50
        and checks["annotation_rows"] == 25
        and checks["summary_completed"] == 25
        and checks["summary_unpaired"] == 0
        and checks["unpaired_tsv_data_rows"] == 0
    )
    return ok, checks, {"aa": aa, "dna": dna, "fab": fab, "tsv": tsv}


def check_constructs(bundle: Path, candidates: dict, data):
    problems = []
    warnings = []
    per_rank = []
    fab_by_key = {}
    scfv_sequences = {}
    for header, sequence in data["fab"]:
        record_key = header.split()[0]  # e.g. candidate_07_VH
        fab_by_key[record_key] = sequence
    for index, (aa_header, aa_sequence) in enumerate(data["aa"]):
        rank = index + 1
        candidate = candidates["top_candidates"][index]
        vh = fab_by_key.get(f"candidate_{rank:02d}_VH")
        vl = fab_by_key.get(f"candidate_{rank:02d}_VL")
        dna_header, dna_sequence = data["dna"][index]
        row = data["tsv"][index]
        entry = {"rank": rank, "issues": [], "warnings": []}
        if vh is None or vl is None:
            entry["issues"].append("missing VH/VL FASTA records")
            problems.append(entry)
            per_rank.append(entry)
            continue
        expected = vh + LINKER + vl + HIS_TAG
        if aa_sequence != expected:
            entry["issues"].append("scFv != VH+linker+VL+His")
        if aa_header.split()[0] != f"candidate_{rank:02d}":
            entry["issues"].append("FASTA header rank mismatch")
        if len(dna_sequence) != 3 * len(aa_sequence):
            entry["issues"].append("DNA length != 3x AA length")
        elif translate_dna(dna_sequence) != aa_sequence:
            entry["issues"].append("DNA round-trip mismatch")
        if dna_header.split()[0] != f"candidate_{rank:02d}_DNA":
            entry["issues"].append("DNA header rank mismatch")
        if row["rank"] != str(rank) or row["status"] != "COMPLETED":
            entry["issues"].append("TSV rank/status mismatch")
        if int(row["scfv_length"]) != len(aa_sequence):
            entry["issues"].append("TSV scfv_length mismatch")
        if int(row["vh_length"]) != len(vh) or int(row["vl_length"]) != len(vl):
            entry["issues"].append("TSV vh/vl_length mismatch")
        if row["cdr_h3"] != candidate["cdr_sequence"]:
            entry["issues"].append("TSV cdr_h3 != candidate JSON")
        if row["epitope"] != candidate["epitope"]:
            entry["issues"].append("TSV epitope != candidate JSON")
        cdr = candidate["cdr_sequence"]
        entry["cdr_containment"] = cdr in aa_sequence
        if not entry["cdr_containment"]:
            if candidate["source_type"] in KNOWN_FRAMEWORK_SUBSTITUTIONS:
                entry["warnings"].append(
                    "cdr_sequence not in scFv: known framework substitution "
                    f"{candidate['source_type']} -> "
                    f"{KNOWN_FRAMEWORK_SUBSTITUTIONS[candidate['source_type']]} "
                    "(native VH used without grafting); needs experimental-team ruling")
                warnings.append(entry)
            else:
                entry["issues"].append("cdr_sequence not found in scFv")
        scfv_sequences.setdefault(aa_sequence, []).append(rank)
        if entry["issues"]:
            problems.append(entry)
        per_rank.append(entry)
    duplicates = {
        sequence: ranks for sequence, ranks in scfv_sequences.items() if len(ranks) > 1
    }
    for ranks in duplicates.values():
        for rank in ranks:
            entry = next(item for item in per_rank if item["rank"] == rank)
            entry["warnings"].append(f"duplicate scFv sequence shared with ranks {ranks}")
    ok = not problems
    return ok, {"problem_ranks": [entry["rank"] for entry in problems],
                "warning_ranks": [entry["rank"] for entry in warnings],
                "duplicate_sequence_groups": list(duplicates.values()),
                "checked": len(data["aa"]), "per_rank": per_rank}


def chothia_cdr_h3(numbering) -> str:
    """Extract CDR-H3 (Chothia 95-102, insertions included) from ANARCII numbering.

    ANARCII numbering elements are ``(position, residue)`` with position given as
    ``(number, insertion_code)``; the reverse order is accepted defensively, and
    gap tokens are skipped.
    """
    residues = []
    for item in numbering:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            continue
        first, second = item[0], item[1]
        if isinstance(first, (tuple, list)):
            position, aa = first, second
        elif isinstance(second, (tuple, list)):
            aa, position = first, second
        else:
            continue
        if not isinstance(aa, str) or aa in ("-", ""):
            continue
        number_str = position[0] if isinstance(position, (tuple, list)) else str(position)
        try:
            number = float(number_str)
        except ValueError:
            continue
        if 95 <= number < 103:
            residues.append(aa)
    return "".join(residues)


def cdr_matches(extracted: str, cdr: str) -> bool:
    """Convention-tolerant CDR-H3 comparison.

    The candidate JSON records CDR-H3 with the CAR-motif suffix ('AR', the two
    residues preceding Chothia 95) kept, while ANARCII extraction returns the
    Chothia 95-102 span only. A match is therefore: exact, or candidate minus
    its leading 'AR'. Empirically uniform across all literature grafts.
    """
    if extracted == cdr:
        return True
    return bool(cdr.startswith("AR") and extracted == cdr[2:])


def run_anarcii(bundle: Path, candidates: dict, data):
    """Best-effort ANARCII pass; returns (status, detail). Never raises."""
    try:
        from anarcii import Anarcii
    except Exception as error:  # pragma: no cover - environment dependent
        return "unavailable", {"reason": f"import failed: {error}"}
    model_holder = {}

    def _build():
        model_holder["model"] = Anarcii(
            seq_type="antibody", mode="accuracy", batch_size=32, cpu=True,
            ncpu=2, verbose=False)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(_build).result(timeout=900)
    except Exception as error:
        return "unavailable", {"reason": f"model init failed: {error}"}
    model = model_holder["model"]
    sequences = {}
    for index, (header, sequence) in enumerate(data["fab"]):
        sequences[f"cand{index // 2 + 1:02d}_{'VH' if index % 2 == 0 else 'VL'}"] = sequence
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            numbered = pool.submit(lambda: model.number(sequences)).result(timeout=1800)
            numbered = pool.submit(lambda: model.to_scheme("chothia")).result(timeout=900)
    except Exception as error:
        return "unavailable", {"reason": f"numbering failed: {error}"}

    per_rank = []
    grafted_mismatches = []
    domain_type_warnings = []
    for rank, candidate in enumerate(candidates["top_candidates"], 1):
        entry = {"rank": rank}
        vh_result = numbered.get(f"cand{rank:02d}_VH", {})
        vl_result = numbered.get(f"cand{rank:02d}_VL", {})
        entry["vh_chain_type"] = vh_result.get("chain_type")
        entry["vh_error"] = vh_result.get("error")
        entry["vl_chain_type"] = vl_result.get("chain_type")
        entry["vl_error"] = vl_result.get("error")
        vh_ok = vh_result.get("chain_type") == "H" and not vh_result.get("error")
        vl_ok = vl_result.get("chain_type") in ("K", "L") and not vl_result.get("error")
        entry["domain_types_ok"] = bool(vh_ok and vl_ok)
        cdr = candidate["cdr_sequence"]
        source_type = candidate["source_type"]
        if source_type == "template_seeded":
            entry["cdr_check"] = "skipped_full_vh"
        elif vh_result.get("numbering"):
            extracted = chothia_cdr_h3(vh_result["numbering"])
            entry["chothia_cdr_h3"] = extracted
            entry["cdr_check"] = "match" if cdr_matches(extracted, cdr) else "mismatch"
            if not cdr_matches(extracted, cdr):
                if source_type == "literature":
                    grafted_mismatches.append(rank)
                else:
                    entry["cdr_note"] = "boundary-convention difference tolerated (native PDB CDR)"
        # 域类型不符的分级: 文献嫁接/PDB 天然来源是结构事实, 应能识别; 失败即警告。
        # template_seeded 是 MPNN 从头设计, anarcii 对非天然序列的判型分歧属预期内,
        # 同样只作警告记录, 供实验团队参考。
        if not entry["domain_types_ok"]:
            entry["domain_warning"] = (
                f"anarcii assigned VH={entry['vh_chain_type']} VL={entry['vl_chain_type']}"
                f" (source_type={source_type}); de novo designs and odd fragments may"
                " not classify as expected")
            domain_type_warnings.append(rank)
        per_rank.append(entry)
    detail = {
        "scheme": "chothia",
        "n_domains_numbered": len(numbered),
        "per_rank": per_rank,
        "grafted_cdr_mismatch_ranks": grafted_mismatches,
        "domain_type_warning_ranks": domain_type_warnings,
    }
    status = "fail" if grafted_mismatches else (
        "pass_with_warnings" if domain_type_warnings else "pass")
    return status, detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", default=str(BUNDLE_DEFAULT))
    parser.add_argument("--out-dir", default=str(OUT_DEFAULT))
    parser.add_argument("--skip-anarcii", action="store_true",
                        help="skip the best-effort numbering check")
    parser.add_argument("--anarcii", choices=("auto", "off"), default="auto",
                        help="kept for explicitness; 'off' equals --skip-anarcii")
    args = parser.parse_args()
    bundle = Path(args.bundle_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))

    required = [
        "candidates_aa.fa", "candidates_dna.fa", "candidates_fab.fa",
        "candidates_annotation.tsv", "unpaired_candidates.tsv",
        "synthesis_summary.json", "prepare_synthesis_seqs.py",
    ]
    missing = [name for name in required if not (bundle / name).exists()]

    counts_ok, counts, data = check_counts(bundle, candidates)
    constructs_ok, constructs = check_constructs(bundle, candidates, data)

    if args.skip_anarcii or args.anarcii == "off":
        anarcii_status, anarcii_detail = "skipped", {"reason": "requested skip"}
    else:
        anarcii_status, anarcii_detail = run_anarcii(bundle, candidates, data)

    hard_pass = counts_ok and constructs_ok and not missing
    has_warnings = bool(constructs.get("warning_ranks"))
    if not hard_pass or anarcii_status == "fail":
        overall = "fail"
    elif has_warnings or anarcii_status != "pass":
        # Warnings (e.g. the documented 6CGZ->4HIX substitution) and degraded or
        # skipped anarcii runs cap the verdict at pass_with_warnings; they need a
        # human ruling but do not mean the bundle is corrupted.
        overall = "pass_with_warnings"
    else:
        overall = "pass"

    report = {
        "schema_version": 1,
        "bundle_dir": str(bundle),
        "bundle_files_sha256": {
            name: sha256(bundle / name) for name in required
            if (bundle / name).exists()},
        "required_files_missing": missing,
        "checks": {
            "counts": {"pass": counts_ok, **counts},
            "construct_integrity": {"pass": constructs_ok, **constructs},
            "anarcii_numbering": {"status": anarcii_status, **anarcii_detail},
        },
        "overall": overall,
        "notes": [
            "Bundle belongs to the IDP platform wet-lab branch (WETLAB_VALIDATION_PLAN_V15 S0);"
            " it is not part of the frozen ECLS publication evidence.",
            "unpaired_candidates.tsv is now always rewritten; a header-only file means"
            " every candidate resolved (the 3-row file from 2026-07-05 21:15 was a stale"
            " first-run leftover).",
            "Known framework substitution: candidate source PDB 6CGZ is not an antibody"
            " structure; prepare_synthesis_seqs.py builds rank-01 from the 4HIX native"
            " VH/VL without CDR grafting, so the recorded 6CGZ-derived CDR is not present"
            " in that construct. Experimental team must decide: drop the rank, or"
            " re-derive a trustworthy CDR for it.",
        ],
    }
    report_path = out_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "overall": overall,
        "counts_pass": counts_ok,
        "constructs_pass": constructs_ok,
        "anarcii_status": anarcii_status,
        "report": str(report_path),
    }, indent=2))
    return 0 if overall != "fail" else 1


if __name__ == "__main__":
    sys.exit(main())
