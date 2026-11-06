"""flex_campaign_v1 数据先行扫描（只读，不下载结构、不跑模型）。

目的：在写任何战役方案之前，先量化 v21 没覆盖的数据缺口——
单域/非配对 H/L 骨架（nanobody、scFv）结合短肽或小蛋白抗原的公共存量，
以及带亲和力注释的条目数。输入是本地 SAbDab all-summary 快照。

输出：results/flex_campaign_v1_scaffold_scan.json
"""
import csv
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TSV = REPO / "data" / "sabdab_summary_all.tsv"
OUT = REPO / "results" / "flex_campaign_v1_scaffold_scan.json"

PEPTIDE_TYPES = {"peptide", "peptide-like"}
SHORT_PROTEIN_MAX_LEN = 50  # v21 的执行范围也是 5-50 残基


def split_field(value: str) -> list[str]:
    return [p.strip() for p in (value or "").split("|") if p.strip() and p.strip() != "NA"]


def main() -> None:
    rows = list(csv.DictReader(TSV.open(encoding="utf-8", newline=""), delimiter="\t"))
    stats = {
        "total_rows": len(rows),
        "paired_HL": 0,
        "no_light_chain": 0,
        "scfv_true": 0,
        "no_light_by_antigen_kind": Counter(),
        "peptide_antigen": 0,
        "nanobody_or_scfv_peptide_antigen": 0,
        "nanobody_or_scfv_short_protein_antigen": 0,
        "with_affinity_total": 0,
        "with_affinity_peptide_antigen": 0,
        "with_affinity_nanobody_peptide": 0,
        "methods": Counter(),
    }
    candidate_pdbs = []
    for row in rows:
        h = (row.get("Hchain") or "").strip()
        l = (row.get("Lchain") or "").strip()
        has_pair = bool(h) and h != "NA" and bool(l) and l != "NA"
        is_scfv = (row.get("scfv") or "").strip().lower() == "true"
        if has_pair:
            stats["paired_HL"] += 1
        else:
            stats["no_light_chain"] += 1
        if is_scfv:
            stats["scfv_true"] += 1

        ag_types = {t.lower() for t in split_field(row.get("antigen_type"))}
        ag_chains = split_field(row.get("antigen_chain"))
        has_peptide = bool(ag_types & PEPTIDE_TYPES)
        has_short_protein = "protein" in ag_types and bool(ag_chains) and not has_peptide
        affinity = (row.get("affinity") or "").strip()
        has_affinity = affinity not in ("", "None", "NA")

        if has_peptide:
            stats["peptide_antigen"] += 1
        if has_affinity:
            stats["with_affinity_total"] += 1
            if has_peptide:
                stats["with_affinity_peptide_antigen"] += 1
        if not has_pair and has_peptide:
            stats["nanobody_or_scfv_peptide_antigen"] += 1
            stats["no_light_by_antigen_kind"]["peptide"] += 1
            if has_affinity:
                stats["with_affinity_nanobody_peptide"] += 1
            candidate_pdbs.append(
                {
                    "pdb": row["pdb"],
                    "date": row.get("date"),
                    "method": row.get("method"),
                    "resolution": row.get("resolution"),
                    "antigen_chain": row.get("antigen_chain"),
                    "antigen_name": row.get("antigen_name"),
                    "affinity": affinity or None,
                    "engineered": row.get("engineered"),
                    "pmid": row.get("pmid"),
                }
            )
        elif not has_pair and has_short_protein:
            stats["nanobody_or_scfv_short_protein_antigen"] += 1
            stats["no_light_by_antigen_kind"]["short_protein"] += 1
        stats["methods"][row.get("method") or "unknown"] += 1

    out = {
        "source": str(TSV),
        "note": "只读快照统计；候选列表未做坐标/链身份核验，资格门槛另行执行",
        "stats": {k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.items()},
        "candidate_pdbs_no_light_peptide": candidate_pdbs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {k: v for k, v in out["stats"].items() if k != "methods"}
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
