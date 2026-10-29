"""Sequence-only coordinate compatibility and H3-only feasibility; no label access."""
# 用途：仅基于序列做坐标兼容性与"仅 H3 突变"可行性审计（不接触任何数值标签）。
# 流程：读取结构审计清单与序列投影（校验 sha256 防篡改），用滑窗把表序列
# 唯一映射到观测坐标序列，再逐行分类；输出 sequence_audit.json 与
# 暂定的 H3 坐标兼容行 JSONL。共识序列仅作诊断，不代表实验亲本序列。
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/abbibench_sequence_mapping_v1"
STRUCTURES = ROOT / "data/abbibench_structural_audit_v1"
AA = set("ACDEFGHIKLMNPQRSTVWY")


def consensus(sequences):
    """对多条序列取逐位众数共识；先按最常见长度过滤，保证对齐。"""
    lengths = Counter(map(len, sequences))
    n = lengths.most_common(1)[0][0]
    selected = [s for s in sequences if len(s) == n]
    return "".join(Counter(s[i] for s in selected).most_common(1)[0][0] for i in range(n))


def window_mapping(reference, query):
    """Single best ungapped coordinate window; ambiguity and missing residues fail.

    This is a diagnostic mapping, not an inferred experimental parent. A single
    map is selected on table consensus before testing individual variants.
    """
    if len(query) > len(reference):
        return {"status": "query_longer_than_observed_coordinates"}
    scores = [(sum(a != b for a, b in zip(reference[i:i+len(query)], query)), i)
              for i in range(len(reference)-len(query)+1)]
    best = min(s[0] for s in scores)
    offsets = [i for distance, i in scores if distance == best]
    if len(offsets) != 1:
        return {"status": "ambiguous_window", "best_mismatches": best}
    offset = offsets[0]
    return {"status": "unique_window", "offset": offset, "length": len(query),
            "consensus_mismatches": [i for i, (a, b) in enumerate(zip(reference[offset:offset+len(query)], query)) if a != b]}


def classify(h, l, h_ref, l_ref, h3):
    """按严格程度递进的分类规则判定一条 (重链, 轻链) 记录的坐标兼容性。"""
    if set(h + l) - AA:
        # 含非标准氨基酸则直接排除
        return "noncanonical"
    if len(h) != len(h_ref) or len(l) != len(l_ref):
        return "length_mismatch"
    changed = {i for i, (a, b) in enumerate(zip(h, h_ref)) if a != b}
    if l != l_ref:
        return "light_chain_differs"
    if changed - set(h3):
        return "heavy_changes_outside_H3"
    if not changed:
        return "identical_to_coordinate_sequence"
    return "H3_only_coordinate_compatible"


def main():
    metadata = json.loads((STRUCTURES / "mapping.json").read_text())
    structures = json.loads((STRUCTURES / "structural_manifest.json").read_text())["records"]
    by_file = {r["file"]: r for r in structures}
    table_structure = {}
    for r in metadata["rows"]:
        for table in r["tables"]:
            if table["resolved"]:
                table_structure[table["resolved"]] = by_file[r["structure"]]
    acquisition = json.loads((OUT / "acquisition.json").read_text())
    results, eligible = [], []
    for receipt in acquisition["receipts"]:
        # 校验序列投影文件未被改动（冻结哈希一致性检查）
        path = OUT / "sequences" / (Path(receipt["file"]).stem + ".jsonl")
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["projection_sha256"]:
            raise ValueError("Sequence projection changed")
        data = [json.loads(line) for line in path.read_text().splitlines()]
        r = table_structure[receipt["file"]]
        h_full, l_full = r["sequences"][r["heavy_chain"]], r["sequences"][r["light_chain"]]
        hc, lc = consensus([x["heavy_chain_seq"] for x in data]), consensus([x["light_chain_seq"] for x in data])
        hm, lm = window_mapping(h_full, hc), window_mapping(l_full, lc)
        counts = Counter()
        accepted = []
        if hm["status"] == lm["status"] == "unique_window":
            h_ref = h_full[hm["offset"]:hm["offset"]+hm["length"]]
            l_ref = l_full[lm["offset"]:lm["offset"]+lm["length"]]
            h3 = [i-hm["offset"] for i in r["h3_indices_zero_based_observed_heavy"]]
            if any(i < 0 or i >= len(h_ref) for i in h3):
                counts["H3_outside_mapped_window"] = len(data)
            else:
                for item in data:
                    status = classify(item["heavy_chain_seq"], item["light_chain_seq"], h_ref, l_ref, h3)
                    counts[status] += 1
                    if status == "H3_only_coordinate_compatible":
                        accepted.append(item)
        else:
            counts["coordinate_mapping_unresolved"] = len(data)
        unique = {(x["heavy_chain_seq"], x["light_chain_seq"]) for x in accepted}
        result = {"table": receipt["file"], "structure": r["instance"], "rows": len(data),
                  "heavy_length_counts": dict(Counter(len(x["heavy_chain_seq"]) for x in data)),
                  "light_length_counts": dict(Counter(len(x["light_chain_seq"]) for x in data)),
                  "heavy_mapping": hm, "light_mapping": lm, "diagnostic_heavy_consensus_not_parent": hc,
                  "diagnostic_light_consensus_not_parent": lc, "classification_counts": dict(counts),
                  "coordinate_compatible_unique_pairs": len(unique),
                  # 审计边界：仅序列层面的坐标可行性，不验证实验亲本，也不可直接用于打分
                  "experimental_parent_verified": False, "ready_for_scoring": False}
        results.append(result)
        eligible.extend({"table": receipt["file"], "structure": r["instance"], **x} for x in accepted)
    payload = {"classification": "sequence_only_coordinate_feasibility_not_final_experimental_eligibility",
               "rows": results, "numeric_labels_analyzed": False,
               "total_source_rows": sum(r["rows"] for r in results),
               "provisional_H3_coordinate_compatible_rows": len(eligible),
               "caveat": "Consensus is diagnostic only. Coordinate compatibility does not establish correct experimental parent, absence of terminal indels, target strain match, or uncensored labels."}
    (OUT / "sequence_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (OUT / "provisional_H3_sequence_rows.jsonl").open("w") as handle:
        for row in eligible:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps([{k: r[k] for k in ("table", "rows", "classification_counts", "coordinate_compatible_unique_pairs")} for r in results], indent=2))


if __name__ == "__main__":
    main()
