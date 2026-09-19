"""Conservative short-sequence sensitivity check, retaining the original MMseqs audit."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_abbibench_structural_audit import OUT, SOURCE, read, freeze, digest
from scripts.audit_successor_v3_isolation import connected_components


def ungapped_hit(a, b, threshold, coverage=.8):
    """Find an offset with sufficient overlap on BOTH sequences and identity.

    Identity counts matches / aligned overlap. Terminal overhangs are controlled
    by coverage. This is exhaustive only over ungapped alignments, not indels.
    """
    if not a or not b:
        return None
    best = None
    for offset in range(-len(b) + 1, len(a)):
        start, stop = max(0, offset), min(len(a), offset + len(b))
        overlap = stop - start
        if overlap / len(a) < coverage or overlap / len(b) < coverage:
            continue
        matches = sum(a[i] == b[i-offset] for i in range(start, stop))
        identity = matches / overlap
        if identity >= threshold and (best is None or identity > best["identity"]):
            best = {"identity": identity, "query_coverage": overlap / len(a),
                    "target_coverage": overlap / len(b), "offset": offset}
    return best


def main():
    records = read(OUT / "structural_manifest.json")["records"]
    references = read(SOURCE.parent / "reference_union.json")["records"]
    original = read(OUT / "isolation_audit.json")
    checks, self_hits = [], {"h3": [], "antigen": []}
    for axis, field, threshold in (("h3", "cdr_h3_sequence", .5), ("antigen", "antigen_sequence", .3)):
        for r in records:
            sequences = [r[field]] if axis == "h3" else list(r["antigen_chain_sequences"].values())
            sequences = [s for s in sequences if len(s) <= 50]
            matches = []
            for ref in references:
                b = ref.get(field, "")
                for a in sequences:
                    hit = ungapped_hit(a, b, threshold)
                    if hit:
                        matches.append({"query": r["instance"], "target": ref["reference_id"], **hit})
                        break
            checks.append({"instance": r["instance"], "axis": axis, "hit_count": len(matches),
                           "best_hits": sorted(matches, key=lambda h: -h["identity"])[:3]})
            for target in records:
                bs = [target[field]] if axis == "h3" else list(target["antigen_chain_sequences"].values())
                for a in sequences:
                    for b in bs:
                        hit = ungapped_hit(a, b, threshold)
                        if hit:
                            self_hits[axis].append({"query": r["instance"], "target": target["instance"], **hit})
    rows = []
    for r in original["audit"]:
        additional = sorted({c["axis"] for c in checks if c["instance"] == r["instance"] and c["hit_count"]})
        rows.append({"instance": r["instance"], "mmseqs_failed_axes": r["failed_axes"],
                     "ungapped_failed_axes": additional,
                     "passes_combined_sequence_screen": r["passes_project_sequence_screen"] and not additional})
    # Preserve previous component edges; add short-sequence edges.
    for component in original["all_candidate_components"]:
        for member in component[1:]:
            self_hits["h3"].append({"query": component[0], "target": member, "identity": 1.0})
    components = connected_components([r["instance"] for r in rows], self_hits)
    surviving = {r["instance"] for r in rows if r["passes_combined_sequence_screen"]}
    result = {"classification": "conservative_short_sequence_overlap_sensitivity_not_validation",
              "original_audit_sha256": digest(OUT / "isolation_audit.json"),
              "reference_sha256": digest(SOURCE.parent / "reference_union.json"),
              "method": "exhaustive ungapped overlap, identity >= H3 0.5 or antigen 0.3; both coverages >=0.8; query length <=50",
              "limitation": "does not enumerate gapped alignments; absence of hits is not proof of independence",
              "checks": checks, "audit": rows, "all_components": components,
              "passing_records": sorted(surviving),
              "components_with_no_overlapping_member": [c for c in components if set(c) <= surviving]}
    freeze(OUT / "short_sequence_sensitivity.json", result)
    print(json.dumps({"passing_records": result["passing_records"], "clean_components": result["components_with_no_overlapping_member"], "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
