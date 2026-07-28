#!/usr/bin/env python
"""Template analyzer for hit vs non-hit emergent contact features.

Fill the experimental results CSV after BLI/SPR. This script then compares
contact-position chemistry and side-chain volume features between hits and
non-hits to identify variables that were not explicitly hard-coded.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


VOLUME_CLASS = {
    **{aa: "small" for aa in "GAS"},
    **{aa: "medium" for aa in "CTDNVP"},
    **{aa: "large" for aa in "EQHILKMFYRW"},
}
CHEM_CLASS = {
    **{aa: "aromatic" for aa in "FWY"},
    **{aa: "positive" for aa in "KRH"},
    **{aa: "negative" for aa in "DE"},
    **{aa: "polar" for aa in "STNQ"},
    **{aa: "hydrophobic" for aa in "AILMV"},
    "G": "small_flexible",
    "P": "proline",
    "C": "cys",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze hit/non-hit contact features")
    parser.add_argument("--library", required=True, help="top10_experimental_draft_constructs.csv")
    parser.add_argument("--results", required=True, help="binding/state specificity results CSV")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_mutations(mutation_text):
    out = []
    for token in str(mutation_text or "").split(";"):
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 3 or ">" not in parts[2]:
            continue
        native, mutant = parts[2].split(">", 1)
        out.append({"chain": parts[0], "resid": parts[1], "native": native, "mutant": mutant})
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    library = {r["construct_id"]: r for r in load_csv(args.library)}
    results = load_csv(args.results)
    feature_rows = []
    for r in results:
        cid = r["construct_id"]
        lib = library.get(cid, {})
        hit_call = str(r.get("state_specificity_call") or r.get("binding_call") or "").lower()
        is_hit = hit_call in ("hit", "binder", "oligomer_specific", "fibril_specific", "state_specific")
        for m in parse_mutations(lib.get("applied_mutations", "")):
            feature_rows.append({
                "construct_id": cid,
                "hit_group": "hit" if is_hit else "non_hit",
                "chain": m["chain"],
                "resid": m["resid"],
                "native": m["native"],
                "mutant": m["mutant"],
                "mutant_chemistry": CHEM_CLASS.get(m["mutant"], "other"),
                "mutant_volume": VOLUME_CLASS.get(m["mutant"], "other"),
            })
    counts = defaultdict(int)
    for r in feature_rows:
        key = (r["hit_group"], r["chain"], r["resid"], r["mutant_chemistry"], r["mutant_volume"])
        counts[key] += 1
    summary_rows = [
        {
            "hit_group": k[0], "chain": k[1], "resid": k[2],
            "mutant_chemistry": k[3], "mutant_volume": k[4], "count": v,
        }
        for k, v in sorted(counts.items())
    ]
    write_csv(out_dir / "hit_nonhit_contact_features.csv", feature_rows)
    write_csv(out_dir / "hit_nonhit_feature_counts.csv", summary_rows)
    with open(out_dir / "hit_nonhit_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump({"constructs_with_results": len(results), "feature_rows": len(feature_rows)}, f, indent=2)
    print(f"Wrote hit/non-hit feature analysis to {out_dir}")
    print(f"Results={len(results)} features={len(feature_rows)}")


def write_csv(path, rows):
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
