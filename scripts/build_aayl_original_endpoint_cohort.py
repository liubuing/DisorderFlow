"""Rebuild a retrospective H3-only cohort from original AlphaSeq dataset 1.

Uses all sequence-eligible source rows, not the curated benchmark subset. Rules
are fixed before ECLS access; the preceding endpoint diagnostic is disclosed.
"""
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.crosswalk_abbibench_aayl_endpoints import original_rows, finite_value
from scripts.audit_abbibench_sequence_mapping import classify

OUT = ROOT / "data/aayl_original_retrospective_v1"
SOURCE = ROOT / "data/abbibench_sequence_mapping_v1"


def freeze(path, value):
    text = json.dumps(value, indent=2) + "\n"
    if path.exists() and path.read_text() != text:
        raise RuntimeError(f"Refusing to rewrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main():
    source = SOURCE / "original_alpha_seq/MITLL_AAlphaBio_Ab_Binding_dataset.csv.zip"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    expected = next(r["sha256"] for r in json.loads((SOURCE / "original_alpha_seq/schema_audit.json").read_text())["rows"] if r["file"] == source.name)
    assert source_hash == expected
    records = json.loads((ROOT / "data/abbibench_structural_audit_v1/structural_manifest.json").read_text())["records"]
    structures = {"AAYL49": next(r for r in records if r["instance"] == "ABB009"),
                  "AAYL51": next(r for r in records if r["instance"] == "ABB011")}
    protocol = {"classification": "retrospective_original_AlphaSeq_H3_functional_extension",
        "source_sha256": source_hash, "source_revision": "72218839adc6abfe26aae1fde35d5d7b390a0459",
        "selection": "all MIT_Target records with POI prefix AAYL49_ or AAYL51_, exact published partner light chain and substitutions restricted to Chothia H3",
        "reason_for_new_source": "curated endpoint conversion discrepancy diagnosed before any ECLS scoring; do not condition inclusion on curated table membership",
        "known_label_exposure": "612 unambiguous curated/source pairs numerically inspected during provenance audit; retrospective, not blinded confirmation",
        "endpoint": "9 - median(original log10 estimated Kd in nM); AlphaSeq assay-derived estimate, higher is stronger",
        "primary_replicates": "exactly three distinct replicate IDs 1,2,3 with finite values, within same assay and POI",
        "duplicates": "any duplicate replicate ID excludes the group; duplicated H/L pairs across source groups exclude all ambiguous groups",
        "missing": "retain group counts, exclude incomplete groups from continuous primary analysis, no imputation; conclusions restricted to complete readings",
        "negative_original_log_values": "valid finite values, retain; never take logarithm again",
        "WT": "exclude zero-mutant parent from primary mutant correlation",
        "minimum_unique_variants_per_assay": 20,
        "methods": ["ECLS", "negative_complex_H3_NLL", "negative_apo_H3_NLL"],
        "primary_contrast": "Spearman(ECLS,endpoint) minus Spearman(negative_complex_H3_NLL,endpoint)",
        "seeds": [0,2281,20260916], "no_training": True,
        "independent_antigens": 1, "inference": "descriptive within-lineage correlations; no antigen-level significance or confirmatory generalization",
        "predicted_structures": True, "source_ML_dataset2": "kept separate, not merged into primary cohort"}
    freeze(OUT / "protocol.json", protocol)
    groups = defaultdict(list)
    classifications = Counter()
    for row in original_rows(source):
        family = next((f for f in structures if row["POI"].startswith(f + "_")), None)
        if family is None or row["Target"] != "MIT_Target":
            continue
        r = structures[family]
        status = classify(row["HC"], row["LC"], r["sequences"][r["heavy_chain"]], r["sequences"][r["light_chain"]], r["h3_indices_zero_based_observed_heavy"])
        classifications[family + ":" + status] += 1
        if status != "H3_only_coordinate_compatible":
            continue
        key = (family, row["Assay"], row["POI"], row["HC"], row["LC"])
        groups[key].append({"replicate": row["Replicate"], "original_log10_nM": finite_value(row["Pred_affinity"])})
    pair_counts = Counter((k[0], k[3], k[4]) for k in groups)
    eligible, rejected = [], []
    for (family, assay, poi, hc, lc), observations in sorted(groups.items()):
        ids = [o["replicate"] for o in observations]
        values = [o["original_log10_nM"] for o in observations]
        reason = None
        if pair_counts[(family, hc, lc)] != 1:
            reason = "ambiguous_sequence_groups"
        elif len(ids) != 3 or set(ids) != {"1", "2", "3"}:
            reason = "replicate_ids_incomplete_or_duplicate"
        elif any(v is None for v in values):
            reason = "missing_or_nonfinite_reading"
        record = {"family": family, "assay": assay, "poi": poi, "heavy_sequence": hc, "light_sequence": lc, "observations": observations}
        if reason:
            rejected.append({**record, "reason": reason})
        else:
            indices = structures[family]["h3_indices_zero_based_observed_heavy"]
            eligible.append({**record, "endpoint": 9-statistics.median(values), "h3_sequence": "".join(hc[i] for i in indices)})
    summaries = []
    for family in structures:
        kept = [r for r in eligible if r["family"] == family]
        dropped = [r for r in rejected if r["family"] == family]
        summaries.append({"family": family, "eligible_unique_variants": len(kept), "excluded_groups": len(dropped),
                          "excluded_reasons": dict(Counter(r["reason"] for r in dropped)),
                          "retained_with_nonpositive_mean_original_log10_nM": sum(statistics.mean(o["original_log10_nM"] for o in r["observations"]) <= 0 for r in kept)})
    freeze(OUT / "cohort.json", {"protocol_sha256": hashlib.sha256((OUT / "protocol.json").read_bytes()).hexdigest(),
        "structures": structures, "summary": summaries, "source_row_classification_counts": dict(classifications),
        "eligible": eligible, "excluded": rejected, "ready_for_descriptive_scoring": all(r["eligible_unique_variants"] >= 20 for r in summaries)})
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
