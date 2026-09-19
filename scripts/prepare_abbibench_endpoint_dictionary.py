"""Freeze literature-backed endpoint status before numeric label access."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/abbibench_sequence_mapping_v1"
PAPER = "https://arxiv.org/pdf/2506.04235v2"
ORIGINAL = "https://www.nature.com/articles/s41597-022-01779-4"


def main():
    headers = json.loads((OUT / "headers.json").read_text())
    rows = []
    for h in headers["rows"]:
        name = Path(h["file"]).stem.replace("_benchmarking_data_trimmed", "").replace("_benchmarking_data", "")
        family = name.split("_")[0]
        kind, transform, source = "unresolved", None, PAPER
        if family.startswith("aayl"):
            kind, transform = "AlphaSeq_calibrated_affinity_estimate", "already_negative_log_Kd_molar_by_benchmark_description"
            source = ORIGINAL if family != "aayl49" or "ML" not in name else "https://www.nature.com/articles/s41467-023-39022-2"
        elif family in ("3gbn", "4fqi"):
            kind, transform = "yeast_display_titration_affinity", "already_negative_log_Kd_molar_by_benchmark_description"
        elif family == "1n8z":
            kind, transform = "Kd_affinity_assay_original_method_to_verify", "already_negative_log_Kd_molar_by_benchmark_description"
        elif family in ("1mlc", "2fjg"):
            kind, transform = "selection_enrichment", "already_log_enrichment_by_benchmark_description"
        elif family == "1mhp":
            kind = "competition_ELISA_relative_affinity"
        elif family == "5a12":
            kind = "selection_enrichment_source_processing_to_verify"
        elif family == "g6":
            kind = "selection_enrichment_LC_processing_to_verify"
        rows.append({"table": h["file"], "assay_name": name, "column": "binding_score",
                     "measurement_class": kind, "reported_transform": transform,
                     "direction": "higher_is_better_benchmark_declared_not_numerically_verified",
                     "apply_additional_log": False, "source": source, "benchmark_source": PAPER,
                     "censoring_column_present": False, "replicate_id_present": False,
                     "ready_for_primary_numeric_analysis": False,
                     "unresolved": ["row-level censoring provenance absent from curated CSV", "exact original-to-curated transformation and parent mapping not yet crosswalked"]})
    payload = {"classification": "endpoint_status_frozen_before_numeric_value_analysis", "revision": headers["revision"],
               "rows": rows, "structure_provenance": {"AAYL": "AlphaFold3 predicted, AbBiBench supplement section2 PDF page18", "named_PDB": "PDB-derived coordinate subsets per benchmark; release-specific modifications require accounting"},
               "numeric_values_inspected": False, "policy": "sequence-only feasibility can proceed; unresolved numeric endpoints must not be scored"}
    text = json.dumps(payload, indent=2) + "\n"
    path = OUT / "endpoint_dictionary.json"
    if path.exists() and path.read_text() != text:
        raise RuntimeError("Refusing to rewrite endpoint snapshot")
    path.write_text(text)
    print("Endpoint status dictionary frozen for 17 tables; numeric analysis remains gated.")


if __name__ == "__main__":
    main()
