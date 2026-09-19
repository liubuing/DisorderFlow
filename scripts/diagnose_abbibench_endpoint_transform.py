"""Reproduce the post-hoc double-log processing diagnostic, without model scores."""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/abbibench_sequence_mapping_v1"


def main():
    crosswalk = json.loads((OUT / "AAYL_endpoint_crosswalk.json").read_text())
    summary, rows = defaultdict(Counter), []
    for row in crosswalk["rows"]:
        if not row["unambiguous_source"]:
            continue
        original_mean = 9 - row["original_groups"][0]["expected_benchmark_score"]
        alternative = 9 - math.log10(original_mean) if original_mean > 0 else None
        residual = row["curated_score"] - alternative if alternative is not None else None
        matches = residual is not None and abs(residual) <= 1e-5
        summary[row["table"]]["unambiguous"] += 1
        summary[row["table"]]["double_log_formula_matches"] += int(matches)
        rows.append({"table": row["table"], "source_data_row": row["source_data_row"],
                     "mean_original_log10_nM": original_mean, "alternative_formula_residual": residual, "matches": matches})
    result = {"classification": "posthoc_processing_diagnostic_not_model_result",
        "formula": "curated score approximately equals 9 - log10(mean(original Pred_affinity)); original Pred_affinity is already log10(nM)",
        "interpretation": "consistent with applying a second logarithm; not proof of upstream implementation without its preprocessing code",
        "ranking_caveat": "monotonic for positive original means, so within that subset Spearman ranks are unchanged; values and nonpositive-domain selection remain problematic",
        "summary": dict(summary), "rows": rows}
    path = OUT / "AAYL_transform_diagnosis.json"
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError("Diagnostic differs from saved result")
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(summary), indent=2))


if __name__ == "__main__":
    main()
