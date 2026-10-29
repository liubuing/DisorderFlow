"""Reproduce the post-hoc double-log processing diagnostic, without model scores."""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/abbibench_sequence_mapping_v1"


def main():
    # 复现性诊断：检验 AAYL 终点"curated score"是否等于对原始分数再取一次对数。
    # 原始 expected_benchmark_score 已是 log10(nM)（9 - x 还原为 nM 的对数均值），
    # 若 9 - log10(原均值) 与 curated 分数在 1e-5 内一致，即证明发生了二次对数变换。
    # 只用 unambiguous_source 的行，避免映射歧义污染结论。
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
    # 结论边界写死在产物里：二次对数对正值单调，Spearman 秩不变，但数值与
    # 非正定义域的样本选择仍有问题；且无上游预处理代码不能定罪实现本身
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
