"""Shared metrics for calibrating predicted conformational variability."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import rankdata, spearmanr


def kabsch_align(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rigidly align one C-alpha trace to another with reflection correction."""
    mobile = np.asarray(mobile, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    mobile_center = mobile.mean(axis=0)
    reference_center = reference.mean(axis=0)
    covariance = (mobile - mobile_center).T @ (reference - reference_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return (mobile - mobile_center) @ rotation.T + reference_center


def per_residue_rmsf(coordinates: np.ndarray) -> np.ndarray:
    """Compute C-alpha RMSF after aligning every model to the first model."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape (models, residues, 3)")
    if coordinates.shape[0] < 2 or coordinates.shape[1] < 3:
        raise ValueError("at least two models and three residues are required")
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates contain non-finite values")
    reference = coordinates[0]
    aligned = np.stack(
        [reference, *(kabsch_align(model, reference) for model in coordinates[1:])]
    )
    mean = aligned.mean(axis=0)
    return np.sqrt(np.mean(np.sum((aligned - mean) ** 2, axis=-1), axis=0))


def sequence_index_map(query: str, target: str) -> dict[int, int]:
    """Map exact residues between two sequences using a local pairwise alignment."""
    from Bio.Align import PairwiseAligner

    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -3.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(query, target)[0]
    mapping = {}
    for (q_start, q_end), (t_start, t_end) in zip(*alignment.aligned, strict=True):
        length = min(q_end - q_start, t_end - t_start)
        for offset in range(length):
            query_index = int(q_start + offset)
            target_index = int(t_start + offset)
            if query[query_index] == target[target_index]:
                mapping[query_index] = target_index
    return mapping


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else None


def top_fraction_recall(predicted: np.ndarray, observed: np.ndarray, fraction=0.2) -> float:
    count = max(1, int(math.ceil(len(predicted) * fraction)))
    predicted_top = set(np.argsort(predicted)[-count:])
    observed_top = set(np.argsort(observed)[-count:])
    return len(predicted_top & observed_top) / count


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Binary ROC-AUC from ranks without requiring scikit-learn."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(scores)
    value = (ranks[labels].sum() - positives * (positives + 1) / 2) / (
        positives * negatives
    )
    return float(value)


def partial_spearman(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float | None:
    """Spearman correlation of x/y after linearly removing ranked control."""
    x_rank = rankdata(x)
    y_rank = rankdata(y)
    control_rank = rankdata(control)
    design = np.column_stack([np.ones(len(control_rank)), control_rank])
    x_residual = x_rank - design @ np.linalg.lstsq(design, x_rank, rcond=None)[0]
    y_residual = y_rank - design @ np.linalg.lstsq(design, y_rank, rcond=None)[0]
    return safe_spearman(x_residual, y_residual)


def calibration_metrics(
    predicted_rmsf: np.ndarray,
    experimental_rmsf: np.ndarray,
    uncertainty: np.ndarray | None = None,
) -> dict[str, float | None]:
    predicted_rmsf = np.asarray(predicted_rmsf, dtype=np.float64)
    experimental_rmsf = np.asarray(experimental_rmsf, dtype=np.float64)
    if predicted_rmsf.shape != experimental_rmsf.shape:
        raise ValueError("predicted and experimental RMSF shapes differ")
    flexible_count = max(1, int(math.ceil(len(experimental_rmsf) * 0.2)))
    flexible = np.zeros(len(experimental_rmsf), dtype=bool)
    flexible[np.argsort(experimental_rmsf)[-flexible_count:]] = True
    result = {
        "spearman": safe_spearman(predicted_rmsf, experimental_rmsf),
        "top20_recall": top_fraction_recall(predicted_rmsf, experimental_rmsf),
        "flexible_auc": rank_auc(predicted_rmsf, flexible),
        "uncertainty_spearman": None,
        "partial_spearman": None,
    }
    if uncertainty is not None:
        uncertainty = np.asarray(uncertainty, dtype=np.float64)
        if uncertainty.shape == predicted_rmsf.shape and np.ptp(uncertainty) > 0:
            result["uncertainty_spearman"] = safe_spearman(predicted_rmsf, uncertainty)
            result["partial_spearman"] = partial_spearman(
                predicted_rmsf, experimental_rmsf, uncertainty
            )
    return result


def calibration_verdict(results: list[dict], minimum_proteins=10) -> dict:
    """Classify whether AF2-seed RMSF is fit for physical supervision."""
    valid = [result for result in results if result.get("spearman") is not None]

    def median(field):
        values = [item[field] for item in valid if item.get(field) is not None]
        return float(np.median(values)) if values else None

    summary = {
        "n_proteins": len(valid),
        "median_spearman": median("spearman"),
        "median_top20_recall": median("top20_recall"),
        "median_flexible_auc": median("flexible_auc"),
        "median_uncertainty_spearman": median("uncertainty_spearman"),
        "median_partial_spearman": median("partial_spearman"),
    }
    if valid:
        rng = np.random.default_rng(2031)
        for field in ("spearman", "top20_recall", "flexible_auc", "partial_spearman"):
            values = np.asarray(
                [item[field] for item in valid if item.get(field) is not None],
                dtype=np.float64,
            )
            if values.size:
                bootstrap = np.asarray([
                    np.median(rng.choice(values, size=len(values), replace=True))
                    for _ in range(2000)
                ])
                summary[f"median_{field}_ci95"] = [
                    float(np.quantile(bootstrap, 0.025)),
                    float(np.quantile(bootstrap, 0.975)),
                ]
    if len(valid) < minimum_proteins:
        verdict = "insufficient_evidence"
    elif (
        summary["median_spearman"] >= 0.30
        and summary["median_top20_recall"] >= 0.30
        and (
            summary["median_partial_spearman"] is None
            or summary["median_partial_spearman"] >= 0.15
        )
    ):
        verdict = "pass_physical_label"
    elif (
        summary["median_spearman"] >= 0.15
        and summary["median_top20_recall"] >= 0.25
    ):
        verdict = "rank_only"
    else:
        verdict = "fail_physical_label"
    summary["verdict"] = verdict
    return summary
