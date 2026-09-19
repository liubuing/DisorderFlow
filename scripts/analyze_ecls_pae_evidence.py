"""Post-hoc development diagnostics; never opens frozen final/GP2 data."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_ecls_pae_joint import digest, joint_scores, read, screening, write

OUT = ROOT / "results/ecls_pae_evidence_dev_v1"


def corr(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def membership(values, k=3):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or not 1 <= k <= len(values):
        raise ValueError("Invalid top-k membership")
    cutoff = np.sort(values)[k - 1]
    lower, equal = values < cutoff, values == cutoff
    return lower.astype(float) + equal * ((k - lower.sum()) / equal.sum())


def concordance(score, target, keep=None):
    ds = np.subtract.outer(score, score)
    dy = np.subtract.outer(target, target)
    mask = np.triu(np.ones(ds.shape, dtype=bool), 1) & (dy != 0)
    if keep is not None:
        mask &= keep
    accuracy = np.where(ds == 0, .5, np.sign(ds) == np.sign(dy))
    return {"pairs": int(mask.sum()), "accuracy": float(accuracy[mask].mean()) if mask.any() else None}


def associations(group):
    y = np.array([r["target"] for r in group])
    e = np.array([r["ecls"] for r in group])
    c = np.array([r["mpnn_nll"] for r in group])
    a = c + e
    scores = {"complex_nll": c, "apo_nll": a, "negative_ecls": -e,
              "pae": [r["pae"] for r in group], "ridge": [r["ridge"] for r in group]}
    covariance = lambda x: float(np.cov(x, y, ddof=1)[0, 1])
    return {"n": len(group), "rho_lower_is_better_vs_pae": {k: corr(v, y) for k, v in scores.items()},
            "pair_concordance": {k: concordance(v, y) for k, v in scores.items()},
            "ecls_covariance_decomposition": {"apo_target": covariance(a), "complex_target": covariance(c), "ecls_target": covariance(e)},
            "apo_complex_rho": corr(a, c)}


def teacher_value(matrix, entity):
    h, l, ag, h3 = [entity[k] for k in ("heavy_sequence", "light_sequence", "antigen_sequence", "h3_sequence")]
    starts = [i for i in range(len(h)) if h.startswith(h3, i)] if h3 else []
    if len(starts) != 1 or matrix.shape != (len(h) + len(l) + len(ag),) * 2:
        raise ValueError("Ambiguous H3 or teacher shape")
    start = starts[0]
    return float(matrix[start:start + len(h3), len(h) + len(l):].mean() / 31)


def stability(grid, group):
    """grid = candidate x model x seed. All resamples are paired across candidates."""
    n, models, seeds = grid.shape
    if models != 2 or seeds != 3 or not np.isfinite(grid).all():
        raise ValueError("Expected complete 2 x 3 teacher grid")
    target = grid.mean((1, 2))
    expected = np.array([r["target"] for r in group])
    np.testing.assert_allclose(target, expected, atol=1e-8, rtol=0)
    reference = membership(target)
    replicate_vectors = grid.reshape(n, -1)
    overlaps = [float(membership(v) @ reference / 3) for v in replicate_vectors.T]
    replicates = [membership(v) for v in replicate_vectors.T]
    seed_leaveout = [grid[:, :, [j for j in range(seeds) if j != s]].mean((1, 2)) for s in range(seeds)]
    model_vectors = grid.mean(2)
    scores = {"ecls": -np.array([r["ecls"] for r in group]), "pae": np.array([r["pae"] for r in group]),
              "ridge": np.array([r["ridge"] for r in group]), "mpnn_nll": np.array([r["mpnn_nll"] for r in group])}
    scores["joint"] = joint_scores([r["ecls"] for r in group], scores["pae"])
    rng = np.random.default_rng(20260916)
    counts = np.zeros(n)
    recalls = {k: [] for k in scores}
    # Crossed model/seed bootstrap, not six independent observations.
    for _ in range(1000):
        mi = rng.integers(0, models, models)
        si = rng.integers(0, seeds, seeds)
        sampled = grid[:, mi, :][:, :, si].mean((1, 2))
        inc = membership(sampled)
        counts += inc
        for method, score in scores.items():
            recalls[method].append(float(membership(score) @ inc / 3))
    differences = grid[:, None] - grid[None, :]
    unanimous = np.all(differences > 0, axis=(2, 3)) | np.all(differences < 0, axis=(2, 3))
    boundary = np.argsort(target, kind="stable")[2:4]
    low, high = boundary
    gap = float(target[high] - target[low])
    paired_gap = grid[high] - grid[low]
    allpairs = n * (n - 1) // 2
    return {"n": n, "grid_shape": list(grid.shape),
            "replicate_top3_overlap_with_mean": overlaps,
            "mean_replicate_top3_overlap": float(np.mean(overlaps)),
            "pairwise_replicate_top3_overlap_mean": float(np.mean([a @ b / 3 for a, b in itertools.combinations(replicates, 2)])),
            "model_mean_rho": corr(model_vectors[:, 0], model_vectors[:, 1]),
            "model_mean_top3_overlap": float(membership(model_vectors[:, 0]) @ membership(model_vectors[:, 1]) / 3),
            "leave_one_seed_out_top3_overlap": [float(membership(v) @ reference / 3) for v in seed_leaveout],
            "boundary_gap_normalized": gap, "boundary_pair_replicate_gap_sd": float(paired_gap.std(ddof=1)),
            "boundary_pair_fraction_same_order": float(np.mean(paired_gap > 0)),
            "unanimous_pairs": int(np.triu(unanimous, 1).sum()), "all_pairs": allpairs,
            "unanimous_pair_concordance": {k: concordance(v, target, unanimous) for k, v in scores.items()},
            "entity_top3_resampling_frequency": dict(zip([r["id"] for r in group], (counts / 1000).tolist())),
            "recall_label_resampling_sensitivity": {k: {"mean": float(np.mean(v)), "descriptive_95_interval": np.quantile(v, [.025, .975]).tolist()} for k, v in recalls.items()}}


def main():
    source = ROOT / "results/ecls_pae_joint_dev_v1/scores.json"
    doc = read(source)
    hashes = {source.relative_to(ROOT).as_posix(): digest(source)}
    for path, expected in doc["source_sha256"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"Changed joint input: {path}")
    rows = [r for r in doc["rows"] if r["entity_type"] == "candidate"]
    decomposition = {}
    for scaffold in sorted({r["scaffold"] for r in rows}):
        group = [r for r in rows if r["scaffold"] == scaffold]
        decomposition[scaffold] = {"all_candidates": associations(group),
            "by_generator": {arm: associations([r for r in group if r["arm"] == arm]) for arm in sorted({r["arm"] for r in group})}}
    # Within-arm, within-scaffold rank centering removes between-generator shifts.
    centered = {k: [] for k in ("target", "mpnn_nll", "apo_nll", "ecls", "pae", "ridge")}
    for scaffold in decomposition:
        for arm in sorted({r["arm"] for r in rows}):
            g = [r for r in rows if r["scaffold"] == scaffold and r["arm"] == arm]
            for key in centered:
                values = [r["mpnn_nll"] + r["ecls"] if key == "apo_nll" else r[key] for r in g]
                ranks = rankdata(values)
                centered[key].extend((ranks - ranks.mean()).tolist())
    adjusted = {k: float(np.corrcoef(v, centered["target"])[0, 1]) if np.std(v) else None for k, v in centered.items() if k != "target"}
    # Positive ECLS correlation means conflict with a lower-is-better AF2 target.
    generator_summary = {}
    for arm in sorted({r["arm"] for r in rows}):
        g = [r for r in rows if r["arm"] == arm]
        generator_summary[arm] = {k: float(np.mean([r[k] for r in g])) for k in ("target", "ecls", "mpnn_nll")}
    grids = {r["id"]: {} for r in rows}
    for sub in ("af2", "af2_model2"):
        path = ROOT / f"results/candidate_interface_multiscaffold_calibration_ext/{sub}/results.json"
        af = read(path)
        hashes[path.relative_to(ROOT).as_posix()] = digest(path)
        entities = {e["entity_id"]: e for e in af["entities"]}
        for record in af["results"]:
            eid = record["entity_id"]
            if eid not in grids:
                continue
            if record["status"] != "success":
                raise ValueError("Incomplete teacher result")
            key = (sub, record["af2_seed"])
            if key in grids[eid]:
                raise ValueError("Duplicate teacher replicate")
            row = next(r for r in rows if r["id"] == eid)
            if entities[eid]["h3_sequence"] != row["h3_sequence"] or entities[eid]["antigen_sequence"] != row["antigen_sequence"]:
                raise ValueError("Teacher entity sequence mismatch")
            matrix_path = ROOT / record["pae_npz"]
            if digest(matrix_path) != record["pae_sha256"]:
                raise ValueError("Teacher matrix checksum mismatch")
            hashes[matrix_path.relative_to(ROOT).as_posix()] = digest(matrix_path)
            with np.load(matrix_path) as z:
                grids[eid][key] = teacher_value(z["pae"], entities[eid])
    keys = sorted(next(iter(grids.values())))
    seeds = sorted({seed for _, seed in keys})
    if len(keys) != 6 or len(seeds) != 3 or any(set(g) != set(keys) for g in grids.values()):
        raise ValueError("Incomplete aligned replicate grid")
    stability_report = {}
    for scaffold in decomposition:
        group = [r for r in rows if r["scaffold"] == scaffold]
        grid = np.array([[[grids[r["id"]][(m, s)] for s in seeds] for m in ("af2", "af2_model2")] for r in group])
        stability_report[scaffold] = stability(grid, group)
        print(f"{scaffold}: conflict and teacher stability analyzed", flush=True)
    hashes[Path(__file__).resolve().relative_to(ROOT).as_posix()] = digest(Path(__file__).resolve())
    report = {"status": "post_hoc_development_diagnostics_not_confirmation", "n_candidates": len(rows),
        "source_sha256": hashes, "decomposition": decomposition,
        "within_scaffold_generator_centered_rank_correlation": adjusted,
        "generator_summary_descriptive": generator_summary, "label_stability": stability_report,
        "limitations": ["Six previously inspected components, three candidates per generator/component.",
            "Only two AF2 models and three seeds; bootstrap intervals are sensitivity summaries, not calibrated confidence intervals.",
            "Unanimous ordering means agreement of these six computational replicates, not biological correctness.",
            "No tuning, sign flipping, independent confirmation, binding inference, or frozen-final rerun."]}
    write(OUT / "diagnostics.json", report)
    write(OUT / "teacher_replicates.json", {"model_seed_order": keys, "rows": [{"id": r["id"], "values": [grids[r["id"]][k] for k in keys]} for r in rows]})
    lines = ["# ECLS–PAE 开发证据诊断", "", "仅分析已查看的 6 组件 / 72 候选；所有结果为事后开发诊断。", "",
             "## 评分冲突", "", "下表为各组件内与 AF2 PAE 的 Spearman 相关。NLL、负 ECLS、PAE 均按越低越好定向；正相关表示同向。", "",
             "|组件|complex NLL|apo NLL|负 ECLS|PAE|", "|---|---:|---:|---:|---:|"]
    for sc, data in decomposition.items():
        rho = data["all_candidates"]["rho_lower_is_better_vs_pae"]
        lines.append("|" + sc + "|" + "|".join(f"{rho[k]:.3f}" if rho[k] is not None else "NA" for k in ("complex_nll", "apo_nll", "negative_ecls", "pae")) + "|")
    lines += ["", "去除组件和生成方法之间的秩均值差异后，原始 ECLS 与 PAE 标签的相关为 " + f"{adjusted['ecls']:.3f}" + "（ECLS 越高越好而标签越低越好，因此正值表示冲突）。每小组只有 3 条，不能据此作稳定性或因果结论。", "",
              "## AF2 标签稳定性", "", "|组件|单次 top3 与六次均值 top3 平均重合率|两模型 top3 重合率|六次方向一致的候选对|第3/4名均值差（PAE/31）|", "|---|---:|---:|---:|---:|"]
    for sc, d in stability_report.items():
        lines.append(f"|{sc}|{d['mean_replicate_top3_overlap']:.1%}|{d['model_mean_top3_overlap']:.1%}|{d['unanimous_pairs']}/{d['all_pairs']}|{d['boundary_gap_normalized']:.4f}|")
    lines += ["", "逐候选 top3 重采样频率、留一 seed 结果、稳定候选对排序准确率与召回敏感性详见 diagnostics.json。模型和 seed 采用交叉重采样，候选之间保持配对。区间只表示现有 2×3 计算网格的敏感性，不代表真实结合能力。", "",
              "## 文件与边界", "", "teacher_replicates.json 保存逐候选六次方向性 H3→抗原 PAE/31；均值与既有标签逐项核验。diagnostics.json 记录输入 SHA-256 和 apo/complex 协方差分解。未重新训练、调整评分方向或重跑最终集。", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
