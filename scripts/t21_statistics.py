"""Statistical aggregation for the T2.1 recovery benchmark."""

# T2.1 恢复基准的统计汇总模块：对逐结构结果做 bootstrap 置信区间、
# 开发门槛(gate)判定，以及"给定接触约束 vs 随机约束 vs 空结构约束"
# 三种实验臂的对比与配对差值分析。被 T2.1 相关基准脚本复用。
import numpy as np


def bootstrap_mean_ci(values, seed, trials=10000):
    # 对均值做非参数 bootstrap，返回 95% 置信区间。
    # 默认 10000 次重采样；取排序后 2.5% / 97.5% 位置的分位数
    # （取整数下标，属于保守近似）。seed 保证结果可复现。
    rng = np.random.default_rng(int(seed))
    array = np.asarray(values)
    means = [float(np.mean(array[rng.integers(0, len(array), size=len(array))]))
             for _ in range(int(trials))]
    means.sort()
    return float(means[int(0.025 * len(means))]), float(means[int(0.975 * len(means))])


def aggregate(results, config, cluster_by_id=None):
    # 主汇总入口：
    # - results：逐结构记录（含 valid_t2_1 / torsion_hit_tier / metrics 字段）；
    # - config：冻结配置（bootstrap 参数 + development_gates 门槛）；
    # - cluster_by_id：可选的 id -> 抗原簇映射；提供时把配对差值先按簇
    #   平均再推断（推断单位为 official_antigen_cluster，避免同簇多结构
    #   造成的伪重复），否则推断单位为单结构。
    valid = [row for row in results if row.get("valid_t2_1")]
    hits = [row for row in results if row.get("torsion_hit_tier")]
    if not valid:
        # 无有效结构时无法给出统计结论，直接判门槛失败。
        return {"n_records": len(results), "n_torsion_hits": len(hits),
                "n_valid_structures": 0, "gate_pass": False}

    statistics = config["statistics"]
    trials = int(statistics["bootstrap_trials"])
    seed = int(statistics["bootstrap_seed"])

    def interval(key):
        # 对单个指标的合法值求均值与 bootstrap 95% CI；样本少于 2 个时
        # 只返回均值（CI 为 [None, None]），不做无意义的重采样。
        values = [row["metrics"][key] for row in valid if key in row.get("metrics", {})]
        if len(values) < 2:
            return float(np.mean(values)) if values else None, [None, None]
        return float(np.mean(values)), list(bootstrap_mean_ci(values, seed, trials))

    mean_contact, contact_ci = interval("mean_held_out_contact_recovery")
    mean_rmsd, rmsd_ci = interval("mean_rmsd_recovery_angstrom")
    # 两个有效率：全体记录中的有效结构占比；扭转命中(tier QC 通过)中的有效占比。
    overall_valid = len(valid) / max(1, len(results))
    post_tier_valid = len(valid) / max(1, len(hits))
    contact_values = [
        row["metrics"].get("mean_held_out_contact_recovery", 0) for row in valid]
    positive_fraction = float(np.mean(np.asarray(contact_values) > 0))

    # 开发门槛：有效结构占比达标、接触恢复 CI 下界严格为正、
    # 且正恢复结构比例不低于配置阈值。
    gates = config["development_gates"]
    passed = bool(
        overall_valid >= float(gates["minimum_valid_structure_fraction"])
        and contact_ci[0] is not None and contact_ci[0] > 0
        and positive_fraction >= float(
            gates["minimum_fraction_structures_positive_held_out_recovery"]))

    arm_comparisons = {}
    paired_contrasts = {}
    # 三个对照臂：all_contacts（全量给定接触）、random_restraints（随机约束）、
    # null_structural（空结构约束）。逐臂计算绝对水平与"给定 - 对照"配对差值。
    for arm_name in ("all_contacts", "random_restraints", "null_structural"):
        key = f"{arm_name}_held_out_contact"
        arm_values = [row["metrics"][key] for row in valid if key in row.get("metrics", {})]
        if arm_values:
            arm_comparisons[arm_name] = {
                "mean": float(np.mean(arm_values)),
                # seed+1：与主指标使用不同随机流，避免相关性泄漏。
                "ci95": list(bootstrap_mean_ci(arm_values, seed + 1, trials)),
            }
        paired_rows = [row for row in valid if key in row.get("metrics", {})]
        paired_values = [
            row["metrics"]["mean_held_out_contact_recovery"] - row["metrics"][key]
            for row in paired_rows]
        if not paired_values:
            continue
        if cluster_by_id:
            # 按官方抗原簇聚合：先在簇内取均值，再以簇为推断单位做 bootstrap。
            grouped = {}
            for row, value in zip(paired_rows, paired_values, strict=True):
                grouped.setdefault(cluster_by_id[row["id"]], []).append(value)
            inference_values = [float(np.mean(values)) for values in grouped.values()]
            inference_unit = "official_antigen_cluster"
        else:
            inference_values = paired_values
            inference_unit = "structure"
        paired_contrasts[f"supplied_minus_{arm_name}"] = {
            "mean": float(np.mean(inference_values)),
            # seed+2：配对差值使用独立随机流。
            "ci95": list(bootstrap_mean_ci(inference_values, seed + 2, trials)),
            "n_inference_units": len(inference_values),
            "inference_unit": inference_unit,
        }

    return {
        "n_records": len(results),
        "n_torsion_hits": len(hits),
        "n_valid_structures": len(valid),
        "overall_valid_structure_fraction": overall_valid,
        "post_tier_qc_fraction": post_tier_valid,
        "mean_held_out_contact_recovery": mean_contact,
        "held_out_contact_recovery_ci95": contact_ci,
        "fraction_structures_positive_held_out_recovery": positive_fraction,
        "mean_rmsd_recovery_angstrom": mean_rmsd,
        "rmsd_recovery_ci95": rmsd_ci,
        "arm_comparisons": arm_comparisons,
        "paired_arm_contrasts": paired_contrasts,
        "gate_pass": passed,
    }
