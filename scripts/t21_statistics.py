"""Statistical aggregation for the T2.1 recovery benchmark."""

import numpy as np


def bootstrap_mean_ci(values, seed, trials=10000):
    rng = np.random.default_rng(int(seed))
    array = np.asarray(values)
    means = [float(np.mean(array[rng.integers(0, len(array), size=len(array))]))
             for _ in range(int(trials))]
    means.sort()
    return float(means[int(0.025 * len(means))]), float(means[int(0.975 * len(means))])


def aggregate(results, config, cluster_by_id=None):
    valid = [row for row in results if row.get("valid_t2_1")]
    hits = [row for row in results if row.get("torsion_hit_tier")]
    if not valid:
        return {"n_records": len(results), "n_torsion_hits": len(hits),
                "n_valid_structures": 0, "gate_pass": False}

    statistics = config["statistics"]
    trials = int(statistics["bootstrap_trials"])
    seed = int(statistics["bootstrap_seed"])

    def interval(key):
        values = [row["metrics"][key] for row in valid if key in row.get("metrics", {})]
        if len(values) < 2:
            return float(np.mean(values)) if values else None, [None, None]
        return float(np.mean(values)), list(bootstrap_mean_ci(values, seed, trials))

    mean_contact, contact_ci = interval("mean_held_out_contact_recovery")
    mean_rmsd, rmsd_ci = interval("mean_rmsd_recovery_angstrom")
    overall_valid = len(valid) / max(1, len(results))
    post_tier_valid = len(valid) / max(1, len(hits))
    contact_values = [
        row["metrics"].get("mean_held_out_contact_recovery", 0) for row in valid]
    positive_fraction = float(np.mean(np.asarray(contact_values) > 0))

    gates = config["development_gates"]
    passed = bool(
        overall_valid >= float(gates["minimum_valid_structure_fraction"])
        and contact_ci[0] is not None and contact_ci[0] > 0
        and positive_fraction >= float(
            gates["minimum_fraction_structures_positive_held_out_recovery"]))

    arm_comparisons = {}
    paired_contrasts = {}
    for arm_name in ("all_contacts", "random_restraints", "null_structural"):
        key = f"{arm_name}_held_out_contact"
        arm_values = [row["metrics"][key] for row in valid if key in row.get("metrics", {})]
        if arm_values:
            arm_comparisons[arm_name] = {
                "mean": float(np.mean(arm_values)),
                "ci95": list(bootstrap_mean_ci(arm_values, seed + 1, trials)),
            }
        paired_rows = [row for row in valid if key in row.get("metrics", {})]
        paired_values = [
            row["metrics"]["mean_held_out_contact_recovery"] - row["metrics"][key]
            for row in paired_rows]
        if not paired_values:
            continue
        if cluster_by_id:
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
