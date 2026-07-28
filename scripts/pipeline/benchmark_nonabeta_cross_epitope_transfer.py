#!/usr/bin/env python
"""Leave-one-epitope-region-out transfer of mutation rules across antibody families."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "modules"))
sys.path.insert(0, str(ROOT / "scripts/pipeline"))

from build_contact_guided_mutation_plans import infer_chain_roles, region_label  # noqa: E402
from state_contact_scorer import (  # noqa: E402
    _hotspot_positions, _position_allowed_residues, extract_contact_map,
    score_sequence_on_contact_map,
)


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def effects(family):
    contact_map = extract_contact_map(
        str(ROOT / family["pdb_path"]), peptide_chain=family["antigen_chain"]
    )
    native = contact_map["paratope_sequence"]
    native_score = score_sequence_on_contact_map(native, contact_map)["state_contact_score"]
    roles = infer_chain_roles(contact_map)
    hotspots = _hotspot_positions(contact_map)
    rows = []
    for index, residue in enumerate(contact_map["paratope_residues"]):
        role = roles.get(residue["chain"].upper(), residue["chain"].upper())
        region = region_label(role, residue["resid"])
        if region == "framework_or_unknown":
            continue
        for mutant in _position_allowed_residues(contact_map, index, hotspots):
            if mutant == native[index]:
                continue
            sequence = native[:index] + mutant + native[index + 1:]
            score = score_sequence_on_contact_map(sequence, contact_map)["state_contact_score"]
            rows.append({
                "target": family["target"], "family": family["family"], "pdb_id": family["pdb_id"],
                "epitope": family["epitope"], "region": f"{role}_{region}",
                "native_aa": native[index], "mutant_aa": mutant,
                "mutation_key": f"{role}_{region}:{native[index]}>{mutant}",
                "effect": score - native_score,
            })
    return rows


def main():
    families = [row for row in load_csv(ROOT / "outputs/non_abeta_idp_family_panel_v1/family_panel.csv") if row["family_structure_status"] == "usable"]
    all_effects = []
    failed = []
    for family in families:
        try:
            all_effects.extend(effects(family))
        except Exception as error:
            failed.append({"family": family["family"], "error": str(error)})
    folds = []
    predictions = []
    for heldout in families:
        test = [row for row in all_effects if row["family"] == heldout["family"]]
        train = [row for row in all_effects if row["target"] == heldout["target"] and row["family"] != heldout["family"]]
        learned = defaultdict(list)
        for row in train:
            learned[row["mutation_key"]].append(row["effect"])
        covered = []
        for row in test:
            if row["mutation_key"] not in learned:
                continue
            prediction = float(np.mean(learned[row["mutation_key"]]))
            record = {**row, "predicted_effect": prediction, "actual_effect": row["effect"]}
            predictions.append(record); covered.append(record)
        if len(covered) >= 3:
            correlation = float(spearmanr(
                [row["predicted_effect"] for row in covered],
                [row["actual_effect"] for row in covered],
            ).statistic)
            if not np.isfinite(correlation):
                correlation = None
            direction = float(np.mean([
                (row["predicted_effect"] >= 0) == (row["actual_effect"] >= 0) for row in covered
            ]))
        else:
            correlation = None; direction = None
        folds.append({
            "target": heldout["target"], "heldout_family": heldout["family"],
            "heldout_epitope": heldout["epitope"], "train_families": len({row["family"] for row in train}),
            "test_mutations": len(test), "covered_mutations": len(covered),
            "coverage": len(covered) / max(len(test), 1),
            "spearman": correlation, "direction_accuracy": direction,
            "fold_status": "scoreable" if len(covered) >= 3 else "insufficient_rule_overlap",
        })
    out = ROOT / "outputs/non_abeta_idp_cross_epitope_transfer_v1"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("single_mutation_effects.csv", all_effects), ("cross_epitope_predictions.csv", predictions), ("cross_epitope_folds.csv", folds)):
        if rows:
            with open(out / name, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    scoreable = [row for row in folds if row["fold_status"] == "scoreable"]
    summary = {
        "schema_version": "nonabeta.cross_epitope_transfer.v1",
        "status": "complete" if len(scoreable) == len(families) else "partial",
        "benchmark_name": "leave-one-epitope-region-out mutation-rule transfer",
        "families_total": len(families), "scoreable_folds": len(scoreable),
        "mean_coverage": float(np.mean([row["coverage"] for row in folds])) if folds else None,
        "mean_direction_accuracy": float(np.mean([row["direction_accuracy"] for row in scoreable])) if scoreable else None,
        "failed_families": failed,
        "same_epitope_lofo": {
            "family_data_ready": True,
            "strict_LOFO_complete": False,
            "status": "blocked_insufficient_same_epitope_independent_families",
        },
        "claim_boundary": (
            "This is cross-epitope mutation-rule transfer, not same-epitope LOFO. Family and epitope are confounded."
        ),
    }
    json.dump(summary, open(out / "cross_epitope_summary.json", "w", encoding="utf-8"), indent=2)
    print(f"Cross-epitope benchmark: {summary}")


if __name__ == "__main__":
    main()
