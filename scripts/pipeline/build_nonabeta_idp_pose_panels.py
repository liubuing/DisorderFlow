#!/usr/bin/env python
"""Build strict template-transferred pose panels for tau and alpha-synuclein."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "modules"))

from ensemble_pose_transfer import (  # noqa: E402
    fixed_paratope_contact_map, pose_geometry_audit, refine_pose_contacts,
    refine_pose_local, resolve_pose_clashes, transfer_pose,
)
from state_contact_scorer import extract_contact_map  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/non_abeta_idp_ensemble_prep_v1/ensemble_manifest.json")
    parser.add_argument("--pose-output", default="data/non_abeta_idp/pose_panels_v1")
    parser.add_argument("--out", default="outputs/non_abeta_idp_pose_panels_v1")
    parser.add_argument("--max-severe-clashes", type=int, default=0)
    parser.add_argument("--min-contacts", type=int, default=8)
    parser.add_argument("--max-refinement-translation", type=float, default=6.0)
    parser.add_argument("--max-refinement-rotation", type=float, default=30.0)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(PROJECT_ROOT / args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    rows = []
    for target in manifest["targets"]:
        template = PROJECT_ROOT / target["trimmed_reference_pdb"]
        template_map = extract_contact_map(
            str(template), peptide_chain=target["reference_peptide_chain"]
        )
        if template_map["peptide_sequence"] != target["epitope"]:
            raise ValueError(f"{target['target']}: trimmed reference sequence mismatch")
        for conformer in target["selected_conformers"]:
            input_path = PROJECT_ROOT / conformer["pdb"]
            output = (
                PROJECT_ROOT / args.pose_output / target["target"]
                / f"conformer{conformer['conformer']}_pose.pdb"
            )
            transfer = transfer_pose(
                template, input_path, output, target["reference_peptide_chain"], "A",
                target["heavy_chain"], target["light_chain"], 1,
            )
            retreat = resolve_pose_clashes(
                output, [target["heavy_chain"], target["light_chain"]],
                max_severe_clashes=args.max_severe_clashes,
            )
            initial = pose_geometry_audit(
                output, [target["heavy_chain"], target["light_chain"]]
            )
            initial_contact_map = fixed_paratope_contact_map(output, template_map)
            refinement = {
                "local_refinement_applied": False,
                "local_refinement_translation_norm": 0.0,
                "local_refinement_rotation_norm_degrees": 0.0,
                "local_refinement_attempts": 0,
            }
            if (
                initial["severe_clash_pairs_lt_1_5A"] > args.max_severe_clashes
                or len(initial_contact_map["contacts"]) < args.min_contacts
            ):
                seed = sum(ord(char) for char in f"{target['target']}:{conformer['conformer']}")
                if target["target"] == "tau":
                    refined = refine_pose_contacts(
                        output, [target["heavy_chain"], target["light_chain"]],
                        template_map, seed=seed,
                    )
                else:
                    refined = refine_pose_local(
                        output, [target["heavy_chain"], target["light_chain"]], seed=seed,
                    )
                refinement = {"local_refinement_applied": True, **refined}
            post_refinement_retreat = {
                "post_refinement_retreat_angstrom": 0.0,
                "post_refinement_retreat_direction": [0.0, 0.0, 0.0],
            }
            post_geometry = pose_geometry_audit(
                output, [target["heavy_chain"], target["light_chain"]]
            )
            if post_geometry["severe_clash_pairs_lt_1_5A"] > args.max_severe_clashes:
                post_retreat = resolve_pose_clashes(
                    output, [target["heavy_chain"], target["light_chain"]],
                    max_severe_clashes=args.max_severe_clashes,
                )
                post_refinement_retreat = {
                    "post_refinement_retreat_angstrom": post_retreat["rigid_retreat_angstrom"],
                    "post_refinement_retreat_direction": post_retreat["retreat_direction"],
                }
            contact_map = fixed_paratope_contact_map(output, template_map)
            geometry = pose_geometry_audit(
                output, [target["heavy_chain"], target["light_chain"]]
            )
            checks = {
                "exact_epitope": contact_map["peptide_sequence"] == target["epitope"],
                "zero_severe_clashes": geometry["severe_clash_pairs_lt_1_5A"] == 0,
                "interface_contacts": len(contact_map["contacts"]) >= args.min_contacts,
                "bounded_translation": (
                    refinement["local_refinement_translation_norm"]
                    <= args.max_refinement_translation
                ),
                "bounded_rotation": (
                    refinement["local_refinement_rotation_norm_degrees"]
                    <= args.max_refinement_rotation
                ),
            }
            rows.append({
                "target": target["target"],
                "reference_id": target["reference_id"],
                "conformer": conformer["conformer"],
                "source_model": conformer["source_model"],
                "pose_pdb": output.relative_to(PROJECT_ROOT).as_posix(),
                "pose_provenance": "experimental_antibody_template_plus_experimental_nmr_epitope_local_refinement",
                "experimental_pose": False,
                "epitope_fit_rmsd": transfer["epitope_fit_rmsd"],
                **retreat,
                **refinement,
                **post_refinement_retreat,
                "n_contacts": len(contact_map["contacts"]),
                **geometry,
                "checks": checks,
                "status": "pass" if all(checks.values()) else "fail",
            })
    by_target = {}
    for target in manifest["targets"]:
        target_rows = [row for row in rows if row["target"] == target["target"]]
        passing = [row for row in target_rows if row["status"] == "pass"]
        # Ensemble preparation orders conformers by farthest-point diversity,
        # so the first five strict passing rows preserve that ordering.
        selected_ids = {id(row) for row in passing[:5]}
        for row in target_rows:
            row["selected_for_panel"] = id(row) in selected_ids
        by_target[target["target"]] = {
            "passed": len(passing),
            "total_audited": len(target_rows),
            "selected": min(5, len(passing)),
            "failed_models_retained": len(target_rows) - len(passing),
        }
    report = {
        "schema_version": "nonabeta.idp_pose_panel.v1",
        "status": "pass" if all(row["selected"] == 5 for row in by_target.values()) else "fail",
        "thresholds": {
            "max_severe_clashes": args.max_severe_clashes,
            "min_contacts": args.min_contacts,
            "max_refinement_translation": args.max_refinement_translation,
            "max_refinement_rotation": args.max_refinement_rotation,
        },
        "by_target": by_target,
        "entries": rows,
        "claim_boundary": "Template-transferred computational poses, not experimental binding states.",
    }
    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pose_panel_audit.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Non-A-beta pose panels: status={report['status']} {by_target}")
    if report["status"] != "pass":
        raise SystemExit("Strict non-A-beta pose gate failed")


if __name__ == "__main__":
    main()
