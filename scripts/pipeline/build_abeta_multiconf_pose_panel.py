#!/usr/bin/env python
"""Build and audit template-transferred antibody poses on A-beta42 conformers."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "modules"))

from ensemble_pose_transfer import fixed_paratope_contact_map, pose_geometry_audit, refine_pose_local, resolve_pose_clashes, transfer_pose  # noqa: E402
from state_contact_scorer import extract_contact_map  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/statecontrast/abeta_position_effect_v3_4ref_loro.yml")
    parser.add_argument("--conformers", default="data/abeta_conformations/pdbs")
    parser.add_argument("--pose-output", default="data/abeta_pose_panels/template_transfer_v1")
    parser.add_argument("--out", default="outputs/abeta_multiconf_pose_panel_v1")
    parser.add_argument("--max-epitope-deformation", type=float, default=4.5)
    parser.add_argument("--max-severe-clashes", type=int, default=0)
    parser.add_argument("--min-contacts", type=int, default=8)
    parser.add_argument("--min-pass-fraction", type=float, default=0.90)
    parser.add_argument("--min-passes-per-reference", type=int, default=4)
    parser.add_argument("--max-refinement-translation", type=float, default=6.0)
    parser.add_argument("--max-refinement-rotation", type=float, default=30.0)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(PROJECT_ROOT / args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    conformers = sorted((PROJECT_ROOT / args.conformers).glob("abeta42_seed*_42.pdb"))
    if len(conformers) < 5:
        raise SystemExit(f"Expected at least five A-beta42 conformers, found {len(conformers)}")

    rows = []
    for reference in config["references"]:
        template = PROJECT_ROOT / reference["complex_pdb"]
        template_map = extract_contact_map(str(template), peptide_chain=reference["peptide_chain"])
        for conformer in conformers:
            output = PROJECT_ROOT / args.pose_output / reference["pdb"] / f"{conformer.stem}_pose.pdb"
            transfer = transfer_pose(
                template, conformer, output, reference["peptide_chain"], "P",
                reference["heavy_chain"], reference["light_chain"], reference["target_region"][0],
            )
            retreat = resolve_pose_clashes(
                output, [reference["heavy_chain"], reference["light_chain"]],
                max_severe_clashes=args.max_severe_clashes,
            )
            initial_geometry = pose_geometry_audit(
                output, [reference["heavy_chain"], reference["light_chain"]]
            )
            refinement = {
                "local_refinement_applied": False,
                "local_refinement_translation_norm": 0.0,
                "local_refinement_rotation_norm_degrees": 0.0,
            }
            if initial_geometry["severe_clash_pairs_lt_1_5A"] > args.max_severe_clashes:
                refinement = {
                    "local_refinement_applied": True,
                    **refine_pose_local(
                        output, [reference["heavy_chain"], reference["light_chain"]],
                        seed=sum(ord(char) for char in f"{reference['pdb']}:{conformer.stem}"),
                    ),
                }
            contact_map = fixed_paratope_contact_map(output, template_map)
            geometry = pose_geometry_audit(output, [reference["heavy_chain"], reference["light_chain"]])
            checks = {
                "epitope_deformation": transfer["epitope_fit_rmsd"] <= args.max_epitope_deformation,
                "severe_clashes": geometry["severe_clash_pairs_lt_1_5A"] <= args.max_severe_clashes,
                "interface_contacts": len(contact_map["contacts"]) >= args.min_contacts,
                "fixed_paratope": contact_map["paratope_sequence"] == template_map["paratope_sequence"],
                "bounded_refinement_translation": (
                    refinement["local_refinement_translation_norm"] <= args.max_refinement_translation
                ),
                "bounded_refinement_rotation": (
                    refinement["local_refinement_rotation_norm_degrees"] <= args.max_refinement_rotation
                ),
            }
            rows.append({
                "reference_pdb": reference["pdb"],
                "antibody": reference.get("antibody", ""),
                "conformer": conformer.stem,
                "pose_pdb": (
                    output.relative_to(PROJECT_ROOT).as_posix()
                    if output.is_relative_to(PROJECT_ROOT)
                    else str(output.resolve())
                ),
                "pose_provenance": "crystal_antibody_plus_kabsch_transferred_abeta42_conformer",
                "experimental_pose": False,
                "epitope_fit_rmsd": round(transfer["epitope_fit_rmsd"], 6),
                **retreat,
                **refinement,
                "n_contacts": len(contact_map["contacts"]),
                "n_paratope_positions": len(contact_map["paratope_residues"]),
                **geometry,
                "checks": checks,
                "status": "pass" if all(checks.values()) else "fail",
            })

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    passed = sum(row["status"] == "pass" for row in rows)
    passes_by_reference = {
        reference["pdb"]: sum(
            row["status"] == "pass" for row in rows
            if row["reference_pdb"] == reference["pdb"]
        )
        for reference in config["references"]
    }
    panel_pass = (
        passed / len(rows) >= args.min_pass_fraction
        and all(count >= args.min_passes_per_reference for count in passes_by_reference.values())
    )
    report = {
        "schema_version": "statecontrast.template_pose_panel.v1",
        "status": "pass" if panel_pass else "fail",
        "pose_semantics": "computational template transfer; not an experimental negative or binding measurement",
        "references": len(config["references"]),
        "conformers": len(conformers),
        "poses": len(rows),
        "passed": passed,
        "pass_fraction": passed / len(rows),
        "passes_by_reference": passes_by_reference,
        "thresholds": {
            "max_epitope_deformation": args.max_epitope_deformation,
            "max_severe_clashes": args.max_severe_clashes,
            "min_contacts": args.min_contacts,
            "min_pass_fraction": args.min_pass_fraction,
            "min_passes_per_reference": args.min_passes_per_reference,
            "max_refinement_translation": args.max_refinement_translation,
            "max_refinement_rotation": args.max_refinement_rotation,
        },
        "entries": rows,
    }
    with open(out_dir / "pose_panel_audit.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with open(out_dir / "pose_panel_report.md", "w", encoding="utf-8") as handle:
        handle.write("# A-beta Multi-Conformation Pose Panel\n\n")
        handle.write(f"Status: `{report['status']}`; poses passing: {passed}/{len(rows)}.\n\n")
        handle.write("These are Kabsch template-transferred computational poses, not experimental binding states.\n\n")
        handle.write("| Reference | Conformer | Epitope deformation | Retreat | Local translation | Local rotation | Contacts | Min distance | Severe clashes | Status |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            handle.write(
                f"| {row['reference_pdb']} | {row['conformer']} | {row['epitope_fit_rmsd']} | "
                f"{row['rigid_retreat_angstrom']:.1f} | {row['local_refinement_translation_norm']:.1f} | "
                f"{row['local_refinement_rotation_norm_degrees']:.1f} | {row['n_contacts']} | "
                f"{row['minimum_heavy_atom_distance']:.3f} | "
                f"{row['severe_clash_pairs_lt_1_5A']} | {row['status']} |\n"
            )
    print(f"Pose panel: {passed}/{len(rows)} passed; report={out_dir}")
    if report["status"] != "pass":
        raise SystemExit("Pose-panel gate failed")


if __name__ == "__main__":
    main()
