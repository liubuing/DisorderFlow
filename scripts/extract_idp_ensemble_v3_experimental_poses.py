#!/usr/bin/env python
"""Extract write-once experimental pose panels for the admitted v3 cohort.

Pose sources follow the frozen protocol priority:
1. independent experimental structure, same lineage and target;
2. independent complex copy within the same biological assembly.
Synthetic sampling is NOT performed here.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path

from Bio.PDB import PDBIO, MMCIFParser, Select
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parents[1]

MIN_HEAVY_OVERLAP = 100
MIN_SEQUENCE_RATIO = 0.98

OWN_OVERRIDE = {
    # VHH entries with two assembly copies: one pose = one antibody copy +
    # its antigen copy. The admission lists both copies for contact curation.
    "9YZ9": {"antibody": ["B"], "antigen": ["A"]},
}

EXTRA_PLAN = {
    "9XHY": [
        {"entry": "9XI1", "antibody": ["H", "L"], "antigen": ["E"],
         "source": "independent_experimental_structure_same_lineage_and_target"},
        {"entry": "9XHZ", "antibody": ["G", "J"], "antigen": ["A"],
         "source": "independent_experimental_structure_same_lineage_and_target"},
        {"entry": "9XI0", "antibody": ["G", "J"], "antigen": ["A"],
         "source": "independent_experimental_structure_same_lineage_and_target"},
    ],
    "9XI4": [
        {"entry": "9XI5", "antibody": ["G", "J"], "antigen": ["A"],
         "source": "independent_experimental_structure_same_lineage_and_target"},
    ],
    "9W89": [
        {"entry": "9W7J", "antibody": ["AUTO"], "antigen": ["C"],
         "source": "independent_experimental_structure_same_lineage_and_target"},
    ],
    "9YZ9": [
        {"entry": "9YZ9", "antibody": ["D"], "antigen": ["C"],
         "source": "independent_complex_copy_same_assembly"},
    ],
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chain_sequence(chain):
    return "".join(
        seq1(residue.resname)
        for residue in chain
        if "CA" in residue and is_aa(residue, standard=True)
    )


class RoleSelect(Select):
    def __init__(self, keep):
        self.keep = set(keep)

    def accept_chain(self, chain):
        return chain.id in self.keep


def sequence_ratio(reference, candidate):
    if not reference or not candidate:
        return 0.0
    return difflib.SequenceMatcher(None, reference, candidate).ratio()


def resolve_auto_chain(model, reference_sequence, candidates):
    scored = [
        (sequence_ratio(reference_sequence, chain_sequence(model[chain_id])), chain_id)
        for chain_id in candidates
    ]
    scored.sort(reverse=True)
    return scored[0]


def extract_pose(parser, cif_path, antibody_chains, antigen_chains, pose_path):
    structure = parser.get_structure(cif_path.stem, str(cif_path))
    model = next(iter(structure))
    io = PDBIO()
    io.set_structure(model)
    io.save(str(pose_path), RoleSelect(antibody_chains + antigen_chains))
    return pose_path


def extract(admission_path, coordinate_dir, output_dir):
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite pose panel: {output_dir}")
    admission = json.loads(admission_path.read_text(encoding="ascii"))
    parser = MMCIFParser(QUIET=True)
    component_rows = []
    output_dir.mkdir(parents=True)
    try:
        for component in admission["components"]:
            component_id = component["component_id"]
            override = OWN_OVERRIDE.get(component_id)
            own_chains = override["antibody"] if override else component["antibody_chains"]
            own_antigen = override["antigen"] if override else component["antigen_chains"]
            own_cif = coordinate_dir / f"{component_id}.cif"
            own_structure = parser.get_structure(component_id, str(own_cif))
            own_model = next(iter(own_structure))
            reference_heavy = chain_sequence(own_model[own_chains[0]])
            reference_light = (
                chain_sequence(own_model[own_chains[1]])
                if len(own_chains) > 1 else None
            )
            poses = [
                {
                    "entry": component_id,
                    "antibody": list(own_chains),
                    "antigen": list(own_antigen),
                    "source": "own_entry",
                }
            ] + EXTRA_PLAN.get(component_id, [])
            rows = []
            for index, pose in enumerate(poses):
                entry = pose["entry"]
                cif_path = coordinate_dir / f"{entry}.cif"
                structure = parser.get_structure(entry, str(cif_path))
                model = next(iter(structure))
                antibody = pose["antibody"]
                note = None
                if antibody[0] == "AUTO":
                    ratio, chain_id = resolve_auto_chain(
                        model, reference_heavy, ["B", "D"]
                    )
                    antibody = [chain_id]
                    note = {
                        "auto_resolved_chain": chain_id,
                        "auto_candidates": ["B", "D"],
                    }
                heavy = chain_sequence(model[antibody[0]])
                heavy_ratio = sequence_ratio(reference_heavy, heavy)
                heavy_ok = (
                    len(heavy) >= MIN_HEAVY_OVERLAP
                    and heavy_ratio >= MIN_SEQUENCE_RATIO
                )
                light_ratio = None
                light_ok = True
                if reference_light is not None:
                    if len(antibody) > 1:
                        light = chain_sequence(model[antibody[1]])
                        light_ratio = sequence_ratio(reference_light, light)
                        light_ok = light_ratio >= MIN_SEQUENCE_RATIO
                    else:
                        light_ok = False
                pose_dir = output_dir / component_id
                pose_dir.mkdir(exist_ok=True)
                pose_path = pose_dir / f"pose_{index}_{entry}.pdb"
                extract_pose(
                    parser, cif_path, antibody, pose["antigen"], pose_path
                )
                rows.append({
                    "pose_id": f"pose_{index}_{entry}",
                    "entry_id": entry,
                    "source": pose["source"],
                    "source_priority": (
                        1 if pose["source"] == "own_entry"
                        else 2 if pose["source"] == "independent_complex_copy_same_assembly"
                        else 1
                    ),
                    "path": pose_path.resolve().relative_to(ROOT).as_posix(),
                    "coordinate_sha256": sha256(pose_path),
                    "antibody_chains": antibody,
                    "antigen_chains": pose["antigen"],
                    "heavy_sequence_identity": round(heavy_ratio, 6),
                    "light_sequence_identity": (
                        round(light_ratio, 6) if light_ratio is not None else None
                    ),
                    "sequence_verified": bool(heavy_ok and light_ok),
                    "accepted": bool(heavy_ok and light_ok),
                    "auto_resolution": note,
                })
            for row in rows:
                if row["accepted"]:
                    duplicates = [
                        other for other in rows
                        if other["accepted"]
                        and other["coordinate_sha256"] == row["coordinate_sha256"]
                    ]
                    if len(duplicates) > 1:
                        for duplicate in duplicates[1:]:
                            duplicate["accepted"] = False
            component_rows.append({
                "component_id": component_id,
                "poses": rows,
                "accepted_pose_count": sum(row["accepted"] for row in rows),
            })
        manifest = {
            "schema_version": 1,
            "status": "experimental_pose_panel_extracted",
            "classification": "v3_experimental_pose_panel",
            "supersedes": (
                "experimental_pose_panel_2026_08_15: 9YZ9 assembly copies were "
                "mis-assigned as heavy/light in the v1 own-entry pose"
            ),
            "admission": str(admission_path),
            "admission_sha256": sha256(admission_path),
            "component_count": len(component_rows),
            "components": component_rows,
            "claim_boundary": (
                "experimental pose extraction for a general antibody "
                "multi-conformer panel; no candidate or performance result"
            ),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="ascii"
        )
        return manifest
    except Exception:
        for path in sorted(output_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        output_dir.rmdir()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = extract(args.admission, args.coordinates, args.output_dir)
    summary = {
        row["component_id"]: row["accepted_pose_count"] for row in result["components"]
    }
    print(json.dumps({
        "status": result["status"],
        "accepted_pose_counts": summary,
        "multi_pose_components": sum(count >= 2 for count in summary.values()),
    }, indent=2))


if __name__ == "__main__":
    main()
