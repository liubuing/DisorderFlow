#!/usr/bin/env python
"""Conservative PDB adapters for real StateContrast conformational ensembles.

This module exposes structure metadata and coordinates. It deliberately does
not infer an antibody pose or derive antigen-antibody contacts.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path


class StructureValidationError(ValueError):
    """Raised when a PDB cannot provide valid, selected protein geometry."""


_AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}


def _selected(value: object, available: Sequence[object], label: str) -> list[object]:
    if value is None or value == "":
        return list(available)
    requested = list(value) if isinstance(value, (list, tuple, set)) else [value]
    if label == "model":
        try:
            requested = [int(item) for item in requested]
        except (TypeError, ValueError) as error:
            raise StructureValidationError("model selection must contain integers") from error
    missing = [item for item in requested if item not in available]
    if missing:
        raise StructureValidationError(
            f"selected {label}(s) {missing!r} not present; available: {list(available)!r}"
        )
    return requested


def _sequence_audit(residues: list[dict], seqres: str, expected_sequence: str) -> dict:
    observed = "".join(residue["one_letter_code"] for residue in residues)
    numbers = [residue["residue_number"] for residue in residues
               if not residue["insertion_code"]]
    gaps = []
    for left, right in zip(numbers, numbers[1:], strict=False):
        if right > left + 1:
            gaps.append([left + 1, right - 1])

    expected = expected_sequence.strip().upper()
    audit = {
        "observed_sequence": observed,
        "observed_residue_count": len(residues),
        "residue_number_gaps": gaps,
        "seqres_sequence": seqres,
        "seqres_residue_count": len(seqres),
        "seqres_coverage": len(residues) / len(seqres) if seqres else None,
        "expected_sequence": expected,
        "expected_residue_count": len(expected),
        "expected_coverage": min(len(observed), len(expected)) / len(expected) if expected else None,
        "expected_sequence_exact": observed == expected if expected else None,
    }
    return audit


def parse_pdb_ensemble(
    pdb_path: str | Path,
    model: int | Sequence[int] | None = None,
    chain: str | Sequence[str] | None = None,
    expected_sequence: str = "",
) -> dict:
    """Parse protein coordinates from all PDB MODELs or an explicit selection.

    MODEL identifiers are the serials written in the PDB file. A file without
    MODEL records is represented as model 1. Alternate locations are retained
    as separate atom records and identified by ``altloc``.
    """
    path = Path(pdb_path)
    if not path.is_file():
        raise StructureValidationError(f"PDB does not exist: {path}")
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as error:
        raise StructureValidationError(f"PDB is not readable text: {path}") from error

    models: dict[int, dict[str, dict[tuple, dict]]] = {}
    seqres_parts: dict[str, list[str]] = {}
    current_model = 1
    explicit_models = False
    in_model = False
    for line_number, line in enumerate(lines, 1):
        record = line[:6].strip().upper()
        if record == "MODEL":
            if in_model:
                raise StructureValidationError(f"nested MODEL record at line {line_number}")
            explicit_models = True
            in_model = True
            try:
                current_model = int(line[10:14].strip())
            except ValueError as error:
                raise StructureValidationError(
                    f"invalid MODEL serial at line {line_number}"
                ) from error
            if current_model in models:
                raise StructureValidationError(f"duplicate MODEL serial {current_model}")
            models[current_model] = {}
            continue
        if record == "ENDMDL":
            if not in_model:
                raise StructureValidationError(f"ENDMDL without MODEL at line {line_number}")
            in_model = False
            continue
        if record == "SEQRES":
            chain_id = (line[11:12].strip() or "_")
            seqres_parts.setdefault(chain_id, []).extend(line[19:].split())
            continue
        if record not in {"ATOM", "HETATM"}:
            continue
        if len(line) < 54:
            raise StructureValidationError(f"truncated coordinate record at line {line_number}")
        residue_name = line[17:20].strip().upper()
        if residue_name not in _AA3:
            continue
        if explicit_models and not in_model:
            raise StructureValidationError(
                f"coordinate outside a MODEL block at line {line_number}"
            )
        try:
            atom = {
                "serial": int(line[6:11]),
                "name": line[12:16].strip(),
                "altloc": line[16:17].strip(),
                "x": float(line[30:38]),
                "y": float(line[38:46]),
                "z": float(line[46:54]),
                "occupancy": float(line[54:60]) if line[54:60].strip() else None,
                "bfactor": float(line[60:66]) if line[60:66].strip() else None,
                "element": line[76:78].strip() if len(line) >= 78 else "",
            }
            residue_number = int(line[22:26])
        except ValueError as error:
            raise StructureValidationError(
                f"invalid numeric coordinate field at line {line_number}"
            ) from error
        if not all(-1e6 < atom[axis] < 1e6 for axis in ("x", "y", "z")):
            raise StructureValidationError(f"non-finite or implausible coordinate at line {line_number}")
        chain_id = line[21:22].strip() or "_"
        residue_key = (residue_number, line[26:27].strip(), residue_name)
        chain_residues = models.setdefault(current_model, {}).setdefault(chain_id, {})
        residue = chain_residues.setdefault(residue_key, {
            "residue_name": residue_name,
            "one_letter_code": _AA3[residue_name],
            "residue_number": residue_number,
            "insertion_code": line[26:27].strip(),
            "atoms": [],
        })
        residue["atoms"].append(atom)

    if in_model:
        raise StructureValidationError(f"MODEL {current_model} has no closing ENDMDL")

    available_models = sorted(identifier for identifier, chains in models.items() if chains)
    if not available_models:
        raise StructureValidationError("PDB contains no recognized protein coordinates")
    selected_models = _selected(model, available_models, "model")
    available_chains = sorted({chain_id for identifier in selected_models
                               for chain_id in models[identifier]})
    selected_chains = _selected(chain, available_chains, "chain")

    conformers = []
    for model_id in selected_models:
        chain_rows = []
        for chain_id in selected_chains:
            if chain_id not in models[model_id]:
                raise StructureValidationError(
                    f"chain {chain_id!r} is absent from selected model {model_id}"
                )
            residues = list(models[model_id][chain_id].values())
            seqres = "".join(_AA3[name] for name in seqres_parts.get(chain_id, [])
                             if name in _AA3)
            chain_rows.append({
                "chain_id": chain_id,
                "sequence_audit": _sequence_audit(residues, seqres, expected_sequence),
                "residues": residues,
                "atom_count": sum(len(residue["atoms"]) for residue in residues),
            })
        conformers.append({
            "model_id": model_id,
            "chains": chain_rows,
            "atom_count": sum(row["atom_count"] for row in chain_rows),
        })

    return {
        "schema_version": "statecontrast.pdb_geometry.v1",
        "source_pdb": str(path),
        "selected_models": selected_models,
        "selected_chains": selected_chains,
        "conformer_count": len(conformers),
        "conformers": conformers,
        "geometry_only": True,
        "contacts": None,
    }


def _as_chain_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return []


def audit_ensemble_entries(entries: Iterable[dict], project_root: str | Path = ".") -> list[dict]:
    """Load ensemble geometry and report whether an explicit antibody pose exists."""
    root = Path(project_root)
    rows = []
    for entry in entries:
        pdb_value = entry.get("pdb", "")
        pdb_path = root / pdb_value if pdb_value else root / "__missing_pdb_path__"
        row = {
            "name": entry.get("name", ""),
            "pdb": str(pdb_path),
            "chain": entry.get("chain", ""),
            "state_type": entry.get("state_type", ""),
            "exists": bool(pdb_value) and pdb_path.is_file(),
            "geometry": None,
            "geometry_status": "invalid_structure",
            "contact_scoring_status": "blocked_without_pose",
            "has_antibody_pose": False,
        }
        if not row["exists"]:
            row["status"] = "missing_pdb"
            row["error"] = "PDB path is missing or does not exist"
            rows.append(row)
            continue
        antibody_chains = _as_chain_list(
            entry.get("antibody_chains", entry.get("antibody_chain"))
        )
        requested_chains = _as_chain_list(entry.get("chain")) + antibody_chains
        requested_chains = list(dict.fromkeys(requested_chains)) or None
        try:
            geometry = parse_pdb_ensemble(
                pdb_path,
                model=entry.get("model"),
                chain=requested_chains,
                expected_sequence=str(entry.get("expected_sequence", "")),
            )
        except StructureValidationError as error:
            row["status"] = "invalid_structure"
            row["error"] = str(error)
            rows.append(row)
            continue

        present_by_model = [
            {chain_row["chain_id"] for chain_row in conformer["chains"]}
            for conformer in geometry["conformers"]
        ]
        has_pose = bool(antibody_chains) and all(
            set(antibody_chains).issubset(present) for present in present_by_model
        )
        row.update({
            "geometry": geometry,
            "geometry_status": "geometry_ready",
            "has_antibody_pose": has_pose,
            "contact_scoring_status": "pose_available" if has_pose else "blocked_without_pose",
            "status": "geometry_ready" if has_pose
                      else "geometry_ready_but_contact_scoring_blocked_without_pose",
        })
        rows.append(row)
    return rows


def real_ensemble_mode_status(config: dict, project_root: str | Path = ".") -> dict:
    negative = config.get("negative_states", {})
    if negative.get("mode") != "structure_ensemble":
        return {"mode": negative.get("mode", ""), "status": "not_structure_ensemble"}
    rows = audit_ensemble_entries(negative.get("ensembles", []), project_root)
    ready = [row for row in rows if row["geometry_status"] == "geometry_ready"]
    invalid = [row for row in rows if row["status"] == "invalid_structure"]
    missing = [row for row in rows if row["status"] == "missing_pdb"]
    pose_blocked = [row for row in ready if not row["has_antibody_pose"]]
    if not rows or invalid or missing:
        status = "blocked_missing_or_invalid_ensemble_pdbs"
        geometry_status = "blocked"
    elif pose_blocked:
        status = "geometry_ready_but_contact_scoring_blocked_without_pose"
        geometry_status = "geometry_ready"
    else:
        status = "geometry_ready"
        geometry_status = "geometry_ready"
    return {
        "mode": "structure_ensemble",
        "n_ensembles": len(rows),
        "ready": len(ready),
        "missing": len(missing),
        "invalid": len(invalid),
        "geometry_status": geometry_status,
        "contact_scoring_status": (
            "blocked_without_pose" if pose_blocked else
            "pose_available" if ready and not invalid and not missing else "blocked"
        ),
        "status": status,
        "entries": rows,
    }
