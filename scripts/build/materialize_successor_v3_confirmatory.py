#!/usr/bin/env python
"""Materialize and contact-filter successor-v3 confirmatory candidates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from Bio.PDB import Chain, MMCIFParser, Model, PDBIO, Structure


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.build.materialize_multiscaffold_candidates import (  # noqa: E402
    chain_sequence,
    download_crop,
    number_variable_domains,
)


CONTACT_CUTOFF = 4.5


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolved_residues(chain):
    return [residue for residue in chain.get_residues()
            if residue.id[0] == " " and "CA" in residue]


def contact_metrics(heavy_residues, antigen_residues, h3_indices):
    contacting_positions = 0
    residue_contacts = 0
    minimum_distance = float("inf")
    for index in h3_indices:
        if index < 0 or index >= len(heavy_residues):
            raise ValueError(f"H3 coordinate index {index} is outside resolved heavy chain")
        heavy_coords = np.asarray([
            atom.coord for atom in heavy_residues[index].get_atoms()
            if atom.element != "H"
        ])
        position_contacts = 0
        for antigen_residue in antigen_residues:
            antigen_coords = np.asarray([
                atom.coord for atom in antigen_residue.get_atoms()
                if atom.element != "H"
            ])
            if not len(heavy_coords) or not len(antigen_coords):
                continue
            distance = float(np.linalg.norm(
                heavy_coords[:, None, :] - antigen_coords[None, :, :], axis=-1
            ).min())
            minimum_distance = min(minimum_distance, distance)
            if distance <= CONTACT_CUTOFF:
                position_contacts += 1
        contacting_positions += int(position_contacts > 0)
        residue_contacts += position_contacts
    return {
        "n_contacting_h3_positions": contacting_positions,
        "n_h3_antigen_residue_contacts": residue_contacts,
        "minimum_h3_antigen_distance": (
            round(minimum_distance, 4) if np.isfinite(minimum_distance) else None),
    }


def write_canonical_pdb(path, source_chains):
    structure = Structure.Structure(path.stem)
    model = Model.Model(0)
    structure.add(model)
    for canonical_id, source_chain in source_chains:
        chain = Chain.Chain(canonical_id)
        model.add(chain)
        for residue in source_chain.get_residues():
            chain.add(copy.deepcopy(residue))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(path))


def write_json_once(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="ascii", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    discovery_path = ROOT / args.discovery
    output_dir = ROOT / args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite structure materialization: {output_dir}")
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    if discovery.get("classification") != "metadata_only":
        raise ValueError("Discovery artifact is not metadata-only")

    cif_dir = output_dir / "cif"
    pdb_dir = output_dir / "pdb"
    cif_dir.mkdir(parents=True, exist_ok=False)
    pdb_dir.mkdir(parents=True, exist_ok=False)
    parser_cif = MMCIFParser(QUIET=True, auth_chains=True, auth_residues=True)
    annotated, exclusions = [], []
    try:
        for index, row in enumerate(discovery["candidates"], 1):
            cif_path = cif_dir / f"{row['instance']}.cif"
            try:
                download_crop(row, cif_path)
                structure = parser_cif.get_structure(row["instance"], str(cif_path))
                model = next(structure.get_models())
                if row["heavy_chain"] not in model or row["light_chain"] not in model:
                    raise ValueError("missing paired heavy/light chain in downloaded crop")
                heavy = chain_sequence(model[row["heavy_chain"]])
                light = chain_sequence(model[row["light_chain"]])
                antigen_options = []
                for chain_id in row["antigen_chains"]:
                    if chain_id not in model:
                        continue
                    residues = chain_sequence(model[chain_id])
                    if 5 <= len(residues) <= 50:
                        antigen_options.append((chain_id, residues))
                if len(heavy) < 90 or len(light) < 80 or len(antigen_options) != 1:
                    raise ValueError(
                        "requires paired variable chains and exactly one resolved "
                        f"5-50 aa peptide; H={len(heavy)}, L={len(light)}, "
                        f"peptides={len(antigen_options)}")
                antigen_chain, antigen = antigen_options[0]
                annotated.append({
                    **row,
                    "source_cif": cif_path.relative_to(ROOT).as_posix(),
                    "source_cif_sha256": sha256(cif_path),
                    "heavy_sequence": "".join(item["aa"] for item in heavy),
                    "light_sequence": "".join(item["aa"] for item in light),
                    "antigen_chain": antigen_chain,
                    "antigen_sequence": "".join(item["aa"] for item in antigen),
                })
            except Exception as error:  # noqa: BLE001
                exclusions.append({"instance": row["instance"], "reason": str(error)})
            print(f"Downloaded and parsed {index}/{len(discovery['candidates'])}", flush=True)

        domains_by_id = number_variable_domains(annotated) if annotated else {}
        eligible = []
        for row in annotated:
            try:
                domains = domains_by_id.get(row["instance"], {})
                heavy_domain, light_domain = domains.get("H", {}), domains.get("L", {})
                heavy_cdrs = heavy_domain.get("cdrs", {})
                light_cdrs = light_domain.get("cdrs", {})
                h3 = heavy_cdrs.get("H3", "")
                if heavy_domain.get("chain_type") != "H":
                    raise ValueError("ANARCII heavy-chain role mismatch")
                if light_domain.get("chain_type") not in {"K", "L"}:
                    raise ValueError("ANARCII light-chain role mismatch")
                if not 4 <= len(h3) <= 30:
                    raise ValueError(f"H3 length outside frozen 4-30 range: {len(h3)}")

                cif_path = ROOT / row["source_cif"]
                structure = parser_cif.get_structure(row["instance"], str(cif_path))
                model = next(structure.get_models())
                heavy_residues = resolved_residues(model[row["heavy_chain"]])
                antigen_residues = resolved_residues(model[row["antigen_chain"]])
                metrics = contact_metrics(
                    heavy_residues, antigen_residues, heavy_domain["h3_indices"])
                if metrics["n_contacting_h3_positions"] < 1:
                    raise ValueError("no H3-peptide heavy-atom contact within 4.5 A")

                pdb_path = pdb_dir / f"{row['instance']}.pdb"
                write_canonical_pdb(pdb_path, [
                    ("H", model[row["heavy_chain"]]),
                    ("L", model[row["light_chain"]]),
                    ("P", model[row["antigen_chain"]]),
                ])
                cdrs = {**heavy_cdrs, **light_cdrs}
                eligible.append({
                    **row,
                    "original_chain_ids": {
                        "heavy": row["heavy_chain"], "light": row["light_chain"],
                        "antigen": row["antigen_chain"],
                    },
                    "heavy_chain": "H",
                    "light_chain": "L",
                    "antigen_chain": "P",
                    "pdb_path": pdb_path.relative_to(ROOT).as_posix(),
                    "pdb_sha256": sha256(pdb_path),
                    "vh_sequence": heavy_domain["sequence"],
                    "vl_sequence": light_domain["sequence"],
                    "cdr_sequences": cdrs,
                    "paired_cdr_sequence": "".join(
                        cdrs[name] for name in ("H1", "H2", "H3", "L1", "L2", "L3")),
                    "cdr_h3_sequence": h3,
                    "h3_heavy_indices_zero_based": heavy_domain["h3_indices"],
                    **metrics,
                })
            except Exception as error:  # noqa: BLE001
                exclusions.append({"instance": row["instance"], "reason": str(error)})

        payload = {
            "schema_version": 1,
            "status": "structure_annotation_complete; isolation_pending",
            "classification": "model_free_structural_eligibility",
            "claim_boundary": "structure and sequence eligibility only; no model inference",
            "source_discovery": args.discovery.as_posix(),
            "source_discovery_sha256": sha256(discovery_path),
            "numbering": {
                "tool": "ANARCII",
                "version": importlib.metadata.version("anarcii"),
                "scheme": "chothia",
            },
            "contact_cutoff_angstrom": CONTACT_CUTOFF,
            "counts": {
                "discovered": len(discovery["candidates"]),
                "structurally_eligible": len(eligible),
                "excluded": len(exclusions),
            },
            "records": sorted(eligible, key=lambda item: item["instance"]),
            "exclusions": sorted(exclusions, key=lambda item: item["instance"]),
        }
        write_json_once(output_dir / "structural_manifest.json", payload)
        print(json.dumps(payload["counts"], indent=2))
    except Exception:
        shutil.rmtree(output_dir)
        raise


if __name__ == "__main__":
    main()
