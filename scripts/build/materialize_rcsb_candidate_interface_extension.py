#!/usr/bin/env python
"""Materialize and contact-filter RCSB calibration-extension candidates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBIO
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.build.discover_rcsb_candidate_interface_extension import (  # noqa: E402
    classify_entity,
    description_of,
    length_of,
)
from scripts.build.materialize_multiscaffold_candidates import (  # noqa: E402
    AA,
    CHOTHIA_CDR_RANGES,
)
from scripts.build.materialize_successor_v3_confirmatory import (  # noqa: E402
    CONTACT_CUTOFF,
    contact_metrics,
)

DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
USER_AGENT = "DisorderFlow-candidate-interface-extension/1"
HEAVY_LIGHT_PAIRING_CUTOFF = 8.0
MAX_COMBINATIONS = 4096
MIN_PEPTIDE_LENGTH = 5
MAX_PEPTIDE_LENGTH = 50
MIN_HEAVY_LENGTH = 90
MIN_LIGHT_LENGTH = 80
H3_MIN_LENGTH = 4
H3_MAX_LENGTH = 30


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Materialization timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def download_entry(pdb_id: str, path: Path,
                   opener=urllib.request.urlopen, retries: int = 3) -> None:
    request = urllib.request.Request(
        DOWNLOAD_URL.format(pdb_id=pdb_id),
        headers={"User-Agent": USER_AGENT},
    )
    for attempt in range(retries):
        try:
            with opener(request, timeout=120) as response:
                payload = response.read()
            if not payload.startswith(b"data_"):
                raise RuntimeError("RCSB response is not an mmCIF file")
            path.write_bytes(payload)
            return
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)


def strict_residues(chain):
    return [
        residue for residue in chain.get_residues()
        if residue.id[0] == " " and "CA" in residue
        and seq1(residue.resname, custom_map={"MSE": "M"}) in AA
    ]


def heavy_atom_coords(residues) -> np.ndarray:
    coords = [
        atom.coord for residue in residues for atom in residue.get_atoms()
        if atom.element != "H"
    ]
    return np.asarray(coords, dtype=float).reshape(-1, 3)


def min_distance(left: np.ndarray, right: np.ndarray, block: int = 256) -> float:
    if not len(left) or not len(right):
        return float("inf")
    minimum = float("inf")
    for start in range(0, len(left), block):
        chunk = left[start:start + block]
        minimum = min(minimum, float(np.linalg.norm(
            chunk[:, None, :] - right[None, :, :], axis=-1).min()))
    return minimum


def write_canonical_pdb(path: Path, source_chains) -> None:
    """Write renamed chains without deep-copying parent structure references."""
    import copy as _copy

    structure = Structure(path.stem)
    model = Model(0)
    structure.add(model)
    memo: dict[int, object] = {}
    for canonical_id, source_chain in source_chains:
        chain = Chain(canonical_id)
        memo[id(source_chain)] = chain
        model.add(chain)
        for residue in source_chain.get_residues():
            chain.add(_copy.deepcopy(residue, memo))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(path))


def role_chains(entry: dict[str, object], title: str) -> tuple[list[str], list[str], list[str]]:
    heavy: set[str] = set()
    light: set[str] = set()
    peptide: set[str] = set()
    for entity in entry.get("polymer_entities") or []:
        role = classify_entity(entity, title)
        chains = (entity.get("rcsb_polymer_entity_container_identifiers") or {}).get(
            "auth_asym_ids") or []
        if role == "heavy":
            heavy.update(str(chain) for chain in chains)
        elif role == "light":
            light.update(str(chain) for chain in chains)
        elif role == "unclassified":
            length = length_of(entity)
            if length is not None and MIN_PEPTIDE_LENGTH <= length <= MAX_PEPTIDE_LENGTH:
                peptide.update(str(chain) for chain in chains)
    return sorted(heavy), sorted(light), sorted(peptide)


def number_sequences(sequences: dict[str, str]) -> dict[str, dict]:
    from anarcii import Anarcii

    model = Anarcii(seq_type="antibody", mode="accuracy", batch_size=32,
                    cpu=True, ncpu=1, verbose=False)
    numbered = model.number(sequences)
    numbered = model.to_scheme("chothia")
    domains = {}
    for identifier, result in numbered.items():
        if not result or result.get("error"):
            continue
        role = identifier.rsplit("|", 1)[1]
        query_index = int(result["query_start"])
        domain_sequence = []
        cdrs = {name: [] for name in CHOTHIA_CDR_RANGES[role]}
        indices = []
        for (position, _insertion), aa in result["numbering"]:
            if aa == "-":
                continue
            domain_sequence.append(aa)
            for cdr_name, (start, end) in CHOTHIA_CDR_RANGES[role].items():
                if start <= int(position) <= end:
                    cdrs[cdr_name].append(aa)
            if role == "H" and 93 <= int(position) <= 102:
                indices.append(query_index)
            query_index += 1
        domains[identifier] = {
            "sequence": "".join(domain_sequence),
            "chain_type": result["chain_type"],
            "cdrs": {name: "".join(sequence) for name, sequence in cdrs.items()},
            "h3_indices": indices,
        }
    return domains


def combo_rank(metrics: dict) -> tuple:
    h3_distance = metrics.get("minimum_h3_antigen_distance")
    return (
        -int(metrics["n_contacting_h3_positions"]),
        -int(metrics["n_h3_antigen_residue_contacts"]),
        float("inf") if h3_distance is None else float(h3_distance),
        float(metrics["minimum_antibody_peptide_distance"]),
        (metrics["heavy_chain_id"], metrics["light_chain_id"], metrics["peptide_chain_id"]),
    )


def verify_snapshot(snapshot_dir: Path, discovery: dict[str, object]) -> None:
    manifest_path = snapshot_dir / "acquisition.json"
    if sha256_file(manifest_path) != discovery["snapshot_manifest_sha256"]:
        raise ValueError("Snapshot manifest hash no longer matches discovery artifact")
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    for name, recorded in manifest["files"].items():
        if sha256_file(snapshot_dir / name) != recorded["sha256"]:
            raise ValueError(f"Snapshot file hash mismatch: {name}")


def materialize(discovery_path: Path, output_dir: Path,
                opener=urllib.request.urlopen,
                now=lambda: datetime.now(timezone.utc)) -> dict[str, object]:
    discovery_path = Path(discovery_path).resolve()
    output_dir = Path(output_dir).resolve()
    if (output_dir / "structural_manifest.json").exists():
        raise FileExistsError(
            f"Refusing to overwrite structure materialization manifest: {output_dir}")

    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    if discovery.get("classification") != "metadata_only":
        raise ValueError("Discovery artifact is not metadata-only")
    snapshot_dir = ROOT / discovery["snapshot_dir"]
    verify_snapshot(snapshot_dir, discovery)
    snapshot_entries = json.loads(
        (snapshot_dir / "entries.json").read_text(encoding="ascii"))["entries"]
    entries_by_pdb = {str(entry["rcsb_id"]).upper(): entry for entry in snapshot_entries}
    candidates = [entry for entry in discovery["entries"] if entry.get("status") == "candidate"]

    cif_dir = output_dir / "cif"
    pdb_dir = output_dir / "pdb"
    parser_cif = MMCIFParser(QUIET=True, auth_chains=True, auth_residues=True)
    annotated: list[dict[str, object]] = []
    exclusions: list[dict[str, str]] = []
    checkpoint_path = None
    try:
        cif_dir.mkdir(parents=True, exist_ok=True)
        pdb_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = output_dir / "geometric_checkpoint.json"
        resumed = False
        if checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="ascii"))
            if (checkpoint.get("source_discovery_sha256")
                    == sha256_file(discovery_path)):
                annotated = checkpoint["annotated"]
                exclusions = checkpoint["exclusions"]
                resumed = True
                print(f"Resumed geometric checkpoint: {len(annotated)} records, "
                      f"{len(exclusions)} exclusions", flush=True)
        if not resumed:
            for index, candidate in enumerate(candidates, 1):
                pdb_id = str(candidate["pdb_id"]).upper()
                try:
                    entry = entries_by_pdb[pdb_id]
                    title = str((entry.get("struct") or {}).get("title") or "")
                    heavy_ids, light_ids, peptide_ids = role_chains(entry, title)

                    cif_path = cif_dir / f"{pdb_id}.cif"
                    if not cif_path.exists():
                        download_entry(pdb_id, cif_path, opener=opener)
                    structure = parser_cif.get_structure(pdb_id, str(cif_path))
                    model = next(structure.get_models())
                    del structure

                    heavy_ids = [chain for chain in heavy_ids if chain in model]
                    light_ids = [chain for chain in light_ids if chain in model]
                    peptide_ids = [chain for chain in peptide_ids if chain in model]
                    if not heavy_ids:
                        raise ValueError("no resolved heavy chain in deposited model")
                    if not light_ids:
                        raise ValueError(
                            "no resolved light chain; nanobody-only entries are outside "
                            "the paired heavy/light schema")
                    if not peptide_ids:
                        raise ValueError("no resolved peptide antigen chain in deposited model")

                    combinations = [
                        (heavy, light, peptide)
                        for heavy in heavy_ids
                        for light in light_ids
                        for peptide in peptide_ids
                    ]
                    if len(combinations) > MAX_COMBINATIONS:
                        raise ValueError(
                            f"chain combination count {len(combinations)} exceeds "
                            f"{MAX_COMBINATIONS}")

                    residues_by_chain = {}
                    coords_by_chain = {}
                    for chain_id in sorted(set(heavy_ids) | set(light_ids) | set(peptide_ids)):
                        residues = strict_residues(model[chain_id])
                        residues_by_chain[chain_id] = residues
                        coords_by_chain[chain_id] = heavy_atom_coords(residues)

                    valid_combos = []
                    failure_counts = {
                        "heavy_too_short": 0,
                        "light_too_short": 0,
                        "peptide_resolved_length_out_of_range": 0,
                        "heavy_light_pairing_above_cutoff": 0,
                        "no_antibody_peptide_contact": 0,
                    }
                    antibody_peptide_distances = {}
                    heavy_light_distances = {}
                    for heavy_id, light_id, peptide_id in combinations:
                        heavy = residues_by_chain[heavy_id]
                        light = residues_by_chain[light_id]
                        peptide = residues_by_chain[peptide_id]
                        if len(heavy) < MIN_HEAVY_LENGTH:
                            failure_counts["heavy_too_short"] += 1
                            continue
                        if len(light) < MIN_LIGHT_LENGTH:
                            failure_counts["light_too_short"] += 1
                            continue
                        if not MIN_PEPTIDE_LENGTH <= len(peptide) <= MAX_PEPTIDE_LENGTH:
                            failure_counts["peptide_resolved_length_out_of_range"] += 1
                            continue
                        for antibody_id in (heavy_id, light_id):
                            if (antibody_id, peptide_id) not in antibody_peptide_distances:
                                antibody_peptide_distances[(antibody_id, peptide_id)] = min_distance(
                                    coords_by_chain[antibody_id], coords_by_chain[peptide_id])
                        antibody_peptide = min(
                            antibody_peptide_distances[(heavy_id, peptide_id)],
                            antibody_peptide_distances[(light_id, peptide_id)],
                        )
                        if antibody_peptide > CONTACT_CUTOFF:
                            failure_counts["no_antibody_peptide_contact"] += 1
                            continue
                        if (heavy_id, light_id) not in heavy_light_distances:
                            heavy_light_distances[(heavy_id, light_id)] = min_distance(
                                coords_by_chain[heavy_id], coords_by_chain[light_id])
                        heavy_light = heavy_light_distances[(heavy_id, light_id)]
                        if heavy_light > HEAVY_LIGHT_PAIRING_CUTOFF:
                            failure_counts["heavy_light_pairing_above_cutoff"] += 1
                            continue
                        valid_combos.append({
                            "heavy_chain_id": heavy_id,
                            "light_chain_id": light_id,
                            "peptide_chain_id": peptide_id,
                            "heavy_sequence": "".join(
                                seq1(residue.resname, custom_map={"MSE": "M"}) for residue in heavy),
                            "light_sequence": "".join(
                                seq1(residue.resname, custom_map={"MSE": "M"}) for residue in light),
                            "peptide_sequence": "".join(
                                seq1(residue.resname, custom_map={"MSE": "M"}) for residue in peptide),
                            "minimum_heavy_light_distance": round(heavy_light, 4),
                            "minimum_antibody_peptide_distance": round(antibody_peptide, 4),
                        })
                    if not valid_combos:
                        raise ValueError(
                            "no geometrically valid heavy/light/peptide combination "
                            f"({len(combinations)} tried; {json.dumps(failure_counts)})")
                    annotated.append({
                        "pdb_id": pdb_id,
                        "title": title,
                        "release_date": candidate.get("release_date"),
                        "resolution": candidate.get("resolution"),
                        "methods": candidate.get("methods"),
                        "viral_antigen": bool(candidate.get("viral_antigen")),
                        "source_cif": cif_path.relative_to(ROOT).as_posix(),
                        "source_cif_sha256": sha256_file(cif_path),
                        "combos": valid_combos,
                    })
                except Exception as error:  # noqa: BLE001
                    exclusions.append({"pdb_id": pdb_id, "reason": str(error)})
                print(f"Materialized {index}/{len(candidates)}: {pdb_id}", flush=True)

            with checkpoint_path.open("w", encoding="ascii", newline="\n") as handle:
                json.dump({
                    "source_discovery_sha256": sha256_file(discovery_path),
                    "annotated": annotated,
                    "exclusions": exclusions,
                }, handle)

        sequences: dict[str, str] = {}
        for record in annotated:
            for combo in record["combos"]:
                for role, field in (("H", "heavy_sequence"), ("L", "light_sequence")):
                    key = f"{combo[field]}|{role}"
                    if combo[field] and key not in sequences:
                        sequences[key] = combo[field]

        cache_path = output_dir / "numbering_cache.json"
        numbering_cache: dict[str, dict] = {}
        if cache_path.exists():
            cached_payload = json.loads(cache_path.read_text(encoding="ascii"))
            if cached_payload.get("anarcii_version") == importlib.metadata.version("anarcii"):
                numbering_cache = cached_payload.get("domains", {})
        pending = {key: sequence for key, sequence in sequences.items()
                   if key not in numbering_cache}
        if pending:
            numbering_cache.update(number_sequences(pending))
            with cache_path.open("w", encoding="ascii", newline="\n") as handle:
                json.dump({
                    "anarcii_version": importlib.metadata.version("anarcii"),
                    "domains": numbering_cache,
                }, handle)
        domains = {key: numbering_cache.get(key) for key in sequences}
        cached_count = sum(1 for key in sequences if key in numbering_cache)

        eligible = []
        for record_index, record in enumerate(annotated, 1):
            print(f"Selecting {record_index}/{len(annotated)}: {record['pdb_id']}", flush=True)
            selection = None
            reasons = []
            structure = parser_cif.get_structure(
                record["pdb_id"], str(ROOT / record["source_cif"]))
            model = next(structure.get_models())
            residues_by_chain = {}
            for combo in record["combos"]:
                for chain_id in (combo["heavy_chain_id"], combo["peptide_chain_id"]):
                    if chain_id not in residues_by_chain:
                        residues_by_chain[chain_id] = strict_residues(model[chain_id])
                heavy_domain = domains.get(f"{combo['heavy_sequence']}|H")
                light_domain = domains.get(f"{combo['light_sequence']}|L")
                if not heavy_domain:
                    reasons.append(f"ANARCII failure for heavy chain {combo['heavy_chain_id']}")
                    continue
                if not light_domain:
                    reasons.append(f"ANARCII failure for light chain {combo['light_chain_id']}")
                    continue
                if heavy_domain.get("chain_type") != "H":
                    reasons.append(f"heavy chain role mismatch for {combo['heavy_chain_id']}")
                    continue
                if light_domain.get("chain_type") not in {"K", "L"}:
                    reasons.append(f"light chain role mismatch for {combo['light_chain_id']}")
                    continue
                h3 = heavy_domain["cdrs"]["H3"]
                if not H3_MIN_LENGTH <= len(h3) <= H3_MAX_LENGTH:
                    reasons.append(
                        f"H3 length {len(h3)} outside frozen {H3_MIN_LENGTH}-"
                        f"{H3_MAX_LENGTH} range for {combo['heavy_chain_id']}")
                    continue

                metrics = contact_metrics(
                    residues_by_chain[combo["heavy_chain_id"]],
                    residues_by_chain[combo["peptide_chain_id"]],
                    heavy_domain["h3_indices"])
                if metrics["n_contacting_h3_positions"] < 1:
                    reasons.append(
                        f"no H3-peptide heavy-atom contact within {CONTACT_CUTOFF} A for "
                        f"{combo['heavy_chain_id']}/{combo['peptide_chain_id']}")
                    continue

                merged = {**combo, **metrics}
                if selection is None or combo_rank(merged) < combo_rank(selection):
                    selection = merged
            if selection is None:
                exclusions.append({
                    "pdb_id": record["pdb_id"],
                    "reason": "; ".join(sorted(set(reasons))) or "no valid numbered combination",
                })
                continue

            heavy_domain = domains[f"{selection['heavy_sequence']}|H"]
            light_domain = domains[f"{selection['light_sequence']}|L"]
            instance = (f"{record['pdb_id']}_{selection['heavy_chain_id']}_"
                        f"{selection['light_chain_id']}_{selection['peptide_chain_id']}")
            pdb_path = pdb_dir / f"{instance}.pdb"
            write_canonical_pdb(pdb_path, [
                ("H", model[selection["heavy_chain_id"]]),
                ("L", model[selection["light_chain_id"]]),
                ("P", model[selection["peptide_chain_id"]]),
            ])
            cdrs = {**heavy_domain["cdrs"], **light_domain["cdrs"]}
            eligible.append({
                "instance": instance,
                "pdb_id": record["pdb_id"],
                "title": record["title"],
                "release_date": record["release_date"],
                "methods": record["methods"],
                "viral_antigen": bool(record.get("viral_antigen")),
                "resolution": record["resolution"],
                "source_cif": record["source_cif"],
                "source_cif_sha256": record["source_cif_sha256"],
                "original_chain_ids": {
                    "heavy": selection["heavy_chain_id"],
                    "light": selection["light_chain_id"],
                    "antigen": selection["peptide_chain_id"],
                },
                "heavy_chain": "H",
                "light_chain": "L",
                "antigen_chain": "P",
                "pdb_path": pdb_path.relative_to(ROOT).as_posix(),
                "pdb_sha256": sha256_file(pdb_path),
                "heavy_sequence": selection["heavy_sequence"],
                "light_sequence": selection["light_sequence"],
                "antigen_sequence": selection["peptide_sequence"],
                "vh_sequence": heavy_domain["sequence"],
                "vl_sequence": light_domain["sequence"],
                "cdr_sequences": cdrs,
                "paired_cdr_sequence": "".join(
                    cdrs[name] for name in ("H1", "H2", "H3", "L1", "L2", "L3")),
                "cdr_h3_sequence": heavy_domain["cdrs"]["H3"],
                "h3_heavy_indices_zero_based": heavy_domain["h3_indices"],
                "h3_definition": "Chothia positions 93-102, bounded by Cys92 and Trp103",
                "n_contacting_h3_positions": selection["n_contacting_h3_positions"],
                "n_h3_antigen_residue_contacts": selection["n_h3_antigen_residue_contacts"],
                "minimum_h3_antigen_distance": selection["minimum_h3_antigen_distance"],
                "minimum_heavy_light_distance": selection["minimum_heavy_light_distance"],
                "minimum_antibody_peptide_distance": selection[
                    "minimum_antibody_peptide_distance"],
            })

        payload = {
            "schema_version": "candidate_interface_extension_structural_manifest_v1",
            "status": "structure_annotation_complete; isolation_pending",
            "classification": "model_free_structural_eligibility",
            "claim_boundary": (
                "structure and sequence eligibility only; no homogeneity claim, "
                "no model inference, no confidence target access"),
            "materialized_at_utc": iso_utc(now()),
            "source_discovery": discovery_path.relative_to(ROOT).as_posix()
            if discovery_path.is_relative_to(ROOT) else discovery_path.as_posix(),
            "source_discovery_sha256": sha256_file(discovery_path),
            "snapshot_dir": discovery["snapshot_dir"],
            "numbering": {
                "tool": "ANARCII",
                "version": importlib.metadata.version("anarcii"),
                "scheme": "chothia",
                "unique_sequences": len(sequences),
                "cached_sequences": cached_count,
            },
            "contact_cutoff_angstrom": CONTACT_CUTOFF,
            "heavy_light_pairing_cutoff_angstrom": HEAVY_LIGHT_PAIRING_CUTOFF,
            "eligibility": {
                "min_heavy_residues": MIN_HEAVY_LENGTH,
                "min_light_residues": MIN_LIGHT_LENGTH,
                "peptide_length_range": [MIN_PEPTIDE_LENGTH, MAX_PEPTIDE_LENGTH],
                "h3_length_range": [H3_MIN_LENGTH, H3_MAX_LENGTH],
                "max_chain_combinations": MAX_COMBINATIONS,
            },
            "counts": {
                "discovered_candidates": len(candidates),
                "structurally_eligible": len(eligible),
                "excluded": len(exclusions),
            },
            "records": sorted(eligible, key=lambda item: item["instance"]),
            "exclusions": sorted(exclusions, key=lambda item: item["pdb_id"]),
        }
        with (output_dir / "structural_manifest.json").open("x", encoding="ascii",
                                                            newline="\n") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(json.dumps(payload["counts"], indent=2))
        return payload
    except Exception:
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.discovery, args.output_dir)


if __name__ == "__main__":
    main()
