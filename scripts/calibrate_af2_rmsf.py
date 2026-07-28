#!/usr/bin/env python3
"""Calibrate AF2-seed RMSF against independent solution-NMR ensembles."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import pickle
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import lmdb
import numpy as np
from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB import PDBParser

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from disorderflow.utils.conformation_calibration import (  # noqa: E402
    calibration_metrics,
    calibration_verdict,
    per_residue_rmsf,
    sequence_index_map,
)

UNIPROT_PATTERN = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-\d+)?$"
)
RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_ENTITY = "https://data.rcsb.org/rest/v1/core/polymer_entity/{entry}/{entity}"
RCSB_PDB = "https://files.rcsb.org/download/{entry}.pdb"


def read_lmdb(path):
    env = lmdb.open(path, readonly=True, lock=False, readahead=False, subdir=os.path.isdir(path))
    entries = []
    with env.begin() as txn:
        raw_length = txn.get(b"__len__")
        if raw_length:
            keys = [f"{index:08d}".encode() for index in range(pickle.loads(raw_length))]
        else:
            keys = [key for key, _ in txn.cursor() if not key.startswith(b"__")]
        for source_index, key in enumerate(keys):
            value = txn.get(key)
            if value is not None:
                entry = pickle.loads(value)
                entry["_source_index"] = source_index
                entries.append(entry)
    env.close()
    return entries


def _request_json(request, retries=5):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except (urllib.error.URLError, http.client.RemoteDisconnected, TimeoutError):
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)


def post_json(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "DisorderFlow/0.1"},
    )
    return _request_json(request)


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "DisorderFlow/0.1"})
    return _request_json(request)


def discover_nmr_entities(accessions, batch_size=80):
    """Return accession -> RCSB polymer entity identifiers."""
    discovered = {accession: [] for accession in accessions}
    entity_ids = set()
    for start in range(0, len(accessions), batch_size):
        batch = accessions[start:start + batch_size]
        accession_nodes = [
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": (
                        "rcsb_polymer_entity_container_identifiers."
                        "reference_sequence_identifiers.database_accession"
                    ),
                    "operator": "exact_match",
                    "value": accession,
                },
            }
            for accession in batch
        ]
        payload = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {"type": "group", "logical_operator": "or", "nodes": accession_nodes},
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "exptl.method",
                            "operator": "exact_match",
                            "value": "SOLUTION NMR",
                        },
                    },
                ],
            },
            "request_options": {"return_all_hits": True},
            "return_type": "polymer_entity",
        }
        try:
            response = post_json(RCSB_SEARCH, payload)
        except urllib.error.HTTPError as error:
            if error.code == 204:
                continue
            raise
        entity_ids.update(item["identifier"] for item in response.get("result_set", []))

    for index, identifier in enumerate(sorted(entity_ids), start=1):
        entry, entity = identifier.split("_", 1)
        try:
            metadata = get_json(RCSB_ENTITY.format(entry=entry, entity=entity))
        except Exception as error:
            print(f"Skipping RCSB entity {identifier}: {error}", file=sys.stderr)
            continue
        references = metadata.get("rcsb_polymer_entity_container_identifiers", {}).get(
            "reference_sequence_identifiers", []
        ) or []
        for reference in references:
            accession = str(reference.get("database_accession", "")).split("-")[0]
            if accession in discovered:
                discovered[accession].append(identifier)
        if index % 50 == 0:
            time.sleep(0.5)
    return {key: sorted(set(value)) for key, value in discovered.items() if value}


def download_pdb(entry_id, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"{entry_id}.pdb"
    if not destination.is_file():
        request = urllib.request.Request(
            RCSB_PDB.format(entry=entry_id), headers={"User-Agent": "DisorderFlow/0.1"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            destination.write_bytes(response.read())
    return destination


def residue_letter(residue):
    return protein_letters_3to1.get(residue.get_resname().upper(), "X")


def extract_best_nmr_trace(pdb_path, query_sequence, minimum_residues=20):
    structure = PDBParser(QUIET=True).get_structure(pdb_path.stem, str(pdb_path))
    models = list(structure.get_models())
    if len(models) < 2:
        return None
    best = None
    for chain in models[0].get_chains():
        residues = [residue for residue in chain if residue.id[0] == " " and "CA" in residue]
        sequence = "".join(residue_letter(residue) for residue in residues)
        mapping = sequence_index_map(query_sequence, sequence)
        if len(mapping) < minimum_residues:
            continue
        identity = len(mapping) / max(1, min(len(query_sequence), len(sequence)))
        candidate = (len(mapping), identity, chain.id, residues, sequence, mapping)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        return None
    _, identity, chain_id, residues, chain_sequence, mapping = best
    residue_ids = [residue.id for residue in residues]
    common = []
    for chain_index, residue_id in enumerate(residue_ids):
        if all(
            chain_id in model
            and residue_id in model[chain_id]
            and "CA" in model[chain_id][residue_id]
            for model in models
        ):
            common.append(chain_index)
    reverse_mapping = {chain_index: query_index for query_index, chain_index in mapping.items()}
    common = [chain_index for chain_index in common if chain_index in reverse_mapping]
    if len(common) < minimum_residues:
        return None
    coordinates = np.asarray(
        [
            [model[chain_id][residue_ids[index]]["CA"].coord for index in common]
            for model in models
        ],
        dtype=np.float64,
    )
    return {
        "chain_id": chain_id,
        "n_models": len(models),
        "identity": identity,
        "query_indices": [reverse_mapping[index] for index in common],
        "experimental_rmsf": per_residue_rmsf(coordinates),
        "chain_sequence": chain_sequence,
    }


def confidence_by_sequence(path):
    if not path or not Path(path).exists():
        return {}
    lookup = {}
    for entry in read_lmdb(path):
        plddt = entry.get("af2_plddt")
        if plddt is not None:
            if hasattr(plddt, "detach"):
                plddt = plddt.detach().cpu().numpy()
            lookup[str(entry.get("sequence", ""))] = {
                "plddt": np.asarray(plddt, dtype=np.float64),
                "disorder_fraction": entry.get("disorder_fraction"),
                "disorder_source": entry.get("disorder_source"),
            }
    return lookup


def make_report(entries, entity_map, cache_dir, confidence_lookup, minimum_residues):
    by_accession = {}
    for entry in entries:
        accession = str(entry.get("pdb_id", "")).split("-")[0]
        if UNIPROT_PATTERN.fullmatch(accession):
            by_accession.setdefault(accession, []).append(entry)
    results = []
    failures = []
    for accession, entity_ids in entity_map.items():
        for entry in by_accession.get(accession, []):
            sequence = str(entry["sequence"])
            for identifier in entity_ids:
                pdb_id = identifier.split("_", 1)[0]
                try:
                    pdb_path = download_pdb(pdb_id, cache_dir)
                    trace = extract_best_nmr_trace(pdb_path, sequence, minimum_residues)
                    if trace is None:
                        failures.append({"accession": accession, "pdb_id": pdb_id, "reason": "no_matching_trace"})
                        continue
                    indices = np.asarray(trace.pop("query_indices"), dtype=int)
                    predicted = np.asarray(entry["rmsf"], dtype=np.float64)[indices]
                    experimental = np.asarray(
                        trace.pop("experimental_rmsf"), dtype=np.float64
                    )
                    confidence = confidence_lookup.get(sequence, {})
                    plddt = confidence.get("plddt")
                    uncertainty = None
                    if plddt is not None and len(plddt) == len(sequence):
                        scale = 100.0 if np.nanmax(plddt) > 1.5 else 1.0
                        uncertainty = 1.0 - np.asarray(plddt)[indices] / scale
                    metrics = calibration_metrics(
                        predicted, experimental, uncertainty
                    )
                    results.append({
                        "accession": accession,
                        "pdb_id": pdb_id,
                        "source_index": entry["_source_index"],
                        "sequence_length": len(sequence),
                        "n_matched_residues": len(indices),
                        "predicted_rmsf_mean": float(predicted.mean()),
                        "experimental_rmsf_mean": float(experimental.mean()),
                        "query_indices": indices.tolist(),
                        "predicted_rmsf": predicted.tolist(),
                        "experimental_rmsf": experimental.tolist(),
                        "disorder_fraction": confidence.get("disorder_fraction"),
                        "disorder_source": confidence.get("disorder_source"),
                        **trace,
                        **metrics,
                    })
                    break
                except Exception as error:  # preserve per-target failures in the report
                    failures.append({"accession": accession, "pdb_id": pdb_id, "reason": str(error)[:300]})
    # Keep one independent result per source protein.
    unique = {}
    for result in results:
        current = unique.get(result["source_index"])
        if current is None or result["n_matched_residues"] > current["n_matched_residues"]:
            unique[result["source_index"]] = result
    results = sorted(unique.values(), key=lambda item: item["accession"])
    return results, failures


def write_markdown(report, path):
    summary = report["summary"]
    lines = [
        "# AF2-RMSF Experimental Ensemble Calibration",
        "",
        f"- Verdict: `{summary['verdict']}`",
        f"- Calibrated proteins: {summary['n_proteins']}",
        f"- Median Spearman: {summary['median_spearman']}",
        f"- Median partial Spearman controlling 1-pLDDT: {summary['median_partial_spearman']}",
        f"- Median top-20% recall: {summary['median_top20_recall']}",
        f"- Median flexible-region AUROC: {summary['median_flexible_auc']}",
        f"- Annotated-IDP subset: {report['subgroups']['annotated_idp']}",
        "- Important limitation: deposited NMR models are restraint-compatible ensembles, not Boltzmann-weighted trajectories.",
        "- MSA-depth confounding is not measured because the source LMDB does not retain MSA depth.",
        "",
        "| UniProt | PDB | Models | Residues | Spearman | Partial | Top20 | AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["proteins"]:
        lines.append(
            f"| {item['accession']} | {item['pdb_id']} | {item['n_models']} | "
            f"{item['n_matched_residues']} | {item['spearman']} | "
            f"{item['partial_spearman']} | {item['top20_recall']:.3f} | "
            f"{item['flexible_auc']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmdb", default="data/confidence_conformation_v5/confidence_train.lmdb")
    parser.add_argument("--confidence-lmdb", default="data/confidence_unified_v2")
    parser.add_argument("--cache-dir", default="data/calibration_nmr_pdb")
    parser.add_argument("--output", default="data/confidence_conformation_v5/rmsf_calibration_1289.json")
    parser.add_argument("--minimum-proteins", type=int, default=10)
    parser.add_argument("--minimum-residues", type=int, default=20)
    parser.add_argument("--require-count", type=int, default=0)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    entries = read_lmdb(args.lmdb)
    if args.require_count and len(entries) != args.require_count:
        raise RuntimeError(f"Conformation build incomplete: {len(entries)}/{args.require_count}")
    accessions = sorted({
        str(entry.get("pdb_id", "")).split("-")[0]
        for entry in entries
        if UNIPROT_PATTERN.fullmatch(str(entry.get("pdb_id", "")).split("-")[0])
    })
    output_path = Path(args.output)
    discovery_path = output_path.with_name("rmsf_calibration_entities.json")
    if args.offline:
        entity_map = json.loads(discovery_path.read_text(encoding="utf-8"))
    else:
        entity_map = discover_nmr_entities(accessions)
        discovery_path.parent.mkdir(parents=True, exist_ok=True)
        discovery_path.write_text(json.dumps(entity_map, indent=2), encoding="utf-8")
    results, failures = make_report(
        entries,
        entity_map,
        Path(args.cache_dir),
        confidence_by_sequence(args.confidence_lmdb),
        args.minimum_residues,
    )
    summary = calibration_verdict(results, args.minimum_proteins)
    annotated_idp = [
        item for item in results
        if item.get("disorder_fraction") is not None
        and float(item["disorder_fraction"]) >= 0.20
    ]
    annotated_ordered = [
        item for item in results
        if item.get("disorder_fraction") is not None
        and float(item["disorder_fraction"]) < 0.05
    ]
    subgroups = {
        "annotated_idp": calibration_verdict(annotated_idp, args.minimum_proteins),
        "annotated_ordered": calibration_verdict(annotated_ordered, args.minimum_proteins),
    }
    report = {
        "schema_version": 1,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_lmdb": args.lmdb,
        "source_count": len(entries),
        "n_accessions_queried": len(accessions),
        "n_accessions_with_nmr": len(entity_map),
        "label_definition": "C-alpha RMSF across five AF2 random seeds after Kabsch alignment",
        "reference_definition": "C-alpha RMSF across deposited solution-NMR models after Kabsch alignment",
        "reference_limitation": (
            "Deposited NMR models represent structures compatible with experimental restraints; "
            "they are not Boltzmann-weighted samples and do not define physical populations."
        ),
        "confounders": {
            "plddt": "controlled per protein using partial Spearman against 1-pLDDT",
            "sequence_length": "reported per protein; aggregate performance uses within-protein ranks",
            "msa_depth": "unavailable in source LMDB and therefore not controlled",
        },
        "thresholds": {
            "minimum_proteins": args.minimum_proteins,
            "physical_label_median_spearman": 0.30,
            "physical_label_median_top20_recall": 0.30,
            "physical_label_median_partial_spearman": 0.15,
        },
        "summary": summary,
        "subgroups": subgroups,
        "calibration_source_indices": sorted({item["source_index"] for item in results}),
        "proteins": results,
        "failures": failures,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, output_path.with_suffix(".md"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
