#!/usr/bin/env python
"""Build a deduplicated retrospective expanded-IDP SAbDab registry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_pdb(value):
    text = str(value).strip().casefold()
    if text.startswith("pdb_"):
        text = text[4:]
    return text[-4:]


def split_pipe(value):
    return [part.strip() for part in str(value or "").split("|") if part.strip()]


def classify_target(text, ontology):
    text = text.casefold()
    matches = [
        target for target, spec in ontology.items()
        if any(pattern.casefold() in text for pattern in spec["patterns"])
    ]
    return matches[0] if len(matches) == 1 else None


def build(config_path, registry_path, audit_path):
    for path in (registry_path, audit_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite expanded registry: {path}")
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    final_path = ROOT / config["isolation"]["final_confirmation_contract"]
    if sha256(final_path) != config["isolation"]["final_confirmation_contract_sha256"]:
        raise RuntimeError("Frozen final confirmation contract hash changed")
    sabdab_path = ROOT / config["sources"]["sabdab_summary"]
    reference_path = ROOT / config["sources"]["exposure_reference"]
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    exposed = set(reference["exact_exposed_pdb_ids"])
    ontology = config["target_ontology"]
    candidates, exclusions = [], Counter()
    with sabdab_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            text = " | ".join((
                row.get("antigen_name") or "",
                row.get("compound") or "",
                row.get("short_header") or "",
            ))
            target = classify_target(text, ontology)
            if target is None:
                exclusions["not_one_expanded_idp_target"] += 1
                continue
            if not row.get("Hchain") or row["Hchain"] == "NA":
                exclusions["no_heavy_chain"] += 1
                continue
            antigen_chains = split_pipe(row.get("antigen_chain"))
            if not antigen_chains or antigen_chains == ["NA"]:
                exclusions["no_antigen_chain"] += 1
                continue
            antigen_types = split_pipe(row.get("antigen_type"))
            allowed = set(config["eligibility"]["antigen_types_allowed"])
            if not any(antigen_type in allowed for antigen_type in antigen_types):
                exclusions["antigen_type_not_allowed"] += 1
                continue
            pdb_id = normalize_pdb(row["PDB"])
            lineage = f"{row.get('HEAVY_ID') or 'NA'}|{row.get('LIGHT_ID') or 'NA'}"
            candidates.append({
                "component_id": f"{target.upper()}_{pdb_id.upper()}_{row['Hchain']}",
                "pdb_id": pdb_id,
                "sabdab_id": row.get("SABDAB_ID"),
                "target": target,
                "target_category": ontology[target]["category"],
                "lineage_proxy": lineage,
                "heavy_chain": row["Hchain"],
                "light_chain": None if row.get("Lchain") in (None, "", "NA") else row["Lchain"],
                "antigen_chains": antigen_chains,
                "antigen_type": antigen_types,
                "antigen_name": row.get("antigen_name"),
                "compound": row.get("compound"),
                "method": row.get("method"),
                "resolution": row.get("resolution"),
                "exactly_exposed_before_v2_registry": pdb_id in exposed,
                "exposure_classification": "metadata_exposed_development_only",
            })
    deduplicated = {}
    for row in sorted(candidates, key=lambda value: (
        value["target"], value["lineage_proxy"], value["pdb_id"],
        value["heavy_chain"], value["component_id"],
    )):
        key = (row["target"], row["lineage_proxy"], row["pdb_id"])
        deduplicated.setdefault(key, row)
    records = list(deduplicated.values())
    target_counts = Counter(row["target"] for row in records)
    lineage_sets = defaultdict(set)
    for row in records:
        lineage_sets[row["target"]].add(row["lineage_proxy"])
    registry = {
        "schema_version": 1,
        "status": "expanded_idp_development_registry_frozen",
        "classification": config["classification"],
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "source_sabdab": str(sabdab_path),
        "source_sabdab_sha256": sha256(sabdab_path),
        "source_exposure_reference": str(reference_path),
        "source_exposure_reference_sha256": sha256(reference_path),
        "component_count": len(records),
        "target_count": len(target_counts),
        "lineage_proxy_count": len({row["lineage_proxy"] for row in records}),
        "target_component_counts": dict(sorted(target_counts.items())),
        "target_lineage_proxy_counts": {
            target: len(values) for target, values in sorted(lineage_sets.items())
        },
        "records": records,
        "claim_boundary": config["claim_boundary"],
    }
    audit = {
        "schema_version": 1,
        "status": "expanded_registry_audit_complete",
        "raw_eligible_rows": len(candidates),
        "deduplicated_components": len(records),
        "duplicate_rows_removed": len(candidates) - len(records),
        "exclusion_counts": dict(exclusions),
        "exact_prior_exposure_count": sum(
            row["exactly_exposed_before_v2_registry"] for row in records
        ),
        "newly_metadata_exposed_count": sum(
            not row["exactly_exposed_before_v2_registry"] for row in records
        ),
        "future_confirmation_eligible_count": 0,
        "final_confirmation_contract_sha256": sha256(final_path),
        "isolation_passed": True,
        "decision": "use_for_expanded_development_only",
        "claim_boundary": config["claim_boundary"],
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="ascii")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="ascii")
    return registry, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/benchmarks/idp_ensemble_expanded_development_v2.yml"),
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    registry, audit = build(
        config_path, ROOT / config["outputs"]["registry"],
        ROOT / config["outputs"]["audit"],
    )
    print(json.dumps({
        "status": registry["status"],
        "components": registry["component_count"],
        "targets": registry["target_count"],
        "lineage_proxies": registry["lineage_proxy_count"],
        "target_component_counts": registry["target_component_counts"],
        "target_lineage_proxy_counts": registry["target_lineage_proxy_counts"],
        "future_confirmation_eligible": audit["future_confirmation_eligible_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
