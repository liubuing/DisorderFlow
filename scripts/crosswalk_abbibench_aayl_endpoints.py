"""Audit a frozen expected score transformation on H3-compatible AAYL candidates."""
import csv
import hashlib
import io
import json
import math
import statistics
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/abbibench_sequence_mapping_v1"


def original_rows(path):
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".csv"))
        with archive.open(name) as stream:
            reader = csv.reader(io.TextIOWrapper(stream, encoding="utf-8-sig"))
            fields = next(row for row in reader if {"HC", "LC", "Pred_affinity", "Target"}.issubset(row))
            for values in reader:
                if values:
                    yield dict(zip(fields, values))


def finite_value(value):
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def main():
    protocol = OUT / "AAYL_endpoint_crosswalk_protocol.json"
    rules = json.loads(protocol.read_text())
    if rules["comparison_tolerance"] != .00001:
        raise ValueError("Unexpected frozen tolerance")
    candidates = [json.loads(line) for line in (OUT / "provisional_H3_sequence_rows.jsonl").read_text().splitlines()]
    candidates = [r for r in candidates if "/aayl" in r["table"]]
    by_table = defaultdict(list)
    for row in candidates:
        by_table[row["table"]].append(row)
    schemas = {r["file"]: r for r in json.loads((OUT / "headers.json").read_text())["rows"]}
    receipts = {r["file"]: r for r in json.loads((OUT / "acquisition.json").read_text())["receipts"]}
    labels = {}
    label_cache = OUT / "crosswalk_curated_labels.json"
    if label_cache.exists():
        labels = json.loads(label_cache.read_text())
    else:
        for table, selected in sorted(by_table.items()):
            raw = urllib.request.urlopen(schemas[table]["url"], timeout=60).read()
            if hashlib.sha256(raw).hexdigest() != receipts[table]["raw_transmitted_csv_sha256"]:
                raise ValueError("Fixed CSV hash mismatch")
            indices = {r["source_data_row"] for r in selected}
            labels[table] = {}
            for index, row in enumerate(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))), 1):
                if index in indices:
                    labels[table][str(index)] = finite_value(row["binding_score"])
        label_cache.write_text(json.dumps(labels, indent=2) + "\n")
    original = {}
    source_profile = json.loads((OUT / "original_alpha_seq/schema_audit.json").read_text())
    source_hashes = {}
    for dataset_number, filename in ((1, "MITLL_AAlphaBio_Ab_Binding_dataset.csv.zip"), (2, "MITLL_AAlphaBio_Ab_Binding_dataset2.csv.zip")):
        source_path = OUT / "original_alpha_seq" / filename
        expected = next(r["sha256"] for r in source_profile["rows"] if r["file"] == filename)
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != expected:
            raise ValueError("Original source changed")
        source_hashes[filename] = expected
        wanted = {(r["heavy_chain_seq"], r["light_chain_seq"]) for r in candidates if ("_ML_" in r["table"]) == (dataset_number == 2)}
        groups = defaultdict(lambda: defaultdict(list))
        for row in original_rows(source_path):
            key = (row["HC"], row["LC"])
            if row["Target"] != "MIT_Target" or key not in wanted:
                continue
            groups[key][(row["Assay"], row["POI"])].append({"replicate": row["Replicate"], "value": finite_value(row["Pred_affinity"])})
        original[dataset_number] = groups
    rows = []
    for r in candidates:
        number = 2 if "_ML_" in r["table"] else 1
        groups = original[number].get((r["heavy_chain_seq"], r["light_chain_seq"]), {})
        score = labels[r["table"]][str(r["source_data_row"])]
        comparisons = []
        for (assay, poi), observations in sorted(groups.items()):
            finite = [o["value"] for o in observations if o["value"] is not None]
            expected = 9-statistics.mean(finite) if finite else None
            residual = score-expected if score is not None and expected is not None else None
            comparisons.append({"assay": assay, "poi": poi, "replicate_ids": [o["replicate"] for o in observations],
                "n_finite": len(finite), "n_missing": len(observations)-len(finite),
                "duplicate_replicate_ids": len({o["replicate"] for o in observations}) != len(observations),
                "expected_benchmark_score": expected, "curated_minus_expected": residual,
                "expected_conversion_matches": residual is not None and abs(residual) <= rules["comparison_tolerance"],
                "source_median_negative_log_molar": 9-statistics.median(finite) if finite else None})
        unambiguous = len(comparisons) == 1 and not comparisons[0]["duplicate_replicate_ids"]
        rows.append({"table": r["table"], "source_data_row": r["source_data_row"], "curated_score": score,
                     "original_groups": comparisons, "unambiguous_source": unambiguous,
                     "conversion_verified": unambiguous and comparisons[0]["expected_conversion_matches"]})
    summary = []
    for table in sorted(by_table):
        group = [r for r in rows if r["table"] == table]
        summary.append({"table": table, "candidates": len(group),
                        "matched_original_sequence": sum(bool(r["original_groups"]) for r in group),
                        "unambiguous_source": sum(r["unambiguous_source"] for r in group),
                        "conversion_verified": sum(r["conversion_verified"] for r in group),
                        "has_missing_original_replicate": sum(any(c["n_missing"] for c in r["original_groups"]) for r in group)})
    result = {"classification": "numeric_endpoint_crosswalk_not_model_evaluation", "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
              "source_archive_sha256": source_hashes, "summary": summary, "rows": rows,
              "model_scores_used": False,
              "exposure_note": "Two unrelated light-chain variant values were displayed while identifying dataset2 CSV preamble; crosswalk rules had already been frozen. Do not claim complete label blindness."}
    (OUT / "AAYL_endpoint_crosswalk.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
