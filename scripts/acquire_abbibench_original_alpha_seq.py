"""Acquire original AlphaSeq archives and inspect schemas/targets without numeric labels."""
import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/abbibench_sequence_mapping_v1"


def main():
    tree = json.loads((OUT / "alpha_seq_tree.json").read_text())
    dest = OUT / "original_alpha_seq"
    dest.mkdir(exist_ok=True)
    def acquire(item):
        name = Path(item["path"]).name
        path = dest / name
        url = f"https://raw.githubusercontent.com/mit-ll/AlphaSeq_Antibody_Dataset/{tree['sha']}/{item['path']}"
        if not path.exists():
            raw = urllib.request.urlopen(url, timeout=120).read()
            if len(raw) != item["size"]:
                raise ValueError("Original archive length differs from GitHub tree")
            path.write_bytes(raw)
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX")]
            if len(names) != 1:
                raise ValueError("Expected exactly one CSV")
            with archive.open(names[0]) as stream:
                source_rows = csv.reader(io.TextIOWrapper(stream, encoding="utf-8-sig"))
                fields = next(row for row in source_rows if {"HC", "LC", "Target", "Pred_affinity"}.issubset(row))
                reader = (dict(zip(fields, values)) for values in source_rows if values)
                target_key = next(k for k in fields if k.casefold() == "target")
                assay_key = next(k for k in fields if k.casefold() == "assay")
                targets, assays = Counter(), Counter()
                count = 0
                for row in reader:
                    count += 1
                    targets[row[target_key]] += 1
                    assays[row[assay_key]] += 1
        result = {"file": name, "url": url, "revision": tree["sha"],
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "csv_member": names[0],
                  "columns": fields, "rows": count, "target_counts": dict(targets), "assay_counts": dict(assays),
                  "raw_labels_downloaded_and_stored": True, "numeric_label_values_analyzed": False}
        print(json.dumps(result, indent=2), flush=True)
        return result
    items = [r for r in tree["tree"] if r["path"].endswith(".csv.zip")]
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(acquire, items))
    (dest / "schema_audit.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
