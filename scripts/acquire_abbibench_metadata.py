"""Snapshot only public dataset metadata, never individual affinity labels."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/ecls_independent_feasibility_v1/abbibench_metadata"
BASE = "https://huggingface.co"
REPO = "AbBibench/Antibody_Binding_Benchmark_Dataset"


def fetch(url, path):
    if not path.exists():
        data = urllib.request.urlopen(url, timeout=45).read()
        path.write_bytes(data)
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    info = fetch(f"{BASE}/api/datasets/{REPO}", OUT / "repository.json")
    revision = info["sha"]
    fetch(f"{BASE}/datasets/{REPO}/resolve/{revision}/metadata.json", OUT / "metadata.json")
    files = [r["rfilename"] for r in info["siblings"]]
    reference = json.loads((OUT.parent / "reference_union.json").read_text())
    exposed = set(reference["exact_exposed_pdb_ids"])
    structures = []
    for name in files:
        if not name.startswith("complex_structure/") or not name.endswith(".pdb"):
            continue
        stem = Path(name).stem.split("_")[0].lower()
        pdb = stem if len(stem) == 4 and stem[0].isdigit() else None
        structures.append({"file": name, "filename_pdb_id_unverified": pdb,
                           "exact_prior_exposure": pdb in exposed if pdb else None,
                           "sequence_isolation": "not_assessed"})
    result = {"classification": "metadata_only_functional_extension_feasibility",
              "revision": revision, "repository": f"{BASE}/datasets/{REPO}",
              "affinity_tables_listed_not_downloaded": [x for x in files if x.startswith("binding_affinity/")],
              "structures": structures, "model_scoring": False,
              "individual_measurements_accessed": False,
              "published_aggregate_results_seen": True,
              "independence_established": False,
              "caveat": "Filename IDs are provisional; different mutant rows do not establish antigen or lineage independence. Protein antigens extend the frozen peptide scope.",
              "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT / "repository.json", OUT / "metadata.json")}}
    target = OUT / "audit.json"
    text = json.dumps(result, indent=2) + "\n"
    if target.exists() and target.read_text() != text:
        raise RuntimeError("Refusing to overwrite a different audit")
    target.write_text(text)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
