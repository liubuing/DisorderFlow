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
    # 带本地缓存的下载：文件已存在则直接读缓存，保证脚本可重入且不重复请求网络。
    if not path.exists():
        data = urllib.request.urlopen(url, timeout=45).read()
        path.write_bytes(data)
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    # 只快照数据集级元数据（repo 信息 + metadata.json），绝不下载或查看逐条亲和力标签，
    # 以维持与已冻结肽段范围的独立性审查边界。
    OUT.mkdir(parents=True, exist_ok=True)
    info = fetch(f"{BASE}/api/datasets/{REPO}", OUT / "repository.json")
    revision = info["sha"]
    fetch(f"{BASE}/datasets/{REPO}/resolve/{revision}/metadata.json", OUT / "metadata.json")
    # 遍历 repo 文件清单：仅登记 complex_structure/*.pdb 文件名，从文件名推断（未经验证的）PDB ID，
    # 并标记其是否已出现在 reference_union 的精确暴露列表中；亲和力表只列出文件名、不下载。
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
    # 审计文件只写一次：已存在且内容不同则拒绝覆盖，保证独立可行性审计不可被静默改写。
    target = OUT / "audit.json"
    text = json.dumps(result, indent=2) + "\n"
    if target.exists() and target.read_text() != text:
        raise RuntimeError("Refusing to overwrite a different audit")
    target.write_text(text)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
