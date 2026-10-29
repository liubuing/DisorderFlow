"""Stream fixed public CSVs into sequence-only JSONL; discard binding score strings.

Raw response bytes contain labels and are hashed in transit, but labels are not
converted to numbers, logged, persisted or used in any selection. This is not a
claim that experimental data were never transmitted to this machine.
"""
import csv
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/abbibench_sequence_mapping_v1"


def project(spec):
    # 把固定版本公共 CSV 投影为仅序列 JSONL:
    #   - 边下载边对原始字节做 SHA-256（raw_transmitted_csv_sha256 留证），
    #     但标签字符串不解析为数值、不落盘、不参与任何选择;
    #   - 表头必须与冻结版本完全一致（防静默 schema 变更）;
    #   - 先写 .partial 再原子替换；receipts 记录双向哈希，重跑时校验
    #     缓存投影未被改动。
    name = Path(spec["file"]).stem
    path = OUT / "sequences" / (name + ".jsonl")
    record = OUT / "receipts" / (name + ".json")
    if record.exists():
        result = json.loads(record.read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != result["projection_sha256"]:
            raise ValueError("Cached sequence projection hash mismatch")
        return result
    path.parent.mkdir(parents=True, exist_ok=True)
    record.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    digest = hashlib.sha256()
    n = 0
    with urllib.request.urlopen(spec["url"], timeout=90) as response, partial.open("w", encoding="ascii") as output:
        def lines():
            for line in response:
                digest.update(line)
                yield line.decode("utf-8-sig")
        reader = csv.DictReader(lines())
        if reader.fieldnames != spec["columns"]:
            raise ValueError("Header changed at fixed revision")
        for n, row in enumerate(reader, 1):
            output.write(json.dumps({"source_data_row": n, "heavy_chain_seq": row["heavy_chain_seq"],
                                     "light_chain_seq": row["light_chain_seq"]}) + "\n")
    partial.replace(path)
    result = {"file": spec["file"], "url": spec["url"], "rows": n,
              "raw_transmitted_csv_sha256": digest.hexdigest(),
              "projection_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "score_strings_received_in_csv": True, "score_values_numerically_parsed": False,
              "score_values_persisted": False}
    record.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Projected {name}: {n} sequence rows", flush=True)
    return result


def main():
    headers = json.loads((OUT / "headers.json").read_text())
    # Require an endpoint assessment snapshot before opening any full table.
    # 必须先冻结端点状态字典（endpoint_dictionary.json），防止在未评估
    # 标签可用性之前就接触全表数据。
    endpoint = OUT / "endpoint_dictionary.json"
    if not endpoint.exists():
        raise RuntimeError("Freeze endpoint status dictionary first")
    with ThreadPoolExecutor(max_workers=3) as pool:
        receipts = list(pool.map(project, headers["rows"]))
    result = {"revision": headers["revision"], "endpoint_dictionary_sha256": hashlib.sha256(endpoint.read_bytes()).hexdigest(),
              "receipts": receipts, "total_rows": sum(r["rows"] for r in receipts)}
    (OUT / "acquisition.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
