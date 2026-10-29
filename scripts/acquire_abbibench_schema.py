"""Read only the header line of each version-pinned CSV, never data rows."""
# ECLS 独立可行性审查的数据源模式采集：只取 AbBibench 各 CSV 的表头列名，
# 明确不读任何数据行——满足"评估新数据源不引入其样本"的取证边界。
# 版本用仓库 JSON 里的 commit sha 固定，保证可复现；产物 headers.json
# write-once（已存在则直接回读打印，不重复请求）。
import csv
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/ecls_independent_feasibility_v1/abbibench_metadata"
OUT = ROOT / "data/abbibench_sequence_mapping_v1"


def main():
    info = json.loads((SOURCE / "repository.json").read_text())
    OUT.mkdir(exist_ok=True)
    files = [r["rfilename"] for r in info["siblings"] if r["rfilename"].endswith(".csv")]
    def get_header(name):
        # 按 commit sha 固定版本解析，避免 HEAD 漂移；utf-8-sig 兼容 HF 导出的 BOM
        url = f"https://huggingface.co/datasets/AbBibench/Antibody_Binding_Benchmark_Dataset/resolve/{info['sha']}/{name}"
        with urllib.request.urlopen(url, timeout=60) as response:
            header = response.readline().decode("utf-8-sig").rstrip("\r\n")
        return {"file": name, "url": url, "columns": next(csv.reader([header]))}
    path = OUT / "headers.json"
    if path.exists():
        result = json.loads(path.read_text())
    else:
        with ThreadPoolExecutor(max_workers=3) as pool:
            rows = list(pool.map(get_header, files))
        result = {"revision": info["sha"], "rows": rows, "data_rows_read": False}
        path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
