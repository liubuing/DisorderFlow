# One-off: build SHA256 manifest for iBEC_2026_材料与权重_完整包/02_模型权重
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\biological\DisorderFlow")
ARCHIVE = ROOT / "iBEC_2026_材料与权重_完整包"
WEIGHTS = ARCHIVE / "02_模型权重"

SOURCE_MAP = {
    "主权重_AntibodyDesignBFN_best.pt": "data/pretrained_candidates/AntibodyDesignBFN_best.pt",
    "calibration_artifacts": "calibration_artifacts",
    "训练run_best": "logs/<run>/checkpoints/best.pt",
    "results_pae_surrogate_revision_v4": "results/pae_surrogate_revision_v4",
    "results_v5_1_corrective": "results/v5_1_corrective",
}

WEIGHT_EXTS = {".pt", ".pth", ".pkl", ".json", ".yml", ".yaml"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    entries = []
    t0 = time.time()
    files = sorted(p for p in WEIGHTS.rglob("*") if p.is_file())
    for i, p in enumerate(files, 1):
        rel = p.relative_to(WEIGHTS).as_posix()
        top = rel.split("/", 1)[0]
        src = SOURCE_MAP.get(top, SOURCE_MAP.get(top.split("/")[0], "?"))
        if top == "训练run_best":
            run = rel.split("/", 2)[1]
            src = f"logs/{run}/checkpoints/best.pt"
        elif top == "calibration_artifacts":
            src = f"calibration_artifacts/{rel.split('/', 1)[1]}"
        elif top.startswith("results_"):
            sub = rel.split("/", 1)[1] if "/" in rel else ""
            src = f"results/{top.replace('results_', '', 1)}/{sub}" if sub else f"results/{top.replace('results_', '', 1)}"
        entry = {
            "file": rel,
            "bytes": p.stat().st_size,
            "sha256": sha256(p) if p.suffix.lower() in WEIGHT_EXTS else None,
            "source_in_project": src,
        }
        entries.append(entry)
        if i % 10 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] {rel}", flush=True)

    manifest = {
        "archive_root": "iBEC_2026_材料与权重_完整包",
        "weights_dir": "02_模型权重",
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hash_algorithm": "sha256 (hashed for weight/data files; None for auxiliary logs)",
        "submission_zip_sha256": "a135defcf4febd21f5eb4aec43d522e53d493518c94bc8f729566bd1cc8e2281",
        "notes": [
            "主权重 AntibodyDesignBFN_best.pt 即平台/基准配置引用的设计 checkpoint（data/pretrained_candidates/AntibodyDesignBFN_best.pt），已托管于 HuggingFace: liubuing/disorderflow",
            "训练run_best 收录 69 个非 smoke 正式训练 run 的 best.pt；早期 v11-v19 run 与 2026-07-30 批次的部分 run 其 checkpoints 目录已为空，权重未留存，无法收录",
            "13 个 smoke run 的 best.pt 有意排除（冒烟测试产物）",
            "第三方 AlphaFold/colabfold 参数缓存（约 3.5 GB，位于 dsh runtime cache）不属于自训权重，未收录",
        ],
        "files": entries,
    }
    out = WEIGHTS / "WEIGHTS_MANIFEST.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(e["bytes"] for e in entries)
    print(f"DONE: {len(entries)} files, {total / 1024**3:.2f} GB, manifest -> {out}, elapsed {time.time() - t0:.0f}s")
    sys.exit(0)


if __name__ == "__main__":
    main()
