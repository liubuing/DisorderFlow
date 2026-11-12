#!/usr/bin/env python
"""AlphaSeq 90% identity cluster split with MMseqs2.

Replaces the KMeans-on-ESM2 sensitivity check with actual sequence-identity
clusters. MMseqs2 Windows binary is expected at .tools/mmseqs2/mmseqs/bin/mmseqs.exe.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import esm
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/hf_aux_benchmark_v1/alphaseq"
OUT = ROOT / "results/hf_aux_benchmark_v1/alphaseq"
WORK = OUT / "mmseqs_work"
MMSEQS = ROOT / ".tools/mmseqs2/mmseqs/bin/mmseqs.exe"
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def composition(sequence: str) -> np.ndarray:
    return np.array([sequence.count(aa) / max(1, len(sequence)) for aa in ALPHABET])


def load(path: Path, max_rows: int = 12000) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame = frame[
        frame["mata_sequence"].notna()
        & frame["alphaseq_affinity"].notna()
        & frame["above_background"].eq(True)
        & frame["sufficient_replicate_observations"].eq(True)
    ].copy()
    frame["sequence"] = frame["mata_sequence"].astype(str).str.upper()
    frame = frame[frame["sequence"].str.len().between(20, 600)]
    frame = frame[~frame["sequence"].str.contains(r"[^ACDEFGHIKLMNPQRSTVWY]", regex=True)]
    frame = frame.groupby(["matalpha_description", "sequence"], as_index=False).agg(
        affinity=("alphaseq_affinity", "mean"))
    if len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=20261111).reset_index(drop=True)
    return frame


def embed(sequences, model, alphabet, device, batch_size=64):
    converter = alphabet.get_batch_converter()
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start:start + batch_size]
            data = [(str(i), s) for i, s in enumerate(batch)]
            _, _, tokens = converter(data)
            tokens = tokens.to(device)
            out = model(tokens, repr_layers=[6], return_contacts=False)
            rep = out["representations"][6]
            for i, s in enumerate(batch):
                rows.append(rep[i, 1:len(s) + 1].mean(0).cpu().numpy())
    return np.vstack(rows)


def mmseqs_clusters(sequences: list[str], tag: str) -> list[int]:
    """Return cluster id per sequence using MMseqs2 90% identity / 80% coverage."""
    folder = WORK / tag
    folder.mkdir(parents=True, exist_ok=True)
    fasta = folder / "input.fasta"
    with fasta.open("w", encoding="ascii") as handle:
        for index, sequence in enumerate(sequences):
            handle.write(f">seq_{index}\n{sequence}\n")
    db = folder / "db"
    clu = folder / "clu"
    tmp = folder / "tmp"
    tsv = folder / "clusters.tsv"
    for path in (db, clu, tsv):
        if path.exists():
            path.unlink()
    commands = [
        [str(MMSEQS), "createdb", fasta.as_posix(), db.as_posix(), "-v", "0"],
        [str(MMSEQS), "cluster", db.as_posix(), clu.as_posix(), tmp.as_posix(),
         "--min-seq-id", "0.9", "-c", "0.8", "--cov-mode", "0",
         "--single-step-clustering", "1", "--threads", "4", "-v", "0"],
        [str(MMSEQS), "createtsv", db.as_posix(), db.as_posix(), clu.as_posix(),
         tsv.as_posix(), "-v", "0"],
    ]
    for command in commands:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=1800)
        if completed.returncode:
            raise RuntimeError(f"MMseqs2 failed: {' '.join(command)}\n{completed.stderr[-1000:]}")
    mapping = {}
    with tsv.open(encoding="utf-8") as handle:
        for line in handle:
            representative, member = line.rstrip("\n").split("\t")
            mapping[int(member.split("_")[1])] = representative
    return [mapping.get(index, f"singleton_{index}") for index in range(len(sequences))]


def evaluate(features, y, groups, folds=5):
    cv = GroupKFold(n_splits=folds)
    values = []
    for train, test in cv.split(features, y, groups):
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(features[train], y[train])
        pred = model.predict(features[test])
        if len(np.unique(y[test])) > 1:
            values.append(float(spearmanr(y[test], pred).statistic))
    return float(np.mean(values)) if values else None


def main():
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    rows = []
    for dataset in ("YM_0989", "YM_0852", "YM_0990"):
        frame = load(DATA / f"{dataset}.parquet")
        print(dataset, len(frame), flush=True)
        for target, group in frame.groupby("matalpha_description"):
            if len(group) < 500:
                continue
            group = group.reset_index(drop=True)
            clusters = mmseqs_clusters(group["sequence"].tolist(), f"{dataset}_{target}")
            print(f"  {target}: n={len(group)} clusters={len(set(clusters))}", flush=True)
            features = embed(group["sequence"].tolist(), model, alphabet, device)
            comp = np.vstack([composition(s) for s in group["sequence"]])
            y = group["affinity"].to_numpy()
            row = {"dataset": dataset, "target": target, "n": len(group),
                   "n_clusters": int(len(set(clusters)))}
            row["composition_mmseqs"] = evaluate(comp, y, clusters)
            row["esm2_mmseqs"] = evaluate(features, y, clusters)
            row["combo_mmseqs"] = evaluate(np.hstack([comp, features]), y, clusters)
            rows.append(row)
            print(row, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "mmseqs_split.csv", index=False)
    summary = {
        "n": len(frame),
        "mean": {c: float(frame[c].mean()) for c in
                 ("composition_mmseqs", "esm2_mmseqs", "combo_mmseqs")},
        "note": "MMseqs2 90% identity / 80% coverage, single-step clustering",
    }
    (OUT / "mmseqs_split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
