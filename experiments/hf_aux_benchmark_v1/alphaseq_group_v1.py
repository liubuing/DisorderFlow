#!/usr/bin/env python
"""AlphaSeq cluster-held-out sensitivity check.

Random 5-fold on related variants is optimistic. This script clusters ESM2
embeddings (KMeans) and evaluates with GroupKFold, so near-duplicate sequences
stay in the same fold. It is a sensitivity check, not a sequence-identity
cluster split (MMseqs2 was unavailable in this environment).
"""
from __future__ import annotations

import json
from pathlib import Path

import esm
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/hf_aux_benchmark_v1/alphaseq"
OUT = ROOT / "results/hf_aux_benchmark_v1/alphaseq"
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
            features = embed(group["sequence"].tolist(), model, alphabet, device)
            comp = np.vstack([composition(s) for s in group["sequence"]])
            y = group["affinity"].to_numpy()
            clusters = KMeans(n_clusters=min(50, max(5, len(group) // 100)),
                              n_init=10, random_state=20261111).fit_predict(features)
            row = {"dataset": dataset, "target": target, "n": len(group),
                   "n_clusters": int(len(set(clusters)))}
            row["composition_group"] = evaluate(comp, y, clusters)
            row["esm2_group"] = evaluate(features, y, clusters)
            row["combo_group"] = evaluate(np.hstack([comp, features]), y, clusters)
            rows.append(row)
            print(row, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "group_split.csv", index=False)
    summary = {
        "n": len(frame),
        "mean": {c: float(frame[c].mean()) for c in ("composition_group", "esm2_group", "combo_group")},
        "note": "KMeans-on-ESM2 group split; near-duplicates grouped; not MMseqs2 identity clusters",
    }
    (OUT / "group_split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
