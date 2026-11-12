#!/usr/bin/env python
"""Open AlphaSeq sequence-only affinity benchmark v1.

For each of the nine aalphabio/open-alphaseq datasets:
  - keep above_background and sufficient_replicate_observations rows
  - deduplicate by antibody sequence within each target
  - compare composition-only, ESM2-only, and composition+ESM2 Ridge
  - report within-target 5-fold Spearman

This is an optimistic (random-fold) sequence benchmark, not a cluster-held-out
confirmation. It is a development analysis; it does not change any frozen
ECLS/PAE result and makes no IDP/peptide binding claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import esm
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/hf_aux_benchmark_v1/alphaseq"
OUT = ROOT / "results/hf_aux_benchmark_v1/alphaseq"
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def composition(sequence: str) -> np.ndarray:
    sequence = "".join(aa if aa in ALPHABET else "A" for aa in sequence.upper())
    return np.array([sequence.count(aa) / max(1, len(sequence)) for aa in ALPHABET])


def load_and_filter(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame = frame[
        frame["mata_sequence"].notna()
        & frame["matalpha_sequence"].notna()
        & frame["alphaseq_affinity"].notna()
        & frame["above_background"].eq(True)
        & frame["sufficient_replicate_observations"].eq(True)
    ].copy()
    frame["sequence"] = frame["mata_sequence"].astype(str).str.upper()
    frame = frame[frame["sequence"].str.len().between(20, 600)]
    frame = frame[~frame["sequence"].str.contains(r"[^ACDEFGHIKLMNPQRSTVWY]", regex=True)]
    return frame


def deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["matalpha_description", "sequence"], as_index=False).agg(
        affinity=("alphaseq_affinity", "mean"),
        n=("alphaseq_affinity", "size"),
    )
    return grouped


def embed_esm(sequences: list[str], model, alphabet, device, batch_size=64) -> np.ndarray:
    converter = alphabet.get_batch_converter()
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start:start + batch_size]
            data = [(str(index), sequence) for index, sequence in enumerate(batch)]
            _, _, tokens = converter(data)
            tokens = tokens.to(device)
            output = model(tokens, repr_layers=[6], return_contacts=False)
            representation = output["representations"][6]
            for index, sequence in enumerate(batch):
                rows.append(representation[index, 1:len(sequence) + 1].mean(0).cpu().numpy())
            if start % (batch_size * 20) == 0:
                print(f"  embedded {start + len(batch)}/{len(sequences)}", flush=True)
    return np.vstack(rows)


def evaluate_target(frame: pd.DataFrame, esm_features: np.ndarray, seed: int) -> dict:
    y = frame["affinity"].to_numpy(dtype=float)
    comp = np.vstack([composition(sequence) for sequence in frame["sequence"]])
    folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    results = {"composition": [], "esm2": [], "composition_esm2": []}
    for train, test in folds.split(comp):
        for name, features in (
            ("composition", comp),
            ("esm2", esm_features),
            ("composition_esm2", np.hstack([comp, esm_features])),
        ):
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(features[train], y[train])
            prediction = model.predict(features[test])
            if len(np.unique(y[test])) > 1:
                rho, _ = spearmanr(y[test], prediction)
                results[name].append(float(rho))
    return {name: float(np.mean(values)) if values else None
            for name, values in results.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-target", type=int, default=20000)
    parser.add_argument("--datasets", default="")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    paths = sorted(DATA.glob("*.parquet"))
    if args.datasets:
        keep = {x.strip() for x in args.datasets.split(",") if x.strip()}
        paths = [path for path in paths if path.stem in keep]

    print("loading ESM2 t6 8M", flush=True)
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    rows = []
    for path in paths:
        dataset = path.stem
        frame = load_and_filter(path)
        frame = deduplicate(frame)
        print(f"{dataset}: {len(frame)} unique sequence-target rows", flush=True)
        for target, group in frame.groupby("matalpha_description"):
            group = group.sort_values("affinity", ascending=False)
            if len(group) > args.max_per_target:
                group = group.sample(args.max_per_target, random_state=20261111)
            group = group.reset_index(drop=True)
            if len(group) < 200 or group["affinity"].nunique() < 20:
                continue
            print(f"  {dataset}/{target}: n={len(group)}", flush=True)
            features = embed_esm(group["sequence"].tolist(), model, alphabet, device)
            result = evaluate_target(group, features, seed=20261111)
            rows.append({
                "dataset": dataset, "target": target, "n": len(group), **result,
            })
            print(f"    {result}", flush=True)

    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(OUT / "per_target.csv", index=False)
    summary = {
        "n_datasets": int(result_frame["dataset"].nunique()) if len(result_frame) else 0,
        "n_targets": int(len(result_frame)),
        "means": {
            column: float(result_frame[column].mean())
            for column in ("composition", "esm2", "composition_esm2")
            if column in result_frame
        },
        "fraction_esm2_beats_composition": float(
            (result_frame["esm2"] > result_frame["composition"]).mean()
        ) if len(result_frame) else None,
        "fraction_combo_beats_composition": float(
            (result_frame["composition_esm2"] > result_frame["composition"]).mean()
        ) if len(result_frame) else None,
        "claim_boundary": (
            "optimistic random-fold sequence benchmark on public AlphaSeq affinity; "
            "not cluster-held-out, not IDP/peptide, not binding validation"
        ),
    }
    with (OUT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
