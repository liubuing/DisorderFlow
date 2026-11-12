#!/usr/bin/env python
"""AB-bind mutation-effect benchmark v1.

Frozen protocol: experiments/hf_aux_benchmark_v1/protocol.json

For each AB-bind antibody-antigen complex, score every mutation with
ProteinMPNN leave-one-out conditional probabilities in two states:

  complex           : scored chain + direct partner chain
  partner_stripped  : scored chain only

Derived scores:
  dlogp_complex  = sum logP(mut) - logP(wt) in the complex state
  dlogp_stripped = same in the partner-stripped state
  dils           = dlogp_complex - dlogp_stripped
  composition    = leave-one-complex-out Ridge on 20-dim composition delta
  blosum62       = sum BLOSUM62(wt, mut)

Primary endpoint: within-complex Spearman(score, ddG); negative rho is good
because positive ddG is destabilizing.

This is a development benchmark. It is not an IDP/peptide binding claim and it
does not modify any frozen ECLS/PAE result.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices
from Bio.PDB import Atom, Chain, Model, PDBIO, PDBParser, Residue, Structure
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
AB_BIND = ROOT / ".tools/hf_probe/abbind.parquet"
SABDAB = ROOT / "data/sabdab_summary_all.tsv"
PDB_DIR = ROOT / "data/hf_aux_benchmark_v1/abbind_pdb"
OUT = ROOT / "results/hf_aux_benchmark_v1/abbind"
WORK = OUT / "work"
MPNN = ROOT / "ProteinMPNN" / "protein_mpnn_run.py"
WEIGHTS = ROOT / "ProteinMPNN" / "vanilla_model_weights"
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}
MUT_RE = re.compile(r"^([A-Za-z0-9]+):([A-Z])(-?\d+)([A-Za-z]?)([A-Z])$")
BLOSUM = substitution_matrices.load("BLOSUM62")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_dirs() -> None:
    for path in (OUT, WORK, PDB_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_ab_bind() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not AB_BIND.exists():
        raise FileNotFoundError(AB_BIND)
    ab = pd.read_parquet(AB_BIND)
    ab["pid"] = ab["#PDB"].str.lower().str.replace("hm_", "", regex=False)
    sab = pd.read_csv(SABDAB, sep="\t", dtype=str)
    sab["pid"] = sab["pdb"].str.lower()
    sab = sab.drop_duplicates("pid")[
        ["pid", "Hchain", "Lchain", "antigen_chain", "antigen_type", "antigen_name"]
    ]
    merged = ab.merge(sab, on="pid", how="inner")
    merged = merged[
        merged["Hchain"].notna()
        & merged["antigen_chain"].notna()
        & (merged["antigen_type"].fillna("") == "protein")
    ].copy()
    return ab, merged


def download_pdb(pid: str) -> Path:
    path = PDB_DIR / f"{pid}.pdb"
    if path.exists() and path.stat().st_size > 2000:
        return path
    url = f"https://files.rcsb.org/download/{pid.upper()}.pdb"
    log(f"download {url}")
    completed = subprocess.run(
        ["curl", "-sL", "--max-time", "120", url, "-o", str(path)],
        capture_output=True, text=True,
    )
    if completed.returncode or not path.exists() or path.stat().st_size < 2000:
        raise RuntimeError(f"PDB download failed for {pid}: {completed.stderr[-500:]}")
    return path


def parse_structure(pid: str):
    path = download_pdb(pid)
    return PDBParser(QUIET=True).get_structure(pid, str(path))


def standard_aa(residue) -> str | None:
    name = residue.resname.upper()
    return AA3.get(name)


def chain_residues(structure, chain_id: str):
    model = next(structure.get_models())
    try:
        chain = model[chain_id]
    except KeyError:
        return []
    out = []
    for residue in chain:
        if residue.id[0] != " ":
            continue
        aa = standard_aa(residue)
        if aa is None:
            continue
        out.append((residue, aa))
    return out


def residue_map(structure, chain_id: str) -> dict[tuple[int, str], tuple[int, str]]:
    """(resseq, icode) -> (1-based sequential index, wt aa)."""
    out = {}
    for index, (residue, aa) in enumerate(chain_residues(structure, chain_id), 1):
        key = (int(residue.id[1]), str(residue.id[2]).strip())
        out.setdefault(key, (index, aa))
    return out


def write_renumbered(structure, chain_ids: list[str], path: Path) -> dict[str, int]:
    """Write selected chains with sequential residue numbers, no gaps."""
    model = next(structure.get_models())
    new_structure = Structure.Structure("renumbered")
    new_model = Model.Model(0)
    new_structure.add(new_model)
    lengths = {}
    for chain_id in chain_ids:
        source = model[chain_id]
        target = Chain.Chain(chain_id)
        new_model.add(target)
        count = 0
        for residue in source:
            if residue.id[0] != " ":
                continue
            aa = standard_aa(residue)
            if aa is None:
                continue
            count += 1
            resname = "MET" if residue.resname.upper() == "MSE" else residue.resname
            new_residue = Residue.Residue((" ", count, " "), resname, residue.segid)
            for atom in residue:
                if atom.element == "H":
                    continue
                new_residue.add(Atom.Atom(
                    atom.name, atom.coord, atom.bfactor, atom.occupancy,
                    atom.altloc, atom.fullname, atom.serial_number, atom.element,
                ))
            target.add(new_residue)
        lengths[chain_id] = count
    path.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(new_structure)
    io.save(str(path))
    return lengths


def run_mpnn(name: str, pdb_path: Path, scored_chain: str,
             fixed: dict[str, list[int]] | None) -> dict[int, np.ndarray]:
    """Run ProteinMPNN conditional probabilities and return seq-index -> logp row."""
    out_dir = WORK / name
    result_path = out_dir / "conditional_probs_only" / f"{name}.npz"
    if result_path.exists():
        data = np.load(result_path)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(MPNN),
            "--pdb_path", pdb_path.as_posix(),
            "--pdb_path_chains", scored_chain,
            "--path_to_model_weights", WEIGHTS.as_posix(),
            "--model_name", "v_48_020",
            "--conditional_probs_only", "1",
            "--conditional_probs_only_backbone", "0",
            "--num_seq_per_target", "1",
            "--batch_size", "1",
            "--seed", "2309",
            "--suppress_print", "1",
            "--out_folder", out_dir.as_posix(),
        ]
        if fixed is not None:
            fixed_path = out_dir / "fixed_positions.jsonl"
            fixed_path.write_text(
                json.dumps({pdb_path.stem: fixed}) + "\n", encoding="ascii")
            command += ["--fixed_positions_jsonl", fixed_path.as_posix()]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
        if completed.returncode or not result_path.exists():
            raise RuntimeError(
                f"ProteinMPNN failed for {name}: {completed.stderr[-2000:]}"
            )
        data = np.load(result_path)
    log_p = data["log_p"][0]
    design_rows = np.where(data["design_mask"] > 0)[0]
    sequence = "".join(MPNN_ALPHABET[int(i)] for i in data["S"][design_rows])
    return {index + 1: log_p[row] for index, row in enumerate(design_rows)}, sequence


def parse_mutations(value: str):
    out = []
    for part in str(value).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        match = MUT_RE.match(part)
        if not match:
            return None
        out.append(match.groups())
    return out or None


def map_chain(letter: str, row) -> str | None:
    heavy = row["Hchain"]
    light = row["Lchain"]
    antigen = row["antigen_chain"]
    if letter == "H":
        return heavy
    if letter == "L":
        return light
    if antigen and letter == antigen:
        return antigen
    return letter


def partner_chain(chain: str, row) -> str | None:
    heavy = row["Hchain"]
    light = row["Lchain"]
    antigen = row["antigen_chain"]
    if chain == antigen:
        return heavy or light
    return antigen


def score_row(row, structure, maps, cache):
    mutations = parse_mutations(row["Mutation"])
    if mutations is None:
        return None, "mutation_parse"
    grouped: dict[str, list[tuple[int, str, str]]] = {}
    for letter, wt, position, insertion, mut in mutations:
        chain = map_chain(letter, row)
        if chain is None:
            return None, f"chain_unmapped:{letter}"
        if chain not in maps:
            return None, f"chain_missing:{chain}"
        key = (int(position), insertion.strip())
        if key not in maps[chain]:
            hits = [k for k in maps[chain] if k[0] == int(position)]
            if not hits:
                return None, f"residue_missing:{chain}:{position}"
            key = hits[0]
        index, observed = maps[chain][key]
        if observed != wt:
            return None, f"wt_mismatch:{chain}:{position}:{observed}!={wt}"
        grouped.setdefault(chain, []).append((index, wt, mut))

    pid = row["pid"]
    d_complex = d_stripped = dils = blosum = 0.0
    composition = np.zeros(20, dtype=float)
    for chain, items in grouped.items():
        partner = partner_chain(chain, row)
        if partner is None or partner == chain:
            return None, f"partner_missing:{chain}"
        if chain not in structure[0] or partner not in structure[0]:
            return None, f"pdbfile_chain_missing:{chain}:{partner}"
        complex_logp, complex_seq = get_logp(
            pid, chain, partner, "complex", structure, cache)
        stripped_logp, stripped_seq = get_logp(
            pid, chain, partner, "stripped", structure, cache)
        for index, wt, mut in items:
            if index not in complex_logp or index not in stripped_logp:
                return None, f"logp_missing:{chain}:{index}"
            wi, mi = ALPHABET.index(wt), ALPHABET.index(mut)
            dc = float(complex_logp[index][mi] - complex_logp[index][wi])
            ds = float(stripped_logp[index][mi] - stripped_logp[index][wi])
            d_complex += dc
            d_stripped += ds
            dils += dc - ds
            composition[wi] -= 1.0
            composition[mi] += 1.0
            blosum += float(BLOSUM[wt, mut])
    return {
        "dlogp_complex": d_complex,
        "dlogp_stripped": d_stripped,
        "dils": dils,
        "blosum62": blosum,
        "n_mut": len(mutations),
        "n_chains": len(grouped),
        **{f"comp_{i}": composition[i] for i in range(20)},
    }, None


def get_logp(pid, chain, partner, state, structure, cache):
    key = (pid, chain, partner, state)
    if key in cache:
        return cache[key]
    name = f"{pid}_{chain}_{partner}_{state}"
    if state == "complex":
        pdb_path = WORK / name / f"{name}.pdb"
        lengths = write_renumbered(structure, [chain, partner], pdb_path)
        fixed = {chain: [], partner: list(range(1, lengths[partner] + 1))}
    else:
        pdb_path = WORK / name / f"{name}.pdb"
        lengths = write_renumbered(structure, [chain], pdb_path)
        fixed = {chain: []}
    logp, sequence = run_mpnn(name, pdb_path, chain, fixed)
    expected = "".join(aa for _, aa in chain_residues(structure, chain))
    if sequence != expected:
        raise RuntimeError(
            f"sequence mismatch {name}: expected {expected[:20]} got {sequence[:20]}"
        )
    cache[key] = (logp, sequence)
    return cache[key]


def composition_baseline(scores: pd.DataFrame) -> pd.Series:
    comp_cols = [f"comp_{i}" for i in range(20)]
    prediction = pd.Series(np.nan, index=scores.index, dtype=float)
    ok = scores["status"].eq("ok")
    for pid in sorted(scores.loc[ok, "pid"].unique()):
        test = scores.index[(scores["pid"] == pid) & ok]
        train = scores.index[(scores["pid"] != pid) & ok]
        if len(train) < 20 or len(test) == 0:
            continue
        model = Ridge(alpha=1.0)
        model.fit(scores.loc[train, comp_cols].to_numpy(), scores.loc[train, "ddG"].to_numpy())
        prediction.loc[test] = model.predict(scores.loc[test, comp_cols].to_numpy())
    return prediction


def bootstrap_mean(values: np.ndarray, seed: int, trials: int = 10000):
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return [None, None]
    draws = rng.choice(values, size=(trials, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def analyze(scores: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for pid, group in scores[scores["status"].eq("ok")].groupby("pid"):
        if len(group) < 5 or group["ddG"].nunique() < 3:
            continue
        row = {"pid": pid, "n_rows": len(group)}
        for score in ("dils", "dlogp_complex", "dlogp_stripped", "blosum62", "composition_pred"):
            values = group[score]
            valid = values.notna()
            if valid.sum() < 5 or group.loc[valid, "ddG"].nunique() < 3:
                row[f"rho_{score}"] = np.nan
            else:
                rho, _ = spearmanr(values[valid], group.loc[valid, "ddG"])
                row[f"rho_{score}"] = float(rho)
        rows.append(row)
    per_complex = pd.DataFrame(rows)
    summary: dict = {"n_complexes": int(len(per_complex)),
                     "n_rows": int(scores["status"].eq("ok").sum())}
    for score in ("dils", "dlogp_complex", "dlogp_stripped", "blosum62", "composition_pred"):
        col = f"rho_{score}"
        if col not in per_complex.columns:
            summary[score] = None
            continue
        values = per_complex[col].dropna().to_numpy()
        if len(values) == 0:
            summary[score] = None
            continue
        summary[score] = {
            "n": int(len(values)),
            "median_rho": float(np.median(values)),
            "mean_rho": float(np.mean(values)),
            "fraction_negative": float(np.mean(values < 0)),
            "mean_ci95": bootstrap_mean(values, 20261111),
            "median_ci95": bootstrap_mean(np.array(
                [np.median(np.random.default_rng(20261111 + i).choice(values, len(values)))
                 for i in range(2000)]), 20261111) if len(values) > 1 else None,
        }
    # paired dils vs composition
    paired = per_complex[["rho_dils", "rho_composition_pred"]].dropna()
    if len(paired):
        delta = (paired["rho_dils"] - paired["rho_composition_pred"]).to_numpy()
        summary["dils_minus_composition"] = {
            "n": int(len(delta)),
            "mean_delta": float(np.mean(delta)),
            "median_delta": float(np.median(delta)),
            "fraction_dils_better": float(np.mean(delta < 0)),
            "mean_ci95": bootstrap_mean(delta, 20261112),
        }
    summary["gate"] = (
        summary.get("dils", {}).get("fraction_negative", 0) >= 0.7
        and summary.get("dils_minus_composition", {}).get("mean_ci95", [None, None])[1] is not None
        and summary["dils_minus_composition"]["mean_ci95"][1] < 0
    )
    return per_complex, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="limit complexes for smoke")
    parser.add_argument("--complexes", default="", help="comma-separated PDB IDs")
    args = parser.parse_args()
    ensure_dirs()

    _, data = load_ab_bind()
    if args.complexes:
        keep = {x.strip().lower() for x in args.complexes.split(",") if x.strip()}
        data = data[data["pid"].isin(keep)]
    pids = sorted(data["pid"].unique())
    if args.limit:
        pids = pids[: args.limit]
    data = data[data["pid"].isin(pids)]
    log(f"complexes={len(pids)} rows={len(data)}")

    cache: dict = {}
    records = []
    for pid in pids:
        try:
            structure = parse_structure(pid)
        except Exception as exc:  # noqa: BLE001
            log(f"structure failed {pid}: {exc}")
            continue
        subset = data[data["pid"] == pid]
        first = subset.iloc[0]
        chains = [chain.id for chain in next(structure.get_models())]
        maps = {chain: residue_map(structure, chain) for chain in chains}
        for _, row in subset.iterrows():
            try:
                result, error = score_row(row, structure, maps, cache)
            except Exception as exc:  # noqa: BLE001
                result, error = None, f"exception:{type(exc).__name__}:{exc}"
            if result is None:
                records.append({
                    "pid": pid, "mutation": row["Mutation"], "ddG": row["ddG(kcal/mol)"],
                    "status": "skipped", "reason": error,
                })
            else:
                records.append({
                    "pid": pid, "mutation": row["Mutation"], "ddG": row["ddG(kcal/mol)"],
                    "status": "ok", "reason": "", **result,
                })
        log(f"{pid}: {len(subset)} rows, cache={len(cache)}")

    scores = pd.DataFrame(records)
    scores["ddG"] = pd.to_numeric(scores["ddG"], errors="coerce")
    scores["composition_pred"] = composition_baseline(scores)
    scores.to_csv(OUT / "scores.csv", index=False)
    per_complex, summary = analyze(scores)
    per_complex.to_csv(OUT / "per_complex.csv", index=False)
    with (OUT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    report = ["# AB-bind mutation-effect benchmark v1", ""]
    report.append(f"Rows scored: {summary['n_rows']} / {len(scores)}")
    report.append(f"Complexes analyzed: {summary['n_complexes']}")
    report.append("")
    report.append("| score | n | median rho | mean rho | fraction rho<0 | mean CI95 |")
    report.append("|---|---:|---:|---:|---:|---|")
    for score in ("dils", "dlogp_complex", "dlogp_stripped", "blosum62", "composition_pred"):
        item = summary.get(score)
        if not item:
            continue
        report.append(
            f"| {score} | {item['n']} | {item['median_rho']:.3f} | "
            f"{item['mean_rho']:.3f} | {item['fraction_negative']:.2f} | "
            f"[{item['mean_ci95'][0]:.3f}, {item['mean_ci95'][1]:.3f}] |"
        )
    report.append("")
    if "dils_minus_composition" in summary:
        item = summary["dils_minus_composition"]
        report.append(
            f"dils - composition: mean {item['mean_delta']:.3f}, "
            f"median {item['median_delta']:.3f}, "
            f"fraction dils better {item['fraction_dils_better']:.2f}, "
            f"CI95 [{item['mean_ci95'][0]:.3f}, {item['mean_ci95'][1]:.3f}]"
        )
    report.append("")
    report.append(f"Primary gate: **{'PASS' if summary['gate'] else 'FAIL'}**")
    report.append("")
    report.append("Negative rho is the good direction: positive ddG means destabilizing, "
                  "so a good score should decrease as ddG increases.")
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    log(f"done: {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
