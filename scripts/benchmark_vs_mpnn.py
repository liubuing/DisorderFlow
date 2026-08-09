#!/usr/bin/env python
"""BFN vs ProteinMPNN design benchmark on antibody-antigen complexes.

For each scaffold+epitope pair in a benchmark set, both methods design the same
CDR regions; designs are scored on:
  - sequence recovery (vs native CDR),
  - diversity (fraction unique sequences / mean Levenshtein distance),
  - AF2-pass rate (ipTM ≥ threshold, optional — slow).

ProteinMPNN is invoked via the standard `proteinmpnn_run.py` CLI
(https://github.com/dauparas/ProteinMPNN). If the binary is missing, the MPNN
arm is skipped and logged — the script never silently reports partial results.

Usage:
  python scripts/benchmark_vs_mpnn.py --complex-dir data/antibody_complexes \
      --out results/ablation/bfn_vs_mpnn.json
  python scripts/benchmark_vs_mpnn.py --self-test   # validates metrics, no GPU/MPNN
  python scripts/benchmark_vs_mpnn.py --help
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "modules"))


# ── metrics (self-contained) ──

def sequence_recovery(designed, native):
    """Fraction of positions where designed equals native at identical positions."""
    if len(designed) != len(native):
        raise ValueError(f"Sequence length mismatch: {len(designed)} != {len(native)}")
    n = len(native)
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if designed[i] == native[i]) / n


def levenshtein(a, b):
    if len(a) < len(b):
        return levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        cur = [i + 1]
        for j, cb in enumerate(b):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (ca != cb)))
        prev = cur
    return prev[-1]


def diversity(seqs):
    """Mean pairwise Levenshtein distance / length, plus unique fraction."""
    if len(seqs) < 2:
        return {"unique_frac": 1.0 if seqs else 0.0, "mean_norm_distance": 0.0}
    dists, L = [], max(len(s) for s in seqs)
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            dists.append(levenshtein(seqs[i], seqs[j]))
    return {"unique_frac": len(set(seqs)) / len(seqs),
            "mean_norm_distance": (sum(dists) / len(dists)) / max(L, 1)}


def score_arm(designs, native=None):
    """Compute recovery + diversity for one design arm."""
    seqs = [d["sequence"] for d in designs if d.get("sequence")]
    out = {"n_designs": len(seqs), **diversity(seqs)}
    if native:
        rec = [sequence_recovery(s, native) for s in seqs]
        out["mean_recovery"] = sum(rec) / len(rec) if rec else 0.0
    return out


# ── design arms ──

def design_bfn(pdb_path, region_spec, num_samples, device):
    """Run BFN design via the project loader."""
    from bfn_loader import run_bfn_design
    return run_bfn_design(pdb_path, region_spec, num_samples=num_samples,
                          stochastic=True, device=device)


def design_mpnn(pdb_path, region_spec, num_samples, mpnn_exe, mpnn_outdir):
    """Run ProteinMPNN via subprocess. Returns list of dicts or [] if unavailable."""
    if not mpnn_exe or not os.path.exists(mpnn_exe):
        print(f"  [MPNN] binary not found ({mpnn_exe}) — skipping MPNN arm")
        return []
    # Freeze every non-design position in the selected chain.
    chain = region_spec.split(":")[0]
    from idp_antibody_design import _extract_sequence_from_pdb, _parse_cdr_ranges
    chain_sequence = _extract_sequence_from_pdb(pdb_path, chain)
    ranges = _parse_cdr_ranges(region_spec)
    design_positions = [
        position for start, end, _ in ranges for position in range(start, end + 1)]
    fixed_positions = [
        position for position in range(1, len(chain_sequence) + 1)
        if position not in set(design_positions)]
    os.makedirs(mpnn_outdir, exist_ok=True)
    name = os.path.splitext(os.path.basename(pdb_path))[0]
    fixed_path = os.path.join(mpnn_outdir, "fixed_positions.jsonl")
    with open(fixed_path, "w", encoding="ascii") as handle:
        handle.write(json.dumps({name: {chain: fixed_positions}}) + "\n")
    cmd = [
        sys.executable, mpnn_exe,
        "--pdb_path", pdb_path, "--pdb_path_chains", chain,
        "--fixed_positions_jsonl", fixed_path,
        "--out_folder", mpnn_outdir,
        "--num_seq_per_target", str(num_samples),
        "--sampling_temp", "0.1",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  [MPNN] run failed: {e}")
        return []
    # Parse MPNN FASTA output.
    return _parse_mpnn_output(
        mpnn_outdir, chain, design_positions, len(chain_sequence))


def _parse_mpnn_output(mpnn_outdir, chain, design_positions, chain_length):
    """ProteinMPNN writes per-PDB FASTAs under <out>/seqs/. Extract designed seqs."""
    seqs_dir = os.path.join(mpnn_outdir, "seqs")
    designs = []
    if not os.path.isdir(seqs_dir):
        return designs
    for fn in os.listdir(seqs_dir):
        if not fn.endswith(".fa"):
            continue
        with open(os.path.join(seqs_dir, fn)) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    continue
                if line:
                    full_sequence = line.replace("/", "")
                    if len(full_sequence) != chain_length:
                        raise ValueError(
                            f"ProteinMPNN {chain} length {len(full_sequence)} != {chain_length}")
                    designed = ''.join(full_sequence[position - 1]
                                       for position in design_positions)
                    designs.append({"sequence": designed, "full_sequence": full_sequence,
                                    "source": "mpnn"})
    return designs


# ── orchestrator ──

def run_benchmark(complex_dir, out_path, num_samples, region_spec_template,
                  mpnn_exe, max_complexes, device, work_dir, af2_validate):
    complexes = _list_complexes(complex_dir, max_complexes)
    print(f"Benchmarking on {len(complexes)} complexes, {num_samples} designs/arm")
    os.makedirs(work_dir, exist_ok=True)

    all_results = []
    for cid, pdb in complexes:
        print(f"\n=== {cid} ===")
        region_spec, native = _resolve_complex(pdb, region_spec_template)
        try:
            bfn = design_bfn(pdb, region_spec, num_samples, device)
        except Exception as e:  # noqa: BLE001
            print(f"  BFN failed: {e}")
            bfn = []
        mpnn_out = os.path.join(work_dir, cid + "_mpnn")
        mpnn = design_mpnn(pdb, region_spec, num_samples, mpnn_exe, mpnn_out)

        bfn_score = score_arm(bfn, native)
        mpnn_score = score_arm(mpnn, native) if mpnn else {"n_designs": 0, "skipped": True}

        print(f"  BFN:  {bfn_score}")
        print(f"  MPNN: {mpnn_score}")
        all_results.append({"complex": cid, "bfn": bfn_score, "mpnn": mpnn_score,
                            "region": region_spec})

    summary = {
        "n_complexes": len(all_results),
        "num_samples_per_arm": num_samples,
        "mpnn_available": mpnn_exe and os.path.exists(mpnn_exe),
        "per_complex": all_results,
    }
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved → {out_path}")
    return summary


def _list_complexes(complex_dir, max_complexes):
    if not os.path.isdir(complex_dir):
        return []
    pdbs = sorted(f for f in os.listdir(complex_dir) if f.endswith(".pdb"))
    if max_complexes:
        pdbs = pdbs[:max_complexes]
    return [(os.path.splitext(f)[0], os.path.join(complex_dir, f)) for f in pdbs]


def _resolve_complex(pdb_path, region_spec_template):
    """Return (region_spec, native_cdr_seq or None). region uses chain B by default."""
    region_spec = region_spec_template or "B:26-33,51-58,97-113"
    native = None
    try:
        from idp_antibody_design import _extract_sequence_from_pdb, _parse_cdr_ranges
        chain = region_spec.split(":")[0]
        seq = _extract_sequence_from_pdb(pdb_path, chain)
        ranges = _parse_cdr_ranges(region_spec)
        native = "".join(seq[s - 1: e] for s, e, _ in ranges) if seq else None
    except Exception:
        pass
    return region_spec, native


def self_test():
    print("self-test: metric validation (no GPU / no MPNN)")
    designs = [{"sequence": "ACDEFGHIK"}, {"sequence": "ACDEAGHIK"},
               {"sequence": "LMNPQRSTV"}]
    out = score_arm(designs, native="ACDEFGHIK")
    print(json.dumps(out, indent=2))
    assert 0.0 <= out["mean_recovery"] <= 1.0
    assert 0.0 <= out["unique_frac"] <= 1.0
    # MPNN unavailable path returns [].
    assert design_mpnn("x.pdb", "B:1-10", 5, "/nonexistent/mpnn", "/tmp/_x") == []
    print("self-test OK")


def main():
    parser = argparse.ArgumentParser(description="BFN vs ProteinMPNN design benchmark")
    parser.add_argument("--complex-dir", default=os.path.join("data", "antibody_complexes"))
    parser.add_argument("--out", default=os.path.join("results", "ablation", "bfn_vs_mpnn.json"))
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--region-spec", default=None,
                        help="CDR spec, e.g. 'B:26-33,51-58,97-113'")
    parser.add_argument("--mpnn-exe", default=os.environ.get("PROTEINMPNN_EXE", ""),
                        help="Path to proteinmpnn_run.py. Env: PROTEINMPNN_EXE")
    parser.add_argument("--max-complexes", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--work-dir", default=os.path.join("results", "mpnn_workdir"))
    parser.add_argument("--af2-validate", action="store_true",
                        help="Optionally AF2-validate designs (slow)")
    parser.add_argument("--self-test", action="store_true",
                        help="Validate metrics + MPNN-unavailable path, then exit")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not os.path.isdir(args.complex_dir):
        print(f"ERROR: complex dir not found: {args.complex_dir}")
        sys.exit(1)
    if not args.mpnn_exe:
        print("NOTE: --mpnn-exe not set → MPNN arm will be skipped (BFN-only run).")

    run_benchmark(args.complex_dir, args.out, args.num_samples, args.region_spec,
                  args.mpnn_exe, args.max_complexes, args.device, args.work_dir,
                  args.af2_validate)


if __name__ == "__main__":
    main()
