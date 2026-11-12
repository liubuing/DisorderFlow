"""Command-line entry point for the standalone ECLS scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from .backbone import locate_h3_indices, write_standardized_backbone
from .mpnn import score_backbone
from .scoring import MPNN_ALPHABET, advantage_summary, composition_shuffles, ecls_delta, sequence_nll

DEFINITION = "ECLS(s) = H3 NLL(complex backbone) - H3 NLL(peptide-stripped Fab backbone)"
BOUNDARY = (
    "This tool implements the ECLS definition on user-supplied structures. It does not "
    "reproduce the frozen DisorderFlow ECLS publication dataset end-to-end (that dataset "
    "used LMDB-preprocessed complexes), and the published temporal-final evaluation is "
    "terminal: rerun_permitted=false. Scores are structural likelihood quantities, not "
    "binding energies; composition-matched shuffles are computational counterfactuals, "
    "not experimental non-binders.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_fasta_sequences(path: Path):
    """(name, sequence) pairs from a plain FASTA file."""
    records = []
    name, chunks = None, []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(chunks)))
                name, chunks = line[1:].split()[0], []
            elif line:
                chunks.append(line)
    if name is not None:
        records.append((name, "".join(chunks)))
    return records


def validate_sequence(sequence: str, label: str) -> str:
    sequence = sequence.strip().upper()
    if not sequence:
        raise ValueError(f"{label}: empty sequence")
    unknown = sorted(set(sequence) - set(MPNN_ALPHABET))
    if unknown:
        raise ValueError(f"{label}: characters outside ProteinMPNN alphabet: {unknown}")
    return sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ecls-score",
        description=(
            "Score CDR-H3 sequences with ECLS: ProteinMPNN conditional NLL on an "
            "antibody-peptide complex backbone minus NLL on the peptide-stripped Fab."))
    parser.add_argument("--pdb", required=True, help="input complex PDB (any chain ids)")
    parser.add_argument("--heavy-chain", default="H")
    parser.add_argument("--light-chain", default=None)
    parser.add_argument("--peptide-chain", required=True)
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--h3-sequence", required=True,
                        help="native CDR-H3 sequence to locate and score")
    parser.add_argument("--h3-start", type=int, default=None,
                        help="1-based standardized resseq; inferred from a unique match otherwise")
    parser.add_argument("--sequences-file", default=None,
                        help="FASTA of extra candidate sequences to score (not used for advantage)")
    parser.add_argument("--shuffles", type=int, default=0,
                        help="composition-matched shuffles of --h3-sequence (enables the advantage summary)")
    parser.add_argument("--seed", type=int, default=2309,
                        help="MPNN and shuffle seed (frozen publication runs derived the shuffle seed "
                             "from seed+record-id; the standalone default is --seed directly)")
    parser.add_argument("--mpnn-script", default=os.environ.get("ECLS_MPNN_SCRIPT"),
                        help="path to protein_mpnn_run.py (env ECLS_MPNN_SCRIPT)")
    parser.add_argument("--mpnn-weights", default=os.environ.get("ECLS_MPNN_WEIGHTS"),
                        help="path to the vanilla_model_weights directory (env ECLS_MPNN_WEIGHTS)")
    parser.add_argument("--model-name", default="v_48_020")
    parser.add_argument("--work-dir", default=None, help="scratch directory (default: <output dir>/work)")
    parser.add_argument("--output", default=None, help="output JSON path (default: stdout)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.mpnn_script is None or args.mpnn_weights is None:
        raise SystemExit(
            "ProteinMPNN location required: pass --mpnn-script/--mpnn-weights "
            "or set ECLS_MPNN_SCRIPT/ECLS_MPNN_WEIGHTS")
    mpnn_script = Path(args.mpnn_script)
    weights_dir = Path(args.mpnn_weights)
    if not mpnn_script.exists():
        raise SystemExit(f"protein_mpnn_run.py not found: {mpnn_script}")
    if not weights_dir.exists():
        raise SystemExit(f"model weights directory not found: {weights_dir}")

    pdb_path = Path(args.pdb).resolve()
    if not pdb_path.exists():
        raise SystemExit(f"input PDB not found: {pdb_path}")
    heavy = args.heavy_chain
    peptide = args.peptide_chain
    light = args.light_chain
    if len({heavy, peptide, *( [light] if light else [])}) != \
            len([heavy, peptide, *([light] if light else [])]):
        raise SystemExit("chain roles must be distinct")

    output_path = Path(args.output).resolve() if args.output else None
    work_root = (Path(args.work_dir).resolve() if args.work_dir
                 else (output_path.parent if output_path else Path.cwd()) / "work")
    complex_dir = work_root / "complex"
    apo_dir = work_root / "apo"

    retained = [chain for chain in (heavy, light) if chain]
    complex_manifest = write_standardized_backbone(
        str(pdb_path), str(complex_dir / "complex.pdb"),
        retained + [peptide], model_index=args.model_index)
    apo_manifest = write_standardized_backbone(
        str(pdb_path), str(apo_dir / "apo.pdb"), retained, model_index=args.model_index)
    if complex_manifest["chains"][heavy]["sequence"] != apo_manifest["chains"][heavy]["sequence"]:
        raise SystemExit("heavy-chain standardization differed between complex and apo states")

    heavy_sequence = complex_manifest["chains"][heavy]["sequence"]
    h3_sequence = validate_sequence(args.h3_sequence, "h3-sequence")
    h3_indices = locate_h3_indices(heavy_sequence, h3_sequence, args.h3_start)
    if heavy_sequence[h3_indices[0]:h3_indices[-1] + 1] != h3_sequence:
        raise SystemExit("internal H3 index mismatch")

    complex_logp, complex_command = score_backbone(
        str(mpnn_script), str(weights_dir), args.model_name, args.seed,
        complex_dir / "complex.pdb", heavy, h3_indices, complex_dir)
    apo_logp, apo_command = score_backbone(
        str(mpnn_script), str(weights_dir), args.model_name, args.seed,
        apo_dir / "apo.pdb", heavy, h3_indices, apo_dir)

    sequences = [("native", h3_sequence)]
    if args.sequences_file:
        sequences.extend(
            (name, validate_sequence(sequence, name))
            for name, sequence in read_fasta_sequences(Path(args.sequences_file)))
    if args.shuffles > 0:
        sequences.extend(
            (f"shuffle_{index + 1:03d}", shuffle)
            for index, shuffle in enumerate(
                composition_shuffles(h3_sequence, args.shuffles, args.seed)))

    rows = []
    for name, sequence in sequences:
        complex_nll = sequence_nll(complex_logp, sequence, h3_indices)
        apo_nll = sequence_nll(apo_logp, sequence, h3_indices)
        rows.append({
            "name": name,
            "sequence": sequence,
            "complex_h3_nll": round(complex_nll, 6),
            "apo_h3_nll": round(apo_nll, 6),
            "ecls": round(ecls_delta(complex_nll, apo_nll), 6),
        })

    summary = None
    shuffle_rows = [row for row in rows if row["name"].startswith("shuffle_")]
    if shuffle_rows:
        native_row = rows[0]
        summary = {
            "native": native_row,
            **advantage_summary(
                native_row["ecls"], [row["ecls"] for row in shuffle_rows]),
        }

    weight_file = weights_dir / f"{args.model_name}.pt"
    result = {
        "schema_version": 1,
        "tool": {"name": "ecls-scorer", "version": "0.1.0"},
        "definition": DEFINITION,
        "boundary": BOUNDARY,
        "input": {
            "pdb": str(pdb_path),
            "pdb_sha256": sha256_file(pdb_path),
            "heavy_chain": heavy,
            "light_chain": light,
            "peptide_chain": peptide,
            "model_index": args.model_index,
        },
        "h3": {
            "sequence": h3_sequence,
            "indices_0based": h3_indices,
            "standardized_resseq": [index + 1 for index in h3_indices],
        },
        "sequences": rows,
        "advantage_summary": summary,
        "provenance": {
            "mpnn_script": str(mpnn_script.resolve()),
            "weights_dir": str(weights_dir.resolve()),
            "model_weights_sha256": sha256_file(weight_file) if weight_file.exists() else None,
            "model_name": args.model_name,
            "seed": args.seed,
            "complex_command": complex_command,
            "apo_command": apo_command,
            "heavy_chain_length": complex_manifest["chains"][heavy]["length"],
        },
    }
    payload = json.dumps(result, indent=2) + "\n"
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
        print(json.dumps({"output": str(output_path),
                          "sequences": len(rows),
                          "ecls_advantage": None if summary is None
                          else summary["ecls_advantage"]}))
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
