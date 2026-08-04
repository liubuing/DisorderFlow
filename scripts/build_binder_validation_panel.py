#!/usr/bin/env python3
"""Build a same-scaffold 3STB VHH validation panel for ColabFold."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ABETA42 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"
CDR_SLICES = ((25, 33), (50, 58), (96, 113))

PANEL = {
    "3STB_native": "VQLQESGGGLVQAGGSLRLSCAASGRTLSSYAMGWFRQAPGKEREFVAAINRSGSTFYADAVKGRFTISRDNAKNTVYLQMNSLKPEDTAAYYCAADRFSPVVPGPIPVNTVDSWGQGTQVTVSS",
    "3STB_V17i_best": "VQLQESGGGLVQAGGSLRLSCATTVTTYTVYAMGWFRQAPGKEREFVTTTTTTTYTFYADAVKGRFTISRDNAKNTVYLQMNSLKPEDTAAYYTTTTTTTTTVVVTTTTTTVDSWGQGTQVTVSS",
    "3STB_V5_1_best": "VQLQESGGGLVQAGGSLRLSCAASGAWVINSWLGWFRQAPGKEREFVAAIGRLMELYFADAVKGRFTISRDNAKNTVYLQMNSLKPEDTAAYYCAAFFSQPLDDVQRYDKKTISWGQGTQVTVSS",
    "3STB_4HIX_graft": "VQLQESGGGLVQAGGSLRLSCAGGTGSSYAYAMGWFRQAPGKEREFVVSGSGGSTTFYADAVKGRFTISRDNAKNTVYLQMNSLKPEDTAAYYTYAGDRYSGYPGPIPVNTVDSWGQGTQVTVSS",
    "3STB_5CSZ_graft": "VQLQESGGGLVQAGGSLRLSCAGGTGSSYAYAMGWFRQAPGKEREFVINSASGTRTFYADAVKGRFTISRDNAKNTVYLQMNSLKPEDTAAYYTARYCARGRGYVPIPVNTVDSWGQGTQVTVSS",
}


def composition_scramble(sequence: str, seed: int) -> str:
    positions = [index for start, end in CDR_SLICES for index in range(start, end)]
    native = [sequence[index] for index in positions]
    rng = random.Random(seed)
    scrambled = native[:]
    for _ in range(100):
        rng.shuffle(scrambled)
        if scrambled != native:
            break
    output = list(sequence)
    for index, amino_acid in zip(positions, scrambled):
        output[index] = amino_acid
    return "".join(output)


def write_fasta(path: Path, records: dict[str, str]) -> None:
    with path.open("w", encoding="ascii") as handle:
        for name, sequence in records.items():
            handle.write(f">{name}\n{sequence}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/binder_validation/3stb_panel_v1")
    parser.add_argument("--scramble-seed", type=int, default=4201)
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = dict(PANEL)
    panel["3STB_native_cdr_scrambled"] = composition_scramble(PANEL["3STB_native"], args.scramble_seed)
    lengths = {len(sequence) for sequence in panel.values()}
    if lengths != {125}:
        raise ValueError(f"Panel must keep the 125-residue 3STB scaffold, got lengths={sorted(lengths)}")

    write_fasta(output_dir / "monomer.fasta", panel)
    write_fasta(
        output_dir / "multimer_abeta42.fasta",
        {name: f"{sequence}:{ABETA42}" for name, sequence in panel.items()},
    )
    manifest = {
        "panel_version": "3stb_same_scaffold_v1",
        "comparison_scope": "VHH candidates only; native 4HIX/5CSZ Fab controls are a separate calibration panel",
        "target_sequence": ABETA42,
        "scramble_seed": args.scramble_seed,
        "cdr_slices_zero_based_half_open": list(CDR_SLICES),
        "records": [
            {
                "name": name,
                "vhh_sequence": sequence,
                "vhh_length": len(sequence),
                "multimer_chain_lengths": [len(sequence), len(ABETA42)],
            }
            for name, sequence in panel.items()
        ],
        "colabfold_contract": {
            "version": "1.6.1",
            "msa_mode": "mmseqs2_uniref_env",
            "pair_mode": "unpaired_paired",
            "templates": False,
            "num_recycle": 3,
            "num_seeds": 5,
            "random_seed": 0,
            "num_models": 1,
            "monomer_max_seq": 64,
            "monomer_max_extra_seq": 128,
            "multimer_max_seq": 16,
            "multimer_max_extra_seq": 32,
            "multimer_msa_note": "Reduced for an 8 GB GPU; all panel members use the same depth",
            "multimer_model_type": "alphafold2_multimer_v3",
            "monomer_model_type": "alphafold2_ptm",
        },
        "commands": {
            "monomer": "colabfold_batch monomer.fasta monomer_results --model-type alphafold2_ptm --num-models 1 --num-recycle 3 --num-seeds 5 --random-seed 0 --max-seq 64 --max-extra-seq 128 --rank ptm --disable-unified-memory",
            "multimer": "colabfold_batch multimer_abeta42.fasta multimer_results --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 3 --num-seeds 5 --random-seed 0 --max-seq 16 --max-extra-seq 32 --rank iptm --disable-unified-memory",
        },
        "gates": {
            "monomer_mean_plddt_min": 70.0,
            "multimer_mean_iptm_min": 0.25,
            "multimer_mean_interface_pae_max": 20.0,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
