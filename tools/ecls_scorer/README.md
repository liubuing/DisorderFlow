# ecls-scorer

Standalone command-line tool for **ECLS** (epitope-conditioned likelihood shift)
scoring of antibody CDR-H3 sequences against an antibody–peptide complex
structure:

```text
ECLS(s) = H3 NLL(complex backbone) − H3 NLL(peptide-stripped Fab backbone)
advantage = mean ECLS(composition-matched shuffled H3) − ECLS(native H3)
```

Per-residue conditional log-likelihoods are produced by ProteinMPNN
(`v_48_020`, conditional-probability mode) with everything outside H3 held
fixed. The peptide-stripped reference keeps the Fab coordinates bit-identical
and removes only the peptide chain. Positive ECLS means the peptide coordinate
context favours the sequence.

## Scope and honest boundary

- This tool implements the **published ECLS definition** on user-supplied
  structures. The per-sequence NLL and shuffle arithmetic is ported verbatim
  from the canonical scripts (`scripts/benchmark_h3_epitope_delta.py`,
  `scripts/benchmark_h3_scorer_sanity.py` in the DisorderFlow repository).
- It does **not** reproduce the frozen ECLS publication dataset end-to-end:
  that dataset was preprocessed into LMDB records, while this tool reads PDB
  files directly. Numbers are comparable at the definition level, not as a
  dataset rerun.
- The published temporal-final evaluation is **terminal** (`rerun_permitted:
  false` in `publication/ECLS_SCOPE_FREEZE.yml`). Do not use this tool to
  rerun it.
- ECLS is a structural sequence-likelihood quantity. It is **not** a binding
  energy, affinity prediction, or candidate-ranking guarantee; composition-
  matched shuffles are computational counterfactuals, not measured non-binders.

## Installation

From the repository root, with a Python ≥3.10 environment that has PyTorch:

```bash
pip install ./tools/ecls_scorer
```

The package itself needs only `numpy` and `biopython`; ProteinMPNN runs as a
subprocess and needs `torch` (install `ecls-scorer[mpnn]` or use your system
torch). The vendored `ProteinMPNN/` directory (script + `vanilla_model_weights/`)
must be available on disk; the tool does not redistribute the weights.

## Usage

```bash
ecls-score \
  --pdb data/anti_abeta_refs/4HIX.pdb \
  --heavy-chain H --light-chain L --peptide-chain A \
  --h3-sequence VRYDHYSGSSDY \
  --shuffles 200 --seed 2309 \
  --mpnn-script ProteinMPNN/protein_mpnn_run.py \
  --mpnn-weights ProteinMPNN/vanilla_model_weights \
  --output result.json
```

- The H3 sequence is located by a unique contiguous match in the standardized
  heavy chain; pass `--h3-start N` to disambiguate.
- `--sequences-file extra.fa` scores additional candidate sequences (each gets
  its own ECLS; the advantage summary uses only shuffles of the native H3).
- `ECLS_MPNN_SCRIPT` / `ECLS_MPNN_WEIGHTS` environment variables replace the
  explicit flags (the Docker image sets them).
- Output JSON contains per-sequence `complex_h3_nll`, `apo_h3_nll`, `ecls`,
  the advantage summary, and full provenance (commands, weights SHA-256, seed).

### Docker

Build from the repository root:

```bash
docker build -f tools/ecls_scorer/Dockerfile -t ecls-scorer .
docker run --rm -v "$PWD:/data" ecls-score \
  --pdb /data/data/anti_abeta_refs/4HIX.pdb \
  --heavy-chain H --light-chain L --peptide-chain A \
  --h3-sequence VRYDHYSGSSDY --shuffles 200 \
  --output /data/result.json
```

## Tests

Pure-math tests run anywhere; the smoke test needs torch and the vendored
ProteinMPNN assets (skipped automatically otherwise):

```bash
pytest tools/ecls_scorer/tests
```

## Citation

If you use ECLS or this tool, please cite the manuscript "Peptide-context
scoring of antibody loops: native recognition is not candidate ranking"
(DisorderFlow repository, `publication/MANUSCRIPT_DRAFT.md`) and ProteinMPNN
(Dauparas et al., *Science* 2022).

## License

Tool code: MIT. ProteinMPNN code and model weights follow the licenses in
`ProteinMPNN/LICENSE`; this tool does not redistribute them outside the
repository that already vendored them.
