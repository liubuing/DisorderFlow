# HF aux benchmark v1

External-data dry-lab campaign run on 2026-11-11. Full results and caveats:
`docs/HF_AUX_BENCHMARK_20261111.md`.

| Script | Question | Output |
|---|---|---|
| `abbind_v1.py` | Does ProteinMPNN mutation likelihood rank AB-bind ΔΔG within a complex, beyond composition? | `results/hf_aux_benchmark_v1/abbind/` |
| `alphaseq_seq_v1.py` | How much AlphaSeq affinity is predictable from sequence (composition vs ESM2)? | `results/hf_aux_benchmark_v1/alphaseq/{per_target,summary}` |
| `alphaseq_group_v1.py` | Does the ESM2 signal survive KMeans-on-embedding group splits? | `results/hf_aux_benchmark_v1/alphaseq/{group_split,group_split_summary}` |
| `sabdab2_epi_pcls_v1.py` | Does antibody context favor native peptide over composition-matched shuffles (peptide-side PCLS)? | `results/hf_aux_benchmark_v1/sabdab2_epi/pcls_*` |
| `alphaseq_mmseqs_v1.py` | Does the ESM2 signal survive MMseqs2 90% identity cluster splits? | `results/hf_aux_benchmark_v1/alphaseq/mmseqs_split*` |
| `sabdab2_epi_h3ecls_v1.py` | H3 ECLS on SAbDab2-Epi val/test (ANARCI/IMGT or reference fallback) | `results/hf_aux_benchmark_v1/sabdab2_epi/h3ecls_*` |
| `anarci_h3_wsl.py` | ANARCI/IMGT H3 numbering in WSL (HMMER 3.4 deb) | `results/hf_aux_benchmark_v1/sabdab2_epi/anarci_h3.json` |
| `h3ecls_analyze_v1.py` | H3-ECLS strata, cluster bootstrap, negatives | `results/hf_aux_benchmark_v1/sabdab2_epi/h3ecls_full_analysis.*` |
| `anarci_h3_public_wsl.py` | ANARCI for the public peptide-Kd cohort | `results/hf_aux_benchmark_v1/public_peptide_kd/anarci_h3.json` |
| `public_peptide_kd_h3ecls_v1.py` | H3-ECLS vs wet-lab Kd on 111 public antibody-peptide complexes | `results/hf_aux_benchmark_v1/public_peptide_kd/h3ecls_kd*` |
| `abcdr_d44_v1.py` | D44 2,048 single mutations vs Binder (1MLC parent) | `results/hf_aux_benchmark_v1/abcdr_binding/d44/` |
| `abbind_single_antibody_v1.py` | Single-mutation, antibody-only AB-bind replication and hash-half split | `results/hf_aux_benchmark_v1/abbind/single_antibody_*` |

Wet-lab handoff: `experiments/wetlab_bli_v1/BLI_EXECUTION_PLAN.md` (dry-lab package only;
no BLI/SPR was executed).

## Status

- AB-bind: development gate **PASS**; 591/683 mutations, 18 complexes; `dils` mean rho −0.167, CI [−0.296, −0.045], 76.5% negative; permutation p=0.0004.
- AlphaSeq: 8 datasets / 19 targets; ESM2 mean 0.642 vs composition 0.279; group split ESM2 0.714.
- SAbDab2-Epi: 420 eligible val/test peptide complexes, 261 PDB, 213 clusters; PCLS pilot 25/29 positive, mean advantage 0.546.
- SAbDab2-Epi H3-ECLS: 408 rows / 254 PDB / 157 CDRH3 clusters with ANARCI/IMGT H3; 400/408 positive (98.0%), mean 0.672; cluster-level 98.1%, CI [0.606, 0.703]; val 98.3%, test 97.5%. 12 rows skipped for size/timeout.
- Public wet-lab antibody-peptide Kd: 111 complexes; advantage vs logKd rho **−0.312**, p=0.00085, CI [−0.476, −0.128]; H3≥14 −0.456; antigen-level −0.304, p=0.002. SPR-only n=21 rho −0.371, p=0.097. Exploratory association, not a binding predictor.
- AbCDR-Binding D44: 2,048 single mutations; `dils` overall rho −0.012 (ns), interface (707) +0.119 (p=0.0015), non-interface −0.100; BLOSUM +0.374. `dils` is not a general mutation-effect ranker; it has only a small interface-local signal.
- Cross-complex learned ranker: **FAIL** (direction reverses); not used.
- No frozen ECLS/PAE result changed. No binding/affinity/IDP claim.

## Reproduction

```bash
python experiments/hf_aux_benchmark_v1/abbind_v1.py
python experiments/hf_aux_benchmark_v1/alphaseq_seq_v1.py --max-per-target 20000
python experiments/hf_aux_benchmark_v1/alphaseq_group_v1.py
python experiments/hf_aux_benchmark_v1/sabdab2_epi_pcls_v1.py --limit 30 --shuffles 100
```

Inputs are downloaded on demand from HuggingFace/RCSB into
`data/hf_aux_benchmark_v1/` (AB-bind parquet is cached at
`.tools/hf_probe/abbind.parquet`). ProteinMPNN weights:
`ProteinMPNN/vanilla_model_weights/v_48_020.pt`.

## Important caveats

- AB-bind is antibody–protein, not IDP/peptide.
- AlphaSeq random 5-fold is optimistic; KMeans is not MMseqs2 identity clustering.
- PCLS is peptide-side, not the frozen H3-side ECLS.
- No wet-lab measurement exists for any project candidate.
