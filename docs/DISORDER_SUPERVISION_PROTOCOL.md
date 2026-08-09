# Disorder Supervision Protocol

## Claim boundary

Legacy ROC-AUC values of 0.509 and 0.500 were random-level baselines. Frozen
CAID3 evaluation now establishes external discrimination on the exact AFDB-
mapped, training-homology-independent subset, but not superiority to
metapredict or probability calibration. The corrected production inference
path changed 8.33% of paired 4HIX residues when disorder conditioning was
enabled; this shows pathway response, not beneficial design direction.

## Evidence hierarchy

Residue labels use the following default confidence weights:

| Evidence | Weight | Role |
|---|---:|---|
| DisProt/MobiDB experimental structural state | 1.00 | Primary supervision |
| Missing-density or NMR-ensemble structural proxy | 0.65 | Secondary supervision |
| Multi-conformation RMSF proxy | 0.55 | Secondary supervision |
| External predictor | 0.25 | Auxiliary teacher only |
| Charge-hydropathy heuristic | 0.10 | Legacy weak baseline only |

Unannotated residues have `mask=false`; they are not converted to ordered
labels. Every record requires a homology `cluster_id`, source, confidence, and
per-residue mask.

## Build

Prepare DisProt labels while excluding every CAID2 target and its available
UniRef50 cluster, then map exact accession+sequence pairs to local structures:

```bash
python scripts/build/prepare_disprot_supervision.py \
  --lmdb data/confidence_idp_unified \
  --output-dir data/disorder_supervision/disprot_structures_v1

python scripts/build/build_disorder_supervision.py \
  --input data/disorder_supervision/disprot_structures_v1/train.jsonl \
  --split train --output data/disorder_supervision/train_v3.pkl

python scripts/build/build_disorder_supervision.py \
  --input data/disorder_supervision/disprot_structures_v1/dev.jsonl \
  --split dev --output data/disorder_supervision/dev_v3.pkl
```

The expanded 2026-08-05 AFDB build contains 828 train structures in 799
UniRef50 clusters and 97 development structures in 93 clusters. It provides
59,698 experimentally supervised residues and has zero train/development
cluster overlap. A further 76,164 high-confidence AFDB residues are used only
as low-weight ordered proxies (`pLDDT >= 0.90`, weight 0.25) to prevent the
experimentally annotated regions, which are almost entirely positive, from
admitting an all-disordered solution. Every retained AFDB structure has an
exact DisProt sequence match; 90 unavailable structures and 2 sequence
mismatches were rejected.

## Train

Use `--init`, not `--finetune`. The latter intentionally resets confidence and
disorder heads in the legacy training driver.

```bash
python train.py configs/train/bfn_disorder_v4_afdb_supervised.yml \
  --init logs/bfn_disorder_v2_fixed_2026_08_04__21_13_44/checkpoints/8000.pt \
  --device cuda
```

Run at least three seeds after expanding structure coverage. Select checkpoints
on the UniRef50-disjoint development set only.

The 300-step balanced v4 pilots were stable across seeds 2030, 2031, and 2032.
Pooled UniRef50-disjoint dev ROC-AUC was 0.8076, 0.8075, and 0.8115; MCC at the
fixed 0.5 monitoring threshold was 0.3776, 0.3715, and 0.3792. The corresponding
pooled CAID2 Disorder-NOX ROC-AUC values on the frozen 62-target AFDB subset were
0.6692, 0.6700, and 0.6725. Checkpoints were selected by dev ROC-AUC only.

These are development results, not final estimates. Because CAID2 was used to
detect the all-disordered collapse and select the balanced objective, it is now
a development regression set. The final external estimate must use an untouched
CAID3 release or another sealed homology-independent benchmark. Threshold and
calibration parameters must be selected on the UniRef50-disjoint dev split.

The staged 2,000-step runs completed without collapse. Dev-selected EMA best
checkpoints occurred at steps 1,800, 1,600, and 2,000 for seeds 2030, 2031, and
2032. Their dev ROC-AUC values were 0.8974, 0.8929, and 0.8975 (mean 0.8959,
sample SD 0.0026). CAID2 regression ROC-AUC values were 0.7435, 0.7417, and
0.7400 (mean 0.7418, sample SD 0.0017). Full checkpoint paths and metrics are in
`results/ablation/disorder_v4_balanced_long_summary.json`.

## Calibration

The seed-2032 EMA checkpoint is calibrated only on the UniRef50-disjoint dev
split. Weighted Platt scaling uses experimental labels at confidence 1.0 and
AFDB ordered proxies at confidence 0.25. The fitted scale is 0.8279, bias is
1.5501, and the dev MCC operating threshold is 0.9122. Grouped five-fold OOF
weighted Brier/ECE are 0.1002/0.0298 and OOF MCC is 0.6693. The artifact is
`calibration_artifacts/disorder_v4_balanced_s2032.json`.

Pass that artifact to `predict_disorder(..., calibration=path)` to obtain
calibrated probabilities and the frozen threshold. Absolute probability claims
remain limited because the ordered class uses AFDB proxy evidence rather than
experimental ordered labels.

## External evaluation

CAID evaluation supplies the target-to-cluster manifest, actual training
clusters, direct MMseqs2 eligibility audit, and frozen calibration. The runner
fails if any cluster overlaps or if a scored target has unresolved membership.

```bash
python scripts/benchmark_caid.py \
  --caid-dir data/caid3/disorder_nox \
  --checkpoint <seed-2032-ema-best.pt> \
  --calibration calibration_artifacts/disorder_v4_balanced_s2032.json \
  --cluster-manifest data/caid3/disorder_nox_clusters.json \
  --training-clusters data/caid3/balanced_v4_training_clusters.txt \
  --eligibility-audit data/caid3/disorder_nox_training_homology.json \
  --structure-backend skip --work-dir results/caid3_workdir \
  --out results/ablation/caid3_disorder_nox_balanced_v4_s2032.json
```

Report per-target mean and pooled ROC-AUC/PR-AUC, F1, MCC, Brier score, and ECE.
The frozen result scored 83 targets; 71 mixed-label UniRef50 clusters had macro
ROC-AUC 0.7551, bootstrap 95% CI [0.6945, 0.8119]. Metapredict reached 0.7927;
paired BFN-minus-metapredict was -0.0376, CI [-0.0786, 0.0056]. Brier score
0.2991 and ECE 0.3794 failed calibration criteria. The head may be described as
externally discriminative, but not superior or externally calibrated.
