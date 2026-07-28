# v5.1 IDP Antibody Training Plan

## Scope

v5.1 means the frozen 1,289-entry conformation LMDB after MMseqs2
homology-disjoint splitting and a zero-overlap audit. It does not include CAID2;
CAID2 remains a blind external evaluation set.

The first curriculum uses the existing `phase3_expanded` antibody-antigen complexes
after rebuilding the split across PDB, paired-antibody, and antigen connected clusters.
The SAbDab2 0.1.0 `abag` dataset should replace or extend this source after its official
archive has downloaded and an importer has been validated.

## Curriculum

| Stage | Data | Trainable objective | Iterations |
| --- | --- | --- | ---: |
| 1. Disorder representation | v5.1 ensembles | RMSF-derived disorder; top 2 encoder layers | 2,500 |
| 2. Antigen routing | Phase3 v5.1 | CDR sequence and contact | 4,000 |
| 3. CDR co-design | Phase3 v5.1 | sequence, distance, FAPE, torsions, interface | 3,500 |
| 4. IDP conditioning | Phase3 v5.1 + disorder profiles | co-design, disorder-aligned diversity, anti-degeneration | 3,000 |

All stage transitions use `train.py --init`, which retains every shape-compatible
tensor and starts a fresh optimizer. Legacy `--finetune` is intentionally not used
because it strips sequence, geometry, confidence, and disorder-conditioning heads.

## Hard Gates

Training starts only when all conditions pass:

- Raw v5 contains exactly 1,289 entries and matches the frozen manifest.
- Clustered train, validation, and experimental-calibration holdout counts sum to 1,289.
- AF2-seed RMSF is calibrated against at least 10 solution-NMR ensembles. Stage 1
  starts only for a `pass_physical_label` verdict (median within-protein Spearman
  >=0.30, top-20% recall >=0.30, and partial Spearman controlling 1-pLDDT >=0.15).
- The full NMR comparison and the annotated-IDP subset must pass independently;
  performance driven only by folded domains cannot unlock IDP supervision.
- Every MMseqs2 cluster represented in the NMR calibration set is excluded from both
  training and validation and written to `confidence_calibration.lmdb`.
- Conformation and Phase3 audits report zero exact overlap and zero cross-split
  homology clusters.
- Phase3 contains at least 500 usable antibody-protein-antigen complexes.
- The disorder lookup exists.
- The final stage has at least 500 profile-backed complexes overall and at least 20
  in each split; missing profiles are excluded rather than treated as ordered labels.
- The incoming checkpoint has only finite tensors and covers at least 60% of the
  target model parameters by shape-compatible parameter count.
- Every stage produces a finite `best.pt` before the next stage begins.

These gates establish data and numerical validity, not biological efficacy. Binding,
specificity, developability, and ensemble robustness require separate held-out
generation benchmarks and SPR/BLI or equivalent experimental validation.

The calibration verdict controls label semantics. `rank_only` permits the Stage 1
within-protein pairwise ranking loss, which uses `tanh(RMSF/2)` only as a monotonic
ordering target. It does not permit continuous RMSF regression or Huber supervision.

CDR-antigen contrastive compatibility is disabled in the default v5.1 configs. Both
within-CDR permutation negatives and cross-sample CDR negatives remained at held-out
AUROC approximately 0.50. The current single-positive-per-antigen data cannot establish
that objective; re-enabling it requires validated multi-ligand or experimental negatives.

## Commands

```bash
bash scripts/run_v5_1_training_pipeline.sh --dry-run
bash scripts/run_v5_1_training_pipeline.sh --wait
```

Set `DISORDERFLOW_BASE_CHECKPOINT` to replace the default public
AntibodyDesignBFN candidate. Pipeline state and the final checkpoint path are written
under `logs/v5_1_pipeline/state/`; completed stages are restartable.
