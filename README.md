# DisorderFlow — Candidate-Interface Confidence & BFN Sequence Design

Structure-conditioned protein and antibody design powered by Bayesian Flow
Networks (BFN), with a reproducible **candidate-interface confidence** line
that ranks antibody-peptide candidates by interface PAE.

This repository contains two things:

1. **Candidate-interface PAE confidence** — a reproducible, deployment-grade
   relative ranking signal for antibody-peptide interfaces (the focus of the
   `candidate_interface_*` scripts and configs).
2. **BFN confidence model** — the main-line model that predicts AF2-derived
   pLDDT / ipTM / PAE for protein structures.

> Availability of a feature does not imply it has passed biological validation.
> The candidate-interface line provides a *ranking* signal, not a binding claim.

## Reproducing the PAE deployment

The candidate-interface PAE deployment is fully reproducible. See
[`docs/CANDIDATE_INTERFACE_PAE_REPRODUCIBILITY.md`](docs/CANDIDATE_INTERFACE_PAE_REPRODUCIBILITY.md)
for the pipeline, scripts, prerequisites, and frozen contract.

The frozen deployment contract is
[`publication/candidate_interface_pae_deployment_v1.json`](publication/candidate_interface_pae_deployment_v1.json).

**Summary of the reproducible result** (six independent antibody-peptide scaffolds,
three training seeds):

| Metric | Value | Gate |
|---|---|---|
| Interface PAE median scaffold Spearman | 0.725–0.866 | ≥ 0.5 |
| Post-calibration MAE | 0.022–0.029 | ≤ 0.10 |
| Post-calibration variance ratio | 1.0 | [0.5, 2.0] |
| Pair accuracy | 0.843–0.953 | ≥ 0.65 |
| Bootstrap 95% lower | 0.683–0.873 | > 0.5 |

Deployment-grade signals: **interface PAE only**. pLDDT and ipTM are abstained.

## BFN confidence model (main line)

The BFN confidence model predicts AF2-derived confidence (pLDDT, ipTM, PAE) from
single-sequence input.

| Version | Dataset | Best val loss | ipTM r | pLDDT r |
|---|---|---|---|---|
| V5 Phase 5 | 1,149 proteins | 0.0521 | 0.957 | 0.817 |
| **V6 Phase 2** | **2,032 proteins** | **0.0494** | **0.950** | **0.825** |

- Model checkpoint: [liubuing/disorderflow](https://huggingface.co/liubuing/disorderflow/)
- Training dataset: [liubuing/bfn-confidence-general-proteins](https://huggingface.co/datasets/liubuing/bfn-confidence-general-proteins)

Set `DISORDERFLOW_CHECKPOINT` to the checkpoint path, or set `models.bfn.checkpoint`
in `app_config.yaml`.

### BFN design from the command line

`run_bfn_design` (in `modules/bfn_loader.py`) samples sequences for masked
regions of a fixed backbone:

```python
from bfn_loader import run_bfn_design

designs = run_bfn_design(
    pdb_path="data/misfolding_targets/5IMK.pdb",
    region_spec="B:26-33,51-58,97-113",
    num_samples=10, stochastic=True,
)
# → list of {sequence, ppl, entropy, plddt, iptm, pae, ...}
```

## Directory structure

```
├── disorderflow/        # Core BFN package (models, modules, datasets, utils)
├── modules/             # Loader, design, and validation entry points
├── scripts/             # Discovery, calibration, and evaluation pipelines
│   ├── build/           #   RCSB discovery, isolation, and pipeline config builders
│   └── pipelines/       #   Generation, selection, and AF2 runners
├── configs/             # Training, benchmark, and frozen protocol configs
├── publication/         # Frozen contracts and preregistration artifacts
├── docs/                # Reproducibility and protocol documents
├── tests/               # Unit tests
├── train.py             # Training entry point
└── ProteinMPNN/         # ProteinMPNN reference implementation
```

## Evidence status (honest)

| Capability | Evidence | Allowed interpretation |
|---|---|---|
| Candidate-interface PAE ranking | Deployment-grade, 3 seeds pass 6 gates | Relative ranking signal, not binding |
| Interface PAE transfer | Spearman 0.725–0.866 across 6 independent scaffolds | PAE transfers; pLDDT/ipTM do not |
| BFN confidence (V6) | ipTM r=0.950, pLDDT r=0.825 on general proteins | AF2 confidence is learnable |
| Binding / efficacy | No wet-lab measurements | No binding claim |

## Claim boundary

This repository provides a relative candidate ranking signal and a confidence
model. It does not claim that generated antibodies bind their targets; absolute
binding validation requires wet-lab measurement (SPR/BLI).

## License

MIT License. See [`LICENSE.md`](LICENSE.md).
