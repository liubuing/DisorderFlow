# disorderflow — Protein Sequence Design Platform

**disorderflow** is a full-stack platform for fixed-backbone protein/antibody sequence design powered by **Bayesian Flow Networks (BFN)**. It integrates sequence design, confidence evaluation, cascade filtering, and AlphaFold2 validation into a unified Gradio web interface.

## Overview

| Component | Description |
|-----------|-------------|
| **BFN Model** | Bayesian Flow Network with Geometric Transformer (IPA) backbone |
| **Confidence Heads** | Built-in pLDDT, ipTM (context-only), and PAE prediction (V6 Phase 2) |
| **Cascade Filter** | 3-stage filtering: hard thresholds → dedup → composite scoring (PPL + entropy + pLDDT + ipTM) |
| **Web Platform** | Gradio UI with 8 specialized tabs for design, evaluation, and analysis |
| **Design Tools** | BFN, ProteinMPNN, ESM-IF all integrated with unified workflow |

## Quick Start

```bash
# Install the web application and optional ESM-IF integration
pip install -e ".[ui]"

# Launch web platform
python manage.py start

# Open browser → http://127.0.0.1:7860
```

### BFN design from the command line

`run_bfn_design` (in `modules/bfn_loader.py`) is the core design entry point —
it samples diverse sequences for the masked regions of a fixed backbone:

```python
from bfn_loader import run_bfn_design

# Fixed-backbone (FixBB) design: design CDRs on chain B, residues 26-33,51-58,97-113
designs = run_bfn_design(
    pdb_path="data/misfolding_targets/5IMK.pdb",
    region_spec="B:26-33,51-58,97-113",
    num_samples=10, stochastic=True,
)
# → list of {sequence, ppl, entropy, plddt, iptm, pae, ...} one per sample
```

For antibody CDR design against a target epitope (Complex mode), use the
end-to-end IDP-aware pipeline:

```bash
# Target-aware antibody design with disorder analysis + AF2 validation
python run_idp_design.py \
    --target data/misfolding_targets/2NAO_model1_A_1-42.pdb --target-chain A \
    --scaffold data/misfolding_targets/5IMK.pdb --scaffold-chain B \
    --samples 10 --af2

# Single PDB → sequence design (BFN) with cascade filtering + AF2 validation
python run_design.py --pdb <your.pdb> --chain A --regions "A:10-30"
```

`context_chains=None` includes every non-design PDB chain (Complex mode), while
`context_chains=[]` includes only design chains (FixBB mode). Passing an
explicit chain list is recommended for auditable antigen-aware experiments.

## Web Platform Tabs

| Tab | Function |
|-----|----------|
| 🎯 **单步设计** | Single PDB → sequence design (BFN/ProteinMPNN/ESM-IF) |
| 📦 **批量处理** | Batch process multiple PDBs |
| 🔮 **结构预测** | AlphaFold2 (ColabFold) from FASTA → PDB |
| ⚙️ **设置** | Config editor, model preloading, system status |
| 🎯 **靶点设计** | 5-step target-aware constrained antibody design |
| 🔄 **迭代精修** | AF2-driven iterative refinement (design → validate → redesign) |
| 📊 **置信度评估** | Standalone BFN confidence assessment (pLDDT/ipTM/PAE) without designing |
| 🚀 **统一工作流** | One-click pipeline: confidence → design → filter → AF2 validation |

## Model Versions

| Version | Dataset | Best val loss | ipTM Pearson r | pLDDT Pearson r | Architecture |
|---------|---------|---------------|----------------|-----------------|--------------|
| V5 Phase 5 | 1,149 proteins | 0.0521 | 0.957 | 0.817 | context-only ipTM |
| **V6 Phase 2** | **2,032 proteins** | **0.0494** | **0.950** | **0.825** | 3-layer pLDDT + context-only ipTM, 55.6% trainable |

The V6 Phase 2 model checkpoint is available on Hugging Face: [liubuing/disorderflow](https://huggingface.co/liubuing/disorderflow/)

Set `DISORDERFLOW_CHECKPOINT` to the downloaded checkpoint path, or update
`models.bfn.checkpoint` in `app_config.yaml`. The repository does not bundle BFN
weights; startup now reports a missing checkpoint explicitly instead of silently
selecting a different model version.

The V6 training dataset (2,032 proteins, LMDB format) is available on Hugging Face: [liubuing/bfn-confidence-general-proteins](https://huggingface.co/datasets/liubuing/bfn-confidence-general-proteins)

## Cascade Filter

Three-stage filtering for design result ranking:

1. **Hard thresholds**: pLDDT, ipTM, PPL, entropy
2. **Deduplication**: Keep best PPL per unique sequence
3. **Composite scoring**: `0.35×ipTM + 0.25×pLDDT + 0.15×PPL⁻¹+ 0.10×ent⁻¹+ 0.15×recovery`

Weights configurable in `app_config.yaml` →`workflow.cascade.weights`.

## Sequence Quality Proxy (PPL + Entropy)

Since ipTM is constant per protein (context-only pooling), **PPL** and **entropy** serve as the primary sequence-variant quality metrics within a single protein design:

- **PPL (Perplexity)**: Lower = more confident sequence assignment
- **Entropy**: Lower = sharper prediction distribution (model more certain per position)
- These are combined into a "quality score" component in the cascade filter

## Directory Structure

```
disorderflow-main/
├── app.py                    # Gradio web interface (main entry)
├── manage.py                 # Server management CLI
├── app_config.yaml           # Web platform configuration
├── cascade_filter.py         # 3-stage cascade filter
├── target_design_helpers.py  # Epitope scoring & contact analysis
├── af2_validator.py          # AlphaFold2 validation integration
├── iterative_refiner.py      # AF2-driven iterative refinement
├── configs/                  # Training & inference YAML configs
│   ├── train/                # Training configurations (V4-V6)
│   ├── test/                 # Test/evaluation configs
│   └── demo_design.yml       # Default inference config
├── disorderflow/             # Core BFN package
│   ├── models/               # BFN model architectures
│   ├── modules/              # IPA, attention, diffusion, encoders
│   ├── datasets/             # Data loading (SAbDab, custom, LMDB)
│   ├── tools/                # Docking, evaluation, relaxation
│   └── utils/                # Training, inference, transforms
├── ProteinMPNN/              # ProteinMPNN reference implementation
├── logs/                     # Training logs & checkpoints
├── 日志/                     # Training log archive (parsed summaries)
└── data/                     # Training & evaluation datasets
```

## Training Data

The confidence model (V6) was trained on **2,032 protein structures** from Swiss-Prot plus brain disease-associated proteins, with AF2 confidence scores as training targets.

- **V4**: 503 train / 126 val (Swiss-Prot L=50-250)
- **V5**: 919 train / 230 val (+ disease proteins)
- **V6**: 1,625 train / 407 val (+ brain disease, TrEMBL)

### V5.3 Research Checkpoint

The V5.1-V5.3 research line adds homology-disjoint conformation supervision,
full-group StateContrast structural ranking, and rank-only flexibility adaptation.
The release gates select iteration `700.pt`, rather than the validation-loss
`best.pt`, as the authoritative V5.3 checkpoint. See
[`docs/V5_3_RELEASE_MANIFEST.md`](docs/V5_3_RELEASE_MANIFEST.md) for its checksum,
selection rule, gate results, and scientific claim boundary.

## Datasets

All training and evaluation datasets are hosted on Hugging Face: [liubuing/disorderflow](https://huggingface.co/datasets/liubuing/disorderflow)

```bash
# Download all datasets (~220GB)
pip install huggingface_hub
python scripts/download_assets.py --all

# Or download manually with hf CLI
hf download liubuing/disorderflow --repo-type dataset --local-dir ./hf_data
```

Key dataset directories:

| Directory | Size | Description |
|-----------|------|-------------|
| `confidence_design_variants_v14` | ~98 GB | Grouped design variants for V14 confidence training |
| `confidence_conformation_v5` | ~65 GB | Conformation ensemble dataset (V5) |
| `confidence_unified_v2` | ~20 GB | Unified confidence dataset (V2) |
| `confidence_idp_unified` | ~10 GB | IDP-specific confidence data |
| `statecontrast_structural_v2_1` | ~8 GB | StateContrast structural ranking pairs |
| `processed` | ~1.5 GB | Processed structures (LMDB) |
| `sabdab2_current` | ~850 MB | SAbDab2 antibody database snapshot |
| `disprot_current` | ~31 MB | DisProt disorder annotations |
| `misfolding_targets` | ~6 MB | Disease target PDBs (Aβ, α-syn, etc.) |

## Management CLI

```bash
python manage.py start              # Start web server (foreground)
python manage.py start --bg         # Start in background
python manage.py stop               # Stop server
python manage.py restart            # Restart server
python manage.py status             # Service status
python manage.py batch <dir>        # Batch process PDB directory
python manage.py config             # View current configuration
python manage.py config --edit      # Edit configuration
python manage.py test               # Environment self-test
```

## Known Limitations

- **IDP recognition is not externally validated**: the current CAID-standard result
  (ROC-AUC 0.470, PR-AUC 0.00725) does not support a generalization claim. Disorder
  scores should be treated as experimental features until leakage-free retraining
  and homology-clustered external evaluation are complete.
- **IDP De novo Design — ceiling not yet broken**: While the disorder head identifies IDPs,
  de novo CDR design for IDP targets (e.g. Aβ42) has not surpassed the AF2 folding
  ceiling (ipTM~0.12–0.14 on 3STB scaffold, vs >0.3 target for viable binders).
  The V17i Pair Routing breakthrough (Δ=+30pp antigen signal) and V18 MPNN+BFN hybrid
  pipeline are active directions. For ordered targets, BFN confidence scoring is reliable.
- **Length constraint**: Trained on proteins L=50-250; encoder position embeddings are
  out-of-distribution for L > 250
- **ipTM invariance**: Context-only ipTM is constant per protein, not per sequence
  (by design). V14+ grouped confidence heads provide per-design discrimination.
- **AF2 metric scope**: monomer validation reports pLDDT and pTM only. ipTM is
  reported and used for ranking only when an antigen sequence is explicitly
  supplied to AF2 Multimer.
- **Split policy**: new dataset builds use MMseqs2 homology clusters rather than
  random PDB splits. MMseqs2 is therefore required for publication-grade builds.

## License

MIT License. See `LICENSE.md`.
