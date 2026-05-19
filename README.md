# AntibodyDesignBFN — Protein Sequence Design Platform

**AntibodyDesignBFN** is a full-stack platform for fixed-backbone protein/antibody sequence design powered by **Bayesian Flow Networks (BFN)**. It integrates sequence design, confidence evaluation, cascade filtering, and AlphaFold2 validation into a unified Gradio web interface.

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
# Install
pip install -e .

# Launch web platform
python manage.py start

# Open browser → http://127.0.0.1:7860
```

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

The V6 Phase 2 model checkpoint is available on Hugging Face: [YueHuLab/AntibodyDesignBFN](https://huggingface.co/YueHuLab/AntibodyDesignBFN/)

The V6 training dataset (2,032 proteins, LMDB format) is available on Hugging Face: [liubuing/bfn-confidence-general-proteins](https://huggingface.co/datasets/liubuing/bfn-confidence-general-proteins)

## Cascade Filter

Three-stage filtering for design result ranking:

1. **Hard thresholds**: pLDDT, ipTM, PPL, entropy
2. **Deduplication**: Keep best PPL per unique sequence
3. **Composite scoring**: `0.35×ipTM + 0.25×pLDDT + 0.15×PPL⁻¹ + 0.10×ent⁻¹ + 0.15×recovery`

Weights configurable in `app_config.yaml` → `workflow.cascade.weights`.

## Sequence Quality Proxy (PPL + Entropy)

Since ipTM is constant per protein (context-only pooling), **PPL** and **entropy** serve as the primary sequence-variant quality metrics within a single protein design:

- **PPL (Perplexity)**: Lower = more confident sequence assignment
- **Entropy**: Lower = sharper prediction distribution (model more certain per position)
- These are combined into a "quality score" component in the cascade filter

## Directory Structure

```
AntibodyDesignBFN-main/
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
├── antibodydesignbfn/        # Core BFN package
│   ├── models/               # BFN model architectures
│   ├── modules/               # IPA, attention, diffusion, encoders
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

- **IDP Blindness**: BFN confidence heads (pLDDT/ipTM) are trained on folded globular proteins and produce unreliable estimates on intrinsically disordered proteins
- **Length constraint**: Trained on proteins L=50-250; encoder position embeddings are out-of-distribution for L > 250
- **ipTM invariance**: Context-only ipTM is constant per protein, not per sequence (by design)

## License

Apache 2.0 License
