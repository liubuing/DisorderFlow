# disorderflow 鈥?Protein Sequence Design Platform

**disorderflow** is a full-stack platform for fixed-backbone protein/antibody sequence design powered by **Bayesian Flow Networks (BFN)**. It integrates sequence design, confidence evaluation, cascade filtering, and AlphaFold2 validation into a unified Gradio web interface.

## Overview

| Component | Description |
|-----------|-------------|
| **BFN Model** | Bayesian Flow Network with Geometric Transformer (IPA) backbone |
| **Confidence Heads** | Built-in pLDDT, ipTM (context-only), and PAE prediction (V6 Phase 2) |
| **Cascade Filter** | 3-stage filtering: hard thresholds 鈫?dedup 鈫?composite scoring (PPL + entropy + pLDDT + ipTM) |
| **Web Platform** | Gradio UI with 8 specialized tabs for design, evaluation, and analysis |
| **Design Tools** | BFN, ProteinMPNN, ESM-IF all integrated with unified workflow |

## Quick Start

```bash
# Install
pip install -e .

# Launch web platform
python manage.py start

# Open browser 鈫?http://127.0.0.1:7860
```

## Web Platform Tabs

| Tab | Function |
|-----|----------|
| 馃幆 **鍗曟璁捐** | Single PDB 鈫?sequence design (BFN/ProteinMPNN/ESM-IF) |
| 馃摝 **鎵归噺澶勭悊** | Batch process multiple PDBs |
| 馃敭 **缁撴瀯棰勬祴** | AlphaFold2 (ColabFold) from FASTA 鈫?PDB |
| 鈿欙笍 **璁剧疆** | Config editor, model preloading, system status |
| 馃幆 **闈剁偣璁捐** | 5-step target-aware constrained antibody design |
| 馃攧 **杩唬绮句慨** | AF2-driven iterative refinement (design 鈫?validate 鈫?redesign) |
| 馃搳 **缃俊搴﹁瘎浼?* | Standalone BFN confidence assessment (pLDDT/ipTM/PAE) without designing |
| 馃殌 **缁熶竴宸ヤ綔娴?* | One-click pipeline: confidence 鈫?design 鈫?filter 鈫?AF2 validation |

## Model Versions

| Version | Dataset | Best val loss | ipTM Pearson r | pLDDT Pearson r | Architecture |
|---------|---------|---------------|----------------|-----------------|--------------|
| V5 Phase 5 | 1,149 proteins | 0.0521 | 0.957 | 0.817 | context-only ipTM |
| **V6 Phase 2** | **2,032 proteins** | **0.0494** | **0.950** | **0.825** | 3-layer pLDDT + context-only ipTM, 55.6% trainable |

The V6 Phase 2 model checkpoint is available on Hugging Face: [YueHuLab/disorderflow](https://huggingface.co/YueHuLab/disorderflow/)

The V6 training dataset (2,032 proteins, LMDB format) is available on Hugging Face: [liubuing/bfn-confidence-general-proteins](https://huggingface.co/datasets/liubuing/bfn-confidence-general-proteins)

## Cascade Filter

Three-stage filtering for design result ranking:

1. **Hard thresholds**: pLDDT, ipTM, PPL, entropy
2. **Deduplication**: Keep best PPL per unique sequence
3. **Composite scoring**: `0.35脳ipTM + 0.25脳pLDDT + 0.15脳PPL鈦宦?+ 0.10脳ent鈦宦?+ 0.15脳recovery`

Weights configurable in `app_config.yaml` 鈫?`workflow.cascade.weights`.

## Sequence Quality Proxy (PPL + Entropy)

Since ipTM is constant per protein (context-only pooling), **PPL** and **entropy** serve as the primary sequence-variant quality metrics within a single protein design:

- **PPL (Perplexity)**: Lower = more confident sequence assignment
- **Entropy**: Lower = sharper prediction distribution (model more certain per position)
- These are combined into a "quality score" component in the cascade filter

## Directory Structure

```
disorderflow-main/
鈹溾攢鈹€ app.py                    # Gradio web interface (main entry)
鈹溾攢鈹€ manage.py                 # Server management CLI
鈹溾攢鈹€ app_config.yaml           # Web platform configuration
鈹溾攢鈹€ cascade_filter.py         # 3-stage cascade filter
鈹溾攢鈹€ target_design_helpers.py  # Epitope scoring & contact analysis
鈹溾攢鈹€ af2_validator.py          # AlphaFold2 validation integration
鈹溾攢鈹€ iterative_refiner.py      # AF2-driven iterative refinement
鈹溾攢鈹€ configs/                  # Training & inference YAML configs
鈹?  鈹溾攢鈹€ train/                # Training configurations (V4-V6)
鈹?  鈹溾攢鈹€ test/                 # Test/evaluation configs
鈹?  鈹斺攢鈹€ demo_design.yml       # Default inference config
鈹溾攢鈹€ disorderflow/        # Core BFN package
鈹?  鈹溾攢鈹€ models/               # BFN model architectures
鈹?  鈹溾攢鈹€ modules/               # IPA, attention, diffusion, encoders
鈹?  鈹溾攢鈹€ datasets/             # Data loading (SAbDab, custom, LMDB)
鈹?  鈹溾攢鈹€ tools/                # Docking, evaluation, relaxation
鈹?  鈹斺攢鈹€ utils/                # Training, inference, transforms
鈹溾攢鈹€ ProteinMPNN/              # ProteinMPNN reference implementation
鈹溾攢鈹€ logs/                     # Training logs & checkpoints
鈹溾攢鈹€ 鏃ュ織/                     # Training log archive (parsed summaries)
鈹斺攢鈹€ data/                     # Training & evaluation datasets
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
