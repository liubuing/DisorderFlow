# iBEC 2026 Track Decision

## Primary Track

**AI-driven Life Science Discovery**

DisorderFlow is primarily an AI system addressing a life-science discovery problem: structure-conditioned antibody sequence design and prioritization for conformationally heterogeneous peptide epitopes. The BFN, disorder routing, ECLS benchmark, and A-beta candidate funnel are evaluated by their ability to support this biological discovery task.

## Secondary Alignment

**Bioinformatics Resources / Platforms**

The Gradio application, command-line workflows, frozen manifests, Windows/WSL2 runtime, and reproducibility package provide a strong platform implementation. This is an engineering strength, but it is the delivery form rather than the primary scientific objective.

## Why Not the Other Tracks?

### Novel Bioinformatics Algorithms / Methods

ECLS and the disorder-aware BFN receiver are method contributions, but the strongest validated result is a bounded structural scoring contrast. Direct BFN generation has not established superiority over ProteinMPNN. Selecting this track as primary would place disproportionate weight on proving generator-level algorithmic superiority.

### Bio-engineering Applications

The project provides an auditable computational-to-construct handoff, but no construct has been synthesized or experimentally measured. The current evidence therefore fits AI-assisted discovery better than completed bio-engineering implementation.

## Quantitative Fit

| Track | Fit | Rationale |
|---|---:|---|
| AI-driven Life Science Discovery | 92/100 | AI method, disease-relevant antibody design, validated computational discovery workflow |
| Bioinformatics Resources / Platforms | 84/100 | Strong runnable platform and reproducibility, but platform is not the primary scientific question |
| Novel Bioinformatics Algorithms / Methods | 76/100 | Original components exist, but generator superiority is not established |
| Bio-engineering Applications | 58/100 | Construct handoff exists; no wet-lab implementation |

## Submission Narrative

Use AI-driven Life Science Discovery in the registration system. In written materials, describe DisorderFlow as an auditable AI discovery platform and use the platform track language to support the engineering-design score.
