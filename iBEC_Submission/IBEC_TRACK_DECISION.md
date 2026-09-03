# iBEC 2026 Track Decision

## Primary Track

**Bioinformatics Resources / Platforms**

DisorderFlow is a complete computational platform for structure-conditioned
antibody CDR-H3 analysis and candidate prioritization against conformationally
heterogeneous peptide epitopes. The deliverable is the platform itself: a Gradio
web application, command-line runners, BFN/ProteinMPNN/ESM-IF generator
integration, AlphaFold-Multimer and PRODIGY structural triage, frozen evidence
contracts, SHA-256 manifest auditing, and an experimental-handoff schema. The
platform was exercised on two independent disease targets (amyloid-beta and tau)
and produced auditable computational shortlists with explicit claim boundaries.

## Secondary Alignment

**AI-driven Life Science Discovery**

The Bayesian Flow Network, epitope-conditioned likelihood shift (ECLS),
disorder-aware routing, and counterfactual benchmarking methodology constitute an
AI-driven discovery framework. The frozen temporal ECLS result on 15 antigen
clusters is a positive computational finding. However, no AI-generated candidate
has been experimentally validated as a binder, and the multiscaffold design
hypothesis failed all preregistered gates. The discovery narrative is therefore
supporting evidence for platform capability, not the primary claim.

## Why Not the Other Tracks?

### AI-driven Life Science Discovery (now secondary)

The platform's AI components were systematically evaluated. The core design
hypothesis (BFN generates better IDP-targeting antibodies than baselines) was
not supported: contact-specific guidance reached terminal negative, the
20-component multiscaffold experiment failed all four preregistered gates, and
the IDP de novo design ceiling (ipTM ~0.12-0.14) remains unbroken. ECLS
discrimination is positive but bounded. Selecting Discovery as the primary track
would place disproportionate weight on unvalidated AI-driven findings, when the
verified deliverable is the auditable platform that produced and documented them.

### Novel Bioinformatics Algorithms / Methods

ECLS and the disorder-aware BFN receiver are method contributions, but generator
superiority over ProteinMPNN was not established (native recovery 8% vs 36%;
developability pass fraction -16.5 pp). The strongest validated result is a
bounded structural scoring contrast, not an algorithmic advance. Selecting this
track would require proving method-level superiority that current evidence does
not support.

### Bio-engineering Applications

The project provides a complete computational-to-construct handoff (scFv
constructs, blinded BLI layouts, expression tracking), but no construct has been
synthesized or experimentally measured. The platform-to-handoff pipeline is an
engineering strength that supports the Platform track; a standalone
bio-engineering claim requires wet-lab implementation.

## Quantitative Fit

| Track | Fit | Rationale |
|---|---:|---|
| Bioinformatics Resources / Platforms | 92/100 | Complete runnable platform with web UI, CLI, frozen contracts, full AF2/PRODIGY pipeline, two-target case studies, and auditable reproducibility |
| AI-driven Life Science Discovery | 72/100 | Positive ECLS discrimination and systematic AI evaluation, but no validated discovery and core design hypotheses failed |
| Novel Bioinformatics Algorithms / Methods | 65/100 | Original components exist, but generator superiority and method-level advancement are not established |
| Bio-engineering Applications | 58/100 | Construct handoff exists; no wet-lab implementation |

## Submission Narrative

Use Bioinformatics Resources / Platforms in the registration system. In written
materials, lead with the platform architecture, reproducibility infrastructure,
and auditable pipeline as the primary deliverable. Present the AI-driven
discovery components (ECLS, BFN, disorder routing) as platform capabilities that
were systematically evaluated — including honest reporting of negative results —
rather than as the central scientific claim.
