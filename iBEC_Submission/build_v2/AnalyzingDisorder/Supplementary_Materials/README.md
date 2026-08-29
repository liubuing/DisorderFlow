# DisorderFlow iBEC 2026 Supplementary Materials

## Scope

This lightweight package documents the current computational platform and frozen evidence. It does not contain third-party model weights, licensed databases, wet-lab results, or a binding claim.

## Key Evidence

| Artifact | Purpose | Status |
|---|---|---|
| `ecls_final_decision.json` | Frozen 15-cluster temporal result | Confirmatory computational positive |
| `abeta_candidate_summary.json` | 352-to-24-to-12 candidate funnel | Computational case study |
| `abeta_final_shortlist.csv` | Final 12 H3 candidates | Hypotheses only |
| `binder_confidence_status.json` | Experimental-feedback readiness | Abstain, no labels |
| `e1_generator_behavior_summary.json` | BFN/ProteinMPNN behavior | Broader exploration with developability tradeoff |
| `e2_idp_evidence_summary.json` | 1,289-record evidence audit | 219 exact DisProt disorder-region records |
| `tau_sanity_status.json` | Tau/5MP3 second-target sanity | Go for correctly masked extension |
| `tau_candidate_results.json` | Correct-mask Tau H3 extension | 275 unique to 8 diverse computational candidates |
| `REPRODUCIBILITY.md` | Commands and evidence map | Documentation |

## Reproduction Boundaries

- The temporal ECLS final is terminal and must not be rerun.
- Saved summaries can be independently recomputed from frozen result rows.
- Full AF2 reproduction requires ColabFold/AlphaFold parameters distributed separately.
- ProteinMPNN and ESM-IF weights are referenced by checksum and upstream license.
- Shuffled H3 controls are counterfactuals, not experimental non-binders.

## Current Claim Boundary

DisorderFlow provides structure-conditioned sequence analysis and computational candidate prioritization for flexible peptide epitopes. No experimental binding, affinity, specificity, efficacy, or binder probability is claimed.

## Competition Demo Plan

If selected for the final round, the live demo will show:

1. Loading a prevalidated 4HIX antibody-peptide complex.
2. Explicit heavy, light, and antigen chain assignment.
3. Correct 12-residue H3 mask construction.
4. Candidate generation and multi-pose scoring.
5. Frozen filtering and provenance display.
6. Export of the computational shortlist and evidence boundary.
