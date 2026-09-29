# ECLS publication contract and evidence boundary

Updated 2026-09-28. This editorial consolidation preserves the original scientific contract in publication/ECLS_SCOPE_FREEZE.yml and the terminal result decision. Its updated working title is recorded in release/ecls_v1/publication_line.json; the older title in frozen records is historical provenance.

## Current manuscript

Peptide-context scoring of antibody loops: native recognition and the limits of candidate ranking.

Primary source: publication/MANUSCRIPT_DRAFT.md. Supplement: publication/ECLS_SUPPLEMENT.md. Target: Bioinformatics Original Paper. Current status: author-review manuscript and local archive candidate, pending author declarations and remote publication.

## Central question

Does an ECLS preference for native H3 over composition-matched shuffles justify general candidate ranking? Treat the two tests separately. The native-versus-shuffle temporal result is positive; universal ranking superiority was not established in development. Generator-aware calibration remains exploratory.

## Immutable primary result

ECLS = H3 NLL(complex backbone) - H3 NLL(peptide-stripped, complex-derived Fab backbone).

Advantage = mean ECLS(composition-matched shuffled H3) - ECLS(native H3).

31 structures / 15 antigen clusters; mean 0.172281; median 0.118699; 95% cluster-bootstrap CI [0.059156, 0.291920]; 12/15 positive clusters. These are structural likelihood quantities, not binding energies. Shuffles are counterfactuals, not experimentally established non-binders.

The final decision is results/publication/h3_ecls_temporal_final_v1/final_decision.json. Attempt one is terminal and rerun_permitted is false. Frozen data, masks, thresholds, coefficients, checkpoints and source results must not be changed to improve the narrative.

## Evidence hierarchy

| Evidence | Role |
|---|---|
| 46-cluster adaptation | Exposed development evidence |
| 15-cluster temporal final | Sole positive primary inference |
| Seven-cluster universal reranking | Negative development gate; central limitation |
| Generator-aware calibration | Exploratory development; no confirmed superiority over complex NLL |
| Other DisorderFlow platform studies | Historical context; outside this manuscript's inference |
| PAE v20 / v21 | Separate retrospective study and data-availability audit |

The native score comparison does not establish generated-candidate binding, affinity, specificity, causal hotspots, therapeutic efficacy, unbound-state recovery or general antibody-design improvement. Temporal eligibility does not prove complete training independence.

## Publication execution

The decision, checklist, cover letter, manuscript, metadata and upload manifest must agree on the current ECLS title and authors. PAE v3 is superseded for submission purposes; its corrected v4 audit and current v20/v21 limitations must be respected. No three-month spacing rule is imposed. Related manuscripts and actual submission/posting status must be disclosed under journal policy.

Current authors are recorded in release/ecls_v1/submission_metadata.json. Funding, CRediT contributions, conflicts, affiliation mapping, license and all-author approval remain pending. No affirmative declaration is inferred.

## Reproducibility boundary

The current archive verifies saved-score numerical results. Public derivatives normalize machine-specific execution paths; original SHA-256 digests and transformed-copy digests are recorded separately. Third-party weights and raw structural databases are excluded. The legacy reviewer ZIP and contact-v2 artifact bundle remain historical and are excluded from current staging. Full model reproduction is not claimed by the minimal archive.

The public record is complete only after a real immutable URL/DOI is returned and downloaded files match the approved manifest. Local compilation or package validation does not establish publication or journal submission.
