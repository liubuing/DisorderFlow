# Current publication map

Updated 2026-09-16. This is the current routing document; it does not rewrite
frozen scientific decisions. `release/ecls_v1/publication_line.json` is its
machine-readable counterpart.

## Primary manuscript and release

- Line: **ECLS structural sequence scoring**.
- Manuscript title: **Backbone-conditioned likelihood contrasts for computational scoring of antibody CDR-H3 sequences in antibody-peptide complex structures**.
- Manuscript: `publication/MANUSCRIPT_DRAFT.md` (local unpublished draft).
- Protocol: `PUBLICATION_PROTOCOL.md`.
- Frozen scope: `publication/ECLS_SCOPE_FREEZE.yml`.
- Primary result: `results/publication/h3_ecls_temporal_final_v1/final_decision.json`.
- Reviewer package: `results/publication/h3_submission_package_v1/ECLS_GCLC_reviewer_package_v1.zip`.
- Release ID: `disorderflow-ecls-v1`; release directory: `release/ecls_v1/`.
- Deposit staging: `release/zenodo_v1/upload/`, enumerated by `UPLOAD_MANIFEST.json`.
- Stage: final computational evidence frozen; manuscript and local deposit
  preparation. Journal submission, acceptance and remote DOI are unconfirmed.

The primary result is native-versus-composition-matched-shuffle ECLS contrast
on 31 structures / 15 antigen clusters (mean 0.172281; bootstrap 95% CI
[0.059156, 0.291920]). Internal gate acceptance is not journal acceptance.
No universal reranking, successful antibody design, binding or affinity claim
is included. The temporal final must not be rerun.

## Separate manuscripts and supporting work

| Research line | Current entry point | Status / allowed use |
|---|---|---|
| PAE surrogate | `publication/af2_interface_pae_surrogate_manuscript_v4.md`, `docs/PAE_SURROGATE_REVISION_V4.md` | Independent revision; excluded from the ECLS deposit's manuscript and primary evidence |
| AF2 scoring-boundary analysis | `publication/af2_scoring_boundary_manuscript.md` | Historical PAE-related analysis; not an alternative ECLS title |
| Contact-v2 | `publication/successor_v3_contact_v2_registry.json` | Development candidate, no independent confirmation or wet-lab result; legacy ECLS artifact ZIP contains supporting provenance |
| BFN/IDP platform | `publication/idp_platform_capability_v1.json` | Historical capability inventory; PAE-specific claims must be read alongside its v4 corrections |

The legacy large artifact ZIP includes contact-v2 supporting material and the
ECLS reviewer package. Its presence does not promote contact-v2 into the primary
paper claim. PAE revisions neither replace nor retune ECLS evidence.

## Deposit status and consistency

The active metadata, title, manuscript, reviewer package and upload manifest all
refer to ECLS. Old mixed PAE/ECLS staging files are retained under
`release/zenodo_v1/archive/`, outside the active upload directory. Preparation
does not imply deposit publication or journal submission.

Run `python scripts/build_zenodo_upload.py` to rebuild local staging, then
`python scripts/validate_publication_alignment.py` to verify identity and hashes.
Source manifests are refreshed separately; frozen scientific results and
historical artifact ZIPs are preserved.
