# Current publication map

Updated 2026-09-28. Authoritative current routing: release/ecls_v1/publication_line.json. The frozen ECLS scientific scope remains unchanged; current editorial title and routing supersede older draft instructions.

## Primary manuscript and archive

- Title: **Peptide-context scoring of antibody loops: native recognition and the limits of candidate ranking**.
- Source: publication/MANUSCRIPT_DRAFT.md.
- Review PDF: publication/ECLS_MANUSCRIPT.pdf.
- Supplement: publication/ECLS_SUPPLEMENT.md and .pdf.
- Cover letter: publication/Cover_Letter_Bioinformatics.md.
- Target: Bioinformatics, Original Paper.
- Authors and pending declarations: release/ecls_v1/submission_metadata.json.
- Current archive: dist/disorderflow-ecls-v1-reviewer-package.zip.
- Active staging: release/zenodo_v1/upload/, enumerated by UPLOAD_MANIFEST.json.
- Stage: scientific manuscript rewritten for author review; author declarations and verified remote publication pending. No journal submission or DOI is asserted.

The narrative asks whether native-sequence recognition extends to candidate ranking. The primary temporal result remains mean advantage 0.172281, 95% CI [0.059156, 0.291920], on 31 structures in 15 antigen clusters. Universal candidate-ranking superiority was not established. Calibration is exploratory. There is no experimental binding claim.

## Separate and historical materials

| Material | Role |
|---|---|
| publication/pae_screening_evidence_v20_draft.md | Current PAE results/methods draft; retrospective composition-driven screening |
| docs/PAE_EVIDENCE_AUDIT_V20.md and docs/PAE_NEW_FAMILY_FEASIBILITY_V21.md | Current PAE claim limits and lack of newly eligible candidates within the searched scope |
| publication/af2_interface_pae_surrogate_manuscript_v4.md | Corrected historical PAE revision; v4 corrections remain binding context |
| PAE v3 Markdown, TeX and PDF | Superseded submission versions; not current submission artifacts |
| ECLS_GCLC_reviewer_package_v1.zip | Historical archive with older embedded manuscript; excluded from current upload |
| disorderflow-ecls-v1-artifacts.zip and contact-v2 | Historical supporting development bundle; excluded from current upload |

Previous editorial files are preserved under release/zenodo_v1/archive/pre-consolidation-20260928T153543/. They are not included in the current public candidate. Related-manuscript overlap must be disclosed; different internal identifiers do not prove independence. No mandatory three-month separation rule is asserted.

## Rebuild

Run scripts/render_ecls_submission.py, scripts/prepare_ecls_submission.py, scripts/build_zenodo_upload.py and scripts/validate_publication_alignment.py in that order. Numerical verification is available in the current archive as verify_ecls_saved_results.py. The build does not perform model inference or publish a record.
