# Current ECLS deposit candidate

Updated 2026-10-03. Upload only entries in UPLOAD_MANIFEST.json after author declarations and license approval are complete. The current archive is disorderflow-ecls-v1-reviewer-package.zip. Historical reviewer ZIPs, old source manifests, contact-v2 artifacts and prior PAE manuscripts are excluded.

The project recorded planned/reserved DOI 10.5281/zenodo.23037055 on 2026-09-29. Public Zenodo and DataCite API lookups returned HTTP 404 on 2026-10-03. This does not establish whether a private draft exists; it provides no evidence of public publication. Keep the identifier as planned until the correct account's record and uploaded files can be verified. See release/ecls_v1/PUBLICATION_STATUS_CHECK_20261003.json.

## Preparation

Run scripts/render_ecls_submission.py and visually inspect both PDFs. Run scripts/prepare_ecls_submission.py, extract its ZIP to a fresh directory and run verify_ecls_saved_results.py. Refresh the source manifest after code/document changes while preserving the prior manifest. Then run scripts/build_zenodo_upload.py and scripts/validate_publication_alignment.py.

## Metadata

The candidate resource type is dataset / saved-score evidence, with the accompanying manuscript clearly labelled author-review draft. Author order is retained internally in submission_metadata.json. The current rendered manuscript and Zenodo creator metadata withhold author information pending confirmation; preserve that setting until the author metadata are finalized. The existing MIT deposit license is a proposal pending confirmation for manuscript/data; software retains its existing MIT license.

## Unresolved publication inputs

See release/ecls_v1/submission_metadata.json for missing contributions, funding, conflicts, affiliations, related-manuscript status, all-author approval and license confirmation. No authenticated Zenodo API credential was available in the named environment variables during preparation. Use an authenticated account session when publication is ready; do not paste tokens into manuscripts, manifests or chat.

## Publication completion

Create or update a draft under the intended Zenodo account, upload the exact approved files and verify server-side checksums. A reserved DOI is not a published record. Publish after the author inputs are complete, then download each file and compare digests with the approved local manifest. Record the returned version DOI, URL and immutable code revision; only then update publication state and manuscript availability text. Preserve the submitted version as a separate archive revision rather than silently overwriting a deposited version.

This package remains local and unsubmitted until those steps are recorded. Journal submission is a separate action and requires a genuine receipt.
