# Current ECLS deposit candidate

Updated 2026-09-28. Upload only entries in UPLOAD_MANIFEST.json after author declarations and license approval are complete. The current archive is disorderflow-ecls-v1-reviewer-20260928.zip. Historical reviewer ZIPs, old source manifests, contact-v2 artifacts and prior PAE manuscripts are excluded.

## Preparation

Run scripts/render_ecls_submission.py and visually inspect both PDFs. Run scripts/prepare_ecls_submission.py, extract its ZIP to a fresh directory and run verify_ecls_saved_results.py. Refresh the source manifest after code/document changes while preserving the prior manifest. Then run scripts/build_zenodo_upload.py and scripts/validate_publication_alignment.py.

## Metadata

The candidate resource type is dataset / saved-score evidence, with the accompanying manuscript clearly labelled author-review draft. Creators follow the user's order: Chen, Haoyang; Hu, Jing; Dai, Tanyu; Sun, Yuchao; Su, Jing. Jing Su is corresponding author, sj_12346@163.com. English spelling and affiliation mapping remain to be verified. The existing MIT deposit license is a proposal pending confirmation for manuscript/data; software retains its existing MIT license.

## Unresolved publication inputs

See release/ecls_v1/submission_metadata.json for missing contributions, funding, conflicts, affiliations, related-manuscript status, all-author approval and license confirmation. No authenticated Zenodo API credential was available in the named environment variables during preparation. Use an authenticated account session when publication is ready; do not paste tokens into manuscripts, manifests or chat.

## Publication completion

Create or update a draft under the intended Zenodo account, upload the exact approved files and verify server-side checksums. A reserved DOI is not a published record. Publish after the author inputs are complete, then download each file and compare digests with the approved local manifest. Record the returned version DOI, URL and immutable code revision; only then update publication state and manuscript availability text. Preserve the submitted version as a separate archive revision rather than silently overwriting a deposited version.

This package remains local and unsubmitted until those steps are recorded. Journal submission is a separate action and requires a genuine receipt.
