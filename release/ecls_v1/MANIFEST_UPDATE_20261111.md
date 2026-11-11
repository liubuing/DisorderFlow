# Manifest update disclosure (2026-11-11)

Two source documents were editorially corrected on 2026-11-11:

- `README.md`: the `app.py` description now says 73-line launcher (was 65).
- `PUBLICATION_PROTOCOL.md`: the provenance sentence now points to
  `publication/ECLS_SCOPE_FREEZE.yml` for the pre-revision title. The earlier
  text incorrectly said `release/ecls_v1/publication_line.json` retained it;
  that file already carries the current title.

Because both files are members of the source manifest, the following indexes
were refreshed:

- `release/ecls_v1/source_manifest.json`: the two entries now carry the
  current canonical-LF SHA-256 and byte counts.
- `release/zenodo_v1/UPLOAD_MANIFEST.json` and the local
  `release/zenodo_v1/upload/` staging: regenerated with
  `scripts/build_zenodo_upload.py`; all 20 staged files re-hashed.

What did **not** change:

- No frozen scientific result, decision, score, checkpoint, threshold or
  claim was edited.
- `release/ecls_v1/source_manifest_frozen_v1.json` is untouched.
- The manuscript title, authors, license state and DOI state are unchanged.

Known remaining artifact:

- `dist/disorderflow-ecls-v1-reviewer-package.zip` was built before the
  correction and still embeds the pre-revision README title. Rebuild the ZIP
  with the documented render/prepare pipeline before any deposit. The current
  ZIP is not submission-ready.

Verification:

```bash
python scripts/validate_release_lineage.py --source-only
python scripts/validate_publication_alignment.py
python scripts/audit_doc_consistency.py --fail-on-stale
```

`validate_publication_alignment.py` returns `status: valid` with the expected
author/declaration blockers and no hash errors.
