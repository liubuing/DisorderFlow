# Consistency audit and remediation log (2026-11-11)

Scope: prose, release metadata and ignore rules. **No frozen artifact was
edited.** The audit is read-only and checks documents against
`docs/EVIDENCE_REGISTRY.yml`; the registry is the single source of truth for
numbers that appear in more than one place.

## What changed

| Area | Change | File |
|---|---|---|
| Ignore rules | Added the 4.5 GB iBEC bundle and its ZIP; they were untracked but not ignored | `.gitignore` |
| Registry | Canonical values for ECLS, PAE, FLEX, AAYL, CSN, disorder head, T2.1, multiscaffold and 4HIX; explicit stale values and allowed markers | `docs/EVIDENCE_REGISTRY.yml` |
| Audit tool | `scripts/audit_doc_consistency.py`: scans README, protocol, `docs/`, `release/`, `publication/`; reports unmarked stale values; optional missing-source check | `scripts/audit_doc_consistency.py` |
| CI | New step runs `audit_doc_consistency.py --fail-on-stale` on every push/PR job | `.github/workflows/ci.yml` |
| Provenance | Corrected the claim that `publication_line.json` retains the pre-revision title; the old title lives in `ECLS_SCOPE_FREEZE.yml` | `PUBLICATION_PROTOCOL.md` |
| Release README | Replaced the pre-revision title; added a note that the reviewer ZIP still embeds the old title and must be rebuilt | `release/ecls_v1/PUBLIC_ARCHIVE_README.md` |
| Historical handoff | Marked the 2026-09-29 handoff SUPERSEDED; old title retained as provenance only | `release/ecls_v1/EDITORIAL_HANDOFF_20260929.md` |
| Decision record | Marked 0.124 and 0/26 as pre-fix values inline | `docs/DECISION_20261101_MODEL_LINE_CLOSURE.md` |
| FLEX draft v1 | Marked SUPERSEDED by draft v2 | `publication/flex_boundary_manuscript_draft_v1.md` |
| PII | Removed the old corresponding-author name/email from the local submission-scope record; added a SUPERSEDED banner | `publication/SUBMISSION_SCOPE_DECISION.md` (local, ignored) |
| Minor drift | `app.py` is 73 lines, not 65 | `README.md` |
| Release indexes | Refreshed the two changed source-manifest entries; regenerated `UPLOAD_MANIFEST.json` and the local upload staging; added a disclosure | `release/ecls_v1/source_manifest.json`, `release/zenodo_v1/UPLOAD_MANIFEST.json`, `release/ecls_v1/MANIFEST_UPDATE_20261111.md` |

## Audit result

```text
documents with stale values: 6 | unmarked hits: 0 | marked hits: 23
```

The six documents that still quote voided values do so inside explicit
SUPERSEDED / correction / void banners:

- `docs/FLEX_BINDER_CAMPAIGN_V1.md` (8 marked hits; §8–§9 carry section banners)
- `publication/flex_boundary_manuscript_draft_v1.md` (4; void banner)
- `publication/flex_boundary_manuscript_draft_v2.md` (2; correction-chain paragraph)
- `docs/DECISION_20261101_MODEL_LINE_CLOSURE.md` (7; before/after narrative)
- `release/ecls_v1/EDITORIAL_HANDOFF_20260929.md` (1; superseded banner)
- `release/ecls_v1/PUBLIC_ARCHIVE_README.md` (1; correction note quotes the old title)

Run locally:

```bash
python scripts/audit_doc_consistency.py
python scripts/audit_doc_consistency.py --fail-on-stale --fail-on-missing-source
python scripts/audit_doc_consistency.py --json
```

Verification run on 2026-11-11:

```text
python scripts/audit_doc_consistency.py --fail-on-stale        -> exit 0
python scripts/validate_release_lineage.py --source-only       -> lineage-valid, 218 artifacts
python scripts/validate_publication_alignment.py               -> status valid, errors []
pytest tests/test_release_lineage.py tests/test_repository_boundaries.py tests/test_ecls_submission.py -> 15 passed
```

`validate_publication_alignment.py` still lists the author/declaration/DOI
blockers; those are the U2/U3/U4 items below, not hash errors.

## Unresolved items (require rebuild or a human decision)

| # | Item | Why it cannot be fixed by prose alone | Recommended action |
|---|---|---|---|
| U1 | Reviewer ZIP `dist/disorderflow-ecls-v1-reviewer-package.zip` embeds the pre-revision title | The ZIP is a built artifact; editing the source README does not change it | Rebuild with `scripts/render_ecls_submission.py` → `scripts/prepare_ecls_submission.py`; verify hashes in `release/zenodo_v1/UPLOAD_MANIFEST.json` |
| U2 | `release/zenodo_v1/zenodo_metadata.json` declares `license: MIT` while `submission_metadata.json` has `manuscript_data_license_confirmed: false` | License is an author decision, not an editorial one | Set the metadata license only after all-author confirmation; until then leave the upload stage as `local_preparation_not_published` |
| U3 | Author approvals, CRediT, funding, COI, affiliations and corresponding email are still null | Requires author input | Collect the eight pending items in `release/ecls_v1/submission_metadata.json`; do not flip `all_authors_approve_submission` |
| U4 | Reserved DOI `10.5281/zenodo.23037055` is not a published record | Requires an authenticated Zenodo deposit and a download/hash check | Publish only after U2/U3; then verify the downloaded files against the manifest |
| U5 | iBEC package identity exposure and third-party weights | The package is a competition artifact; the blind-review rules and weight licenses are external constraints | Apply `ANONYMIZATION_PLAN.md`; do not share the 4.5 GB package until the weight licenses are reviewed. It is now gitignored but still on disk |
| U6 | Literal `C:` directory in the repo root (7.4 GB) | It is invisible to Git on Windows (the name contains a private-use colon) and cannot be removed safely without inspection | Inspect `C:/Users/34342/.dsh/profiles` and `C:/development/Git/mnt/d`; quarantine or delete manually. Do not `git add` it |
| U7 | ATLAS-derived results are tracked under `results/esel_e0b_v1/` and `results/flex_a2_atlas_nmr_audit_v2/` | ATLAS is CC BY-NC 4.0 | Add a colocated NOTICE and keep the non-commercial restriction visible in the public README |
| U8 | `diffab/` and `HDOCKlite-v1.1/` have no top-level license file | Third-party terms | Add a `THIRD_PARTY_NOTICES.md`; do not redistribute HDOCKlite binaries |
| U9 | Working-tree changes from a parallel process (`manage.py`, `modules/misfolding_pipeline.py`, `webapp/state.py`, `app_v2.py`, `.dsh-research/frontend-audit/`) | Not part of this audit | Review separately; this log does not touch them |

## What this does not claim

- It does not make the manuscript submission-ready.
- It does not change any scientific number or verdict.
- It does not verify the reviewer ZIP or the Zenodo deposit.
- It does not fix the wet-lab gap or the model-capability gap.

The next batch (B2) is the executable one: a real-checkpoint end-to-end smoke
test, a numerical regression test, and a clean-checkout CI fixture bundle.
