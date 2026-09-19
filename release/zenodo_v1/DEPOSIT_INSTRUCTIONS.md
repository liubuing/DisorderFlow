# ECLS v1 local deposit preparation

Primary paper: **Backbone-conditioned likelihood contrasts for computational scoring of antibody CDR-H3 sequences in antibody-peptide complex structures**.

The current manuscript is `publication/MANUSCRIPT_DRAFT.md`. Its scope is the
frozen ECLS native-versus-composition-shuffle structural sequence-scoring result.
The PAE surrogate is a separate manuscript revision and is not included here.

## Status

Files are prepared locally. Journal submission, acceptance, remote publication
and a DOI have not been confirmed. A metadata request to reserve a DOI is not
an issued DOI. No submission or publication occurs during local assembly.

## Rebuild and verify

```bash
python scripts/build_zenodo_upload.py
python scripts/validate_publication_alignment.py
python scripts/validate_release_lineage.py --source-only
```

If the source manifest is stale, generate a new manifest with
`scripts/build/build_release_source_manifest.py --output <new-path>`, preserve
the old manifest, and make the verified new one the active source manifest.
Do not regenerate the frozen ECLS final experiment or overwrite the historical
large artifact ZIP merely to refresh documentation.

## Active files

- `zenodo_metadata.json`: ECLS title, author metadata and bounded claims.
- `UPLOAD_MANIFEST.json`: exact active filenames, source paths and SHA-256 hashes.
- `upload/`: upload only the files listed in the active manifest.
- `archive/`: old mixed PAE/ECLS staging material; exclude from any upload.

The active set includes the ECLS Markdown manuscript, publication protocol,
frozen scope, primary final decision and results, reviewer package, main result
table, reproducibility documents and checksummed legacy artifact bundle.
The legacy bundle includes contact-v2 supporting provenance; this is not a
confirmed contact model or primary ECLS evidence. No PAE manuscript PDF or PAE
bibliography belongs to the ECLS active upload set.

## Later remote publication

After the local package is reviewed and remote publication is authorized, use
the active ECLS metadata and manifest to create the deposit. Record the actual
returned DOI/URL and verify the remotely downloaded files before changing any
status to published. Add only the verified DOI to the ECLS manuscript.
A later journal submission requires its own confirmation record.
