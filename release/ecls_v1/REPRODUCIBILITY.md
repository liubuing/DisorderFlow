# ECLS v1 current reproducibility boundary

The active manuscript is publication/MANUSCRIPT_DRAFT.md and the active reviewer archive is dist/disorderflow-ecls-v1-reviewer-package.zip. This is a numerical reanalysis package built from saved evidence; it does not claim end-to-end model inference reproducibility. The old contact-v2 artifact bundle remains historical and is excluded from active staging.

```bash
python scripts/render_ecls_submission.py
python scripts/prepare_ecls_submission.py
python scripts/build_zenodo_upload.py
python scripts/validate_publication_alignment.py
python scripts/validate_release_lineage.py --source-only
```

Extract the current archive into a new directory and run python -B verify_ecls_saved_results.py there. Python 3.10+ and NumPy are required. The verifier checks file digests, composition controls, record arithmetic, cluster assignments and the original fixed bootstrap calculations, with no model forward pass.

The same verifier reconstructs the 63 development-pool native ranks and all three ranking intervals in the main figures, and compares figure values and the endpoint table with saved evidence. To redraw figures, install Matplotlib and run python -B plot_ecls_narrative.py --archive-root . in a separate extracted working copy after digest verification. Editable SVG, vector PDF and high-resolution PNG outputs are included. Regenerated files may have different hashes across library versions.

Original frozen source files retain their hashes. Portable JSON copies normalize machine-specific execution paths; the separate original and public digests and transformation locations are recorded. Frozen final decision and scope are kept unchanged, including historical title and result references.

The active upload manifest enumerates only the current ECLS package and readable manuscript/provenance assets. The old artifact_bundle_manifest.json describes a separate historical bundle and its unresolved remote upload; it is not evidence that the new archive has been published. Current author declarations and publication state are in submission_metadata.json. Only a verified remote record and downloaded checksums can establish remote publication.
