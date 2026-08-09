# ECLS v1 Reproducibility Release

## Scope

This release supports the bounded ECLS native-versus-composition-shuffle
structural sequence-scoring paper. It does not release a validated antibody
design, binding, affinity, contact-hotspot, or therapeutic model.

## Source Layer

Git contains source code, frozen protocols, tests, lightweight result summaries,
and SHA-256 registries. Run:

```bash
pip install -e ".[publication,dev]"
python scripts/validate_release_lineage.py --source-only
pytest -q tests/test_release_lineage.py tests/test_h3_ecls_statistics.py
```

## Large Artifact Layer

`artifact_bundle_manifest.json` lists every checkpoint, LMDB, structure, and
reviewer archive required by the bundled contact-v2 supporting analysis. Each
file and the deterministic ZIP have independent SHA-256 digests.

Remote publication is a release blocker until the manifest status is
`published` and contains an immutable repository revision and URL. While the
network blocker remains, the local bundle can be restored with:

```bash
python scripts/fetch_release_artifacts.py \
  --bundle dist/disorderflow-ecls-v1-artifacts.zip
```

Verify an existing installation without extracting:

```bash
python scripts/fetch_release_artifacts.py --verify-only
python scripts/validate_release_lineage.py
```

No manuscript or software release should claim remote reproducibility while the
bundle manifest remains `blocked_pending_remote_upload`.
