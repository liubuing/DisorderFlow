# ECLS v1 current manuscript and saved-score reanalysis

This is a local author-review candidate, not a published record. The manuscript title is Peptide-context scoring of antibody loops: native recognition and the limits of candidate ranking. Scientific content has been rewritten around one question and two separate tests. Author declarations, manuscript/data license, immutable public code revision and remote DOI remain pending.

## Run numerical verification

Use Python 3.10 or later with NumPy installed. Extract this archive into a new directory and run:

```bash
python -B verify_ecls_saved_results.py
```

The verifier checks every manifest digest and recomputes record advantages, cluster means and 10000-resample bootstrap intervals with the original fixed seeds. It also reconstructs all 63 development-pool ranks and the three main ranking-gain intervals, then checks plotted values and the endpoint table against the evidence. It reads saved scores only. It has no model runner or network call. The frozen temporal final must not be rerun by the authors.

To redraw the two figures, install Matplotlib and run `python -B plot_ecls_narrative.py --archive-root .` in a separate working copy after verification. The plotting data include the fixed coordinate projection and all plotted cluster-level statistics. Redrawing may change file digests across library versions; the original manifest verifies the distributed files, not regenerated figures.

## Contents and evidence classes

- manuscript/: current main text, supplement, review PDFs and two figures in PNG, vector PDF and editable SVG formats.
- evidence/: portable saved ECLS scores, adaptation assignments, development ranking/calibration results, post hoc sensitivity, original frozen decision and original scope.
- protocols/: original fixed protocols, retained as provenance.
- SOURCE_PROVENANCE.json: original and derived digests, plus normalized path locations.
- primary_results.csv: five endpoints used in the narrative.
- figure_source_data.json and plot_ecls_narrative.py: the data and code for both main figures, including the identifier-selected structural illustration.
- MANIFEST.json: exact included file set and SHA-256 digests.

The primary result is 31 structures in 15 antigen clusters: mean advantage 0.172281, CI [0.059156, 0.291920], 12/15 positive clusters. The 46-cluster adaptation analysis is exposed development. The seven-cluster universal-ranking gate failed; calibration is exploratory. Samples from these analyses must not be added together.

## Transformation and reproducibility limits

Original local result files remain unchanged. Derived public copies normalize machine-specific execution paths while preserving numerical values and scientific strings. The original final decision refers to the original result digest; the portable derivative intentionally has a different digest, documented in SOURCE_PROVENANCE.json. The verifier checks that provenance link and the package's actual file hashes.

This package provides numerical reanalysis, not end-to-end model inference reproduction. Raw structural databases, third-party weights, historical contact-v2 artifacts, earlier manuscript ZIPs and the separate PAE paper are excluded. Historical commands and configs are provenance, not instructions to rerun the terminal final. The older working title in scope_original.yml is preserved rather than rewriting a frozen record.

No experimental binding, affinity or therapeutic claim is made. See LICENSE_SCOPE.md for the unresolved manuscript/data license before public publication.
