# DisorderFlow — ECLS Structural Sequence Scoring

DisorderFlow is a research repository for antibody–peptide sequence scoring
and related BFN design experiments. **The current primary manuscript and
release are ECLS v1**, targeting a computational structural sequence-scoring
paper in Bioinformatics. The project is in manuscript consolidation and local
release preparation; journal submission and remote publication are unconfirmed.

## Primary paper: ECLS

Epitope-conditioned likelihood shift compares the same H3 sequence in two
coordinate contexts:

```text
ECLS = H3 NLL(complex backbone)
     - H3 NLL(peptide-stripped, complex-derived Fab backbone)
advantage = mean ECLS(composition-matched shuffled H3) - ECLS(native H3)
```

The frozen primary claim is that deposited native H3 sequences have a more
favorable contrast than composition-matched shuffles on the temporal structure
panel. It is not a binding/affinity prediction or a general generated-candidate
reranking claim.

| Frozen temporal-final endpoint | Result |
|---|---:|
| Structures / antigen-cluster inference units | 31 / 15 |
| Mean ECLS advantage | 0.172281 |
| Cluster-bootstrap 95% CI | [0.059156, 0.291920] |
| Positive antigen clusters | 12 / 15 |

The result passed the prespecified internal gates. This does not mean journal
acceptance. The final evaluation is terminal and must not be rerun or retuned.

Current entry points:

- [Publication protocol](PUBLICATION_PROTOCOL.md): claim, exclusions and target venue.
- [Frozen ECLS scope](publication/ECLS_SCOPE_FREEZE.yml): immutable evidence contract.
- [Publication map](docs/PUBLICATION_MAP.md): primary and separate research branches.
- [ECLS reproducibility](release/ecls_v1/REPRODUCIBILITY.md): local verification and release status.
- Local manuscript: `publication/MANUSCRIPT_DRAFT.md` (unpublished; not tracked in Git).
- Local upload preparation: `release/zenodo_v1/DEPOSIT_INSTRUCTIONS.md`.

## Other research branches

| Branch | Status | Relationship to ECLS v1 |
|---|---|---|
| PAE surrogate | Separate revision; input-provenance and baseline issues audited; complex-model advantage not established | Not the ECLS manuscript or its primary evidence |
| Successor/contact-v2 | Frozen development candidate awaiting at least 12 independent components | Supplementary/development provenance only |
| Disorder-aware BFN design | Generation and scoring infrastructure; no experimentally validated binder | Research platform, not evidence of successful antibody design |
| T1/T2 and multiscaffold design | Bounded exploratory or negative computational findings | Limitations/supporting history only |

PAE entry points are the [v4 report](docs/PAE_SURROGATE_REVISION_V4.md) and local
`publication/af2_interface_pae_surrogate_manuscript_v4.md`. Earlier PAE contracts
are historical. The old evaluation consumed AF2 output coordinates; its results
do not establish pre-AF2 screening. That branch's limitations do not invalidate
the separate frozen ECLS result.

## Software and verification

`disorderflow/` holds the core models and datasets, `modules/` the runtime
interfaces, `scripts/` the research workflows, and `tests/` the scientific
contracts. Related fixed-backbone BFN sequence generation remains available via
`modules.bfn_loader.run_bfn_design`; feature availability does not establish
binding, specificity or efficacy.

```bash
python scripts/validate_release_lineage.py --source-only
python scripts/validate_publication_alignment.py
```

The local publication bundle has not been verified as remotely published. No
wet-lab binding measurements are claimed. License: [MIT](LICENSE.md).
