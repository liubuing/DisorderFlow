# DisorderFlow Data Sources

Verified on 2026-07-18. Raw downloads live under `data/` and are excluded from Git.

## External evaluation

- CAID2 Disorder-NOX: `https://caid.idpcentral.org/assets/sections/challenge/static/references/2/disorder_nox.fasta`
- CAID2 Disorder-PDB: `https://caid.idpcentral.org/assets/sections/challenge/static/references/2/disorder_pdb.fasta`
- CAID2 Binding: `https://caid.idpcentral.org/assets/sections/challenge/static/references/2/binding.fasta`
- CAID2 Linker: `https://caid.idpcentral.org/assets/sections/challenge/static/references/2/linker.fasta`
- CAID project license: CC BY 3.0. CAID2 paper: DOI `10.1002/prot.26582`.

These are blind external reference sets. They must never be merged into training data.
`-` labels are undefined residues and are excluded from metric calculations.

## Disorder annotations

- DisProt current release: `https://disprot.org/download`
- Structural-state consensus TSV: `https://disprot.org/api/search?release=current&format=tsv&namespace=structural_state&get_consensus=true&show_ambiguous=false&show_obsolete=false`
- DisProt license: CC BY 4.0.
- MobiDB API: `https://mobidb.org/help#swagger`
- MobiDB license: CC BY 4.0.

Use curated experimental regions as labels. Predicted MobiDB-Lite and AlphaFold tracks
are auxiliary features, not experimental ground truth. Use UniRef50/90/100 identifiers
for homology-disjoint splitting.

CAID2 overlaps the current DisProt release: 307 of 348 CAID2 targets are present.
Excluding those targets and their available UniRef50 clusters removes 328 DisProt
records and leaves 3,009 training candidates. This exclusion is mandatory before a
CAID2 evaluation.

## Antibody-antigen structures

- SAbDab2 API: `https://sabdab.opig.stats.ox.ac.uk/api/docs`
- Summary CSV: `https://sabdab.opig.stats.ox.ac.uk/api/download/all-summary`
- ML dataset record: `https://zenodo.org/records/20083995`
- Version DOI: `10.5281/zenodo.20083995`, version 0.1.0, CC BY 4.0.
- `splits.tar.gz`: 876,381,859 bytes; MD5 `0dbb4cc499e9eb77f14008b232f2c38c`.

The local archive is resumable. Run `bash scripts/download_sabdab2_ml.sh`; use
`--wait-for-v5` to avoid competing with the active AF2 build. The script rejects an
incorrect byte count or MD5 and validates the tar index before reporting success.

For antigen-aware work use `abag_split.csv`, which isolates both antibody and antigen
sequence similarity. Do not use the antibody-only `ab_split.csv` for binding claims.

## Related public checkpoint

- Repository: `https://huggingface.co/YueHuLab/AntibodyDesignBFN`
- Candidate: `best.pt`, 112,993,226 bytes, SHA256
  `f3e254773eca35bef98c3875544d0c114ef86547579e56de8f470b03ff977843`.

This checkpoint is publicly associated with AntibodyDesignBFN, not explicitly with
DisorderFlow. It may be used only after architecture/state-dict compatibility checks;
its provenance must not be described as a DisorderFlow checkpoint.
