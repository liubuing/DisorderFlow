# CSN Aβ N-terminal lineage calls (ANARCI/IMGT, 2026-11-11)

Read-only sequence analysis. **No RMSD has been computed at the time this
file was written.** The purpose is to freeze the cohort and lineage calls
before the conformer analysis.

## Method

- Input: `results/csn_abeta_v1/abeta_entries.csv` (SAbDab metadata + CIF
  sequences; 3IFL/4ONG/5CSZ fetched from RCSB).
- Numbering: ANARCI 2024.05.21 (bioconda), IMGT scheme, `--assign_germline`.
- VH and VL were numbered separately (`--restrict heavy` / `--restrict light`).
- Lineage call: same V/J germline family + same CDR3 length/identity, checked
  against VH/VL global identity. Humanized/murine versions of the same CDR3
  are merged into one clonotype.
- `identity_species` is the closest germline species, not the host species.

## Lineage table

| Lineage | Entries | VH germline | VH CDR3 | VL germline | VL CDR3 |
|---|---|---|---|---|---|
| **L1** | 2IPU, 2R0W | IGHV8-12*02 / IGHJ3 | `VRRAHTTGDWFAYVL` / `VRRAHNVGDWFAYVL` | IGKV1-117*01 / IGKJ5 | `FQGSHVPLT` |
| **L2** | 3BAE, 3BKJ | IGHV8-12*02 / IGHJ2 | `ARRGFYGVNHFDYRKYE` | IGKV1-117*01 / IGKJ5 | `FQASLVPLT` |
| **L3** | 3IFL, 3IFN | IGHV8-12*02 / IGHJ2 | `ARRTTTADYFAY` | IGKV1-117*01 / IGKJ5 | `FQSSHVPLT` |
| **L4** | 3IFO | IGHV8-12*01 / IGHJ4 | `VRRPITPVDAMDYVL` | IGKV1-117*01 / IGKJ5 | `FQGSHVPLT` |
| **L5** | 3IFP | IGHV8-12*02 / IGHJ2 | `ARRRIIYEDYFDYDV` | IGKV1-117*01 / IGKJ5 | `FQGSHVPLT` |
| **L6** | 4HIX, 4OJF, 4ONF, 4ONG | human IGHV3-23*03 / IGHJ4; mouse IGHV5-6*01 / IGHJ2 | `VRYDHYSGSSDY`; `VRYDSGSSDY` | IGKV1-135*01 | `WQGTHFPRT` |
| **L7** | 5CSZ | IGHV3-23*04 / IGHJ5 | `ARGKGYVRYFDV` | IGKV3-11*01 / IGKJ1 | `LQIYNMPIT` |
| **L8** | 5W3P | IGHV1-9*02 / IGHJ3 | `AREGSNNNALAY` | IGKV4-59*01 / IGKJ5 | `QNWRSSPT` |
| **L9** | 6CO3 | IGHV3-30-5*03 / IGHJ6 | `ARDRGIGPYYMDVARRG` | IGKV1-39*01 / IGKJ4 | `QQSYSTPLT` |

**Nine independent lineages.** L2 and L6 each contain entries that are the
same antibody in different crystal forms or species formats; they are merged
in the table. L1/L2/L3/L4/L5 share the same VL germline and CDR3 length but
have distinct VH CDR3s and VH germlines; they are separate lineages.

## Exclusions

| Entry | Reason |
|---|---|
| 4F37 | Antigen is colicin-E7 immunity protein; `DAEFRH` is a fusion/tag context. Same VH CDR3 as L2, but not an Aβ complex. |
| 4M1C | Antigen is insulin-degrading enzyme; `DAEFRH` is a substrate context. Coordinates are not in the local archive. |
| 4ONG | Peptide is `DAEFR` (5 residues); it lacks the shared H6 and cannot enter the primary 5-residue core analysis. It belongs to L6 and is retained as a descriptive note. |

## Included entries and peptide core

The 14 included entries carry one of two offset forms of the same core:

| Entry | Peptide sequence | Core `AEFRH` offset (0-based) |
|---|---|---|
| 2IPU | `AEFRHDS` | 0 |
| 2R0W | `AEFRHDS` | 0 |
| 3BAE | `AEFRHDS` | 0 |
| 3BKJ | `AEFRHDS` | 0 |
| 3IFL | `DAEFRHD` | 1 |
| 3IFN | `AEFRHD` | 0 |
| 3IFO | `AEFRHD` | 0 |
| 3IFP | `AEFRHD` | 0 |
| 4HIX | `DAEFRH` | 1 |
| 4OJF | `DAEFRH` | 1 |
| 4ONF | `DAEFRH` | 1 |
| 5CSZ | `DAEFRHDSGYE` / `DAEFRHDSGY` | 1 |
| 5W3P | `DAEFRHDSGYEVHHQK` | 1 |
| 6CO3 | `AEFRHD` | 0 |

Primary RMSD core: **`AEFRH` (5 residues)**. Secondary sensitivity core:
**`AEFRHD` (6 residues)**, available in 11 entries; the D1-start entries
(4HIX/4OJF/4ONF) are excluded from that secondary analysis.

## What this does not establish

- It does not show that the bound conformers are shared.
- It does not show that the peptides are disordered in the free state.
- It does not show that all entries bind the same chemical epitope; the
  offset forms and the longer Aβ constructs must be curated in the RMSD
  analysis.
- The lineage calls are sequence-level and depend on ANARCI germline
  assignment; a second numbering scheme should be run as a sensitivity check.
