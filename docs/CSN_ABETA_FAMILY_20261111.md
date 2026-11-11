# CSN Aβ N-terminal family: first candidate with ≥5 independent lineages

Date: 2026-11-11. Read-only sequence/lineage analysis. **This does not change
the frozen CSN C0/C1/C2 results.** It identifies a new expansion cohort that
must be preregistered before any conformer analysis.

## 1. Why Aβ, not tau

A full-PDB sequence search (`pdb_seqres.txt.gz`, 1,183,790 sequences) plus a
SAbDab intersection gave:

| Target | Peptide | Full-PDB PDBs | SAbDab antibody entries | Independent lineages |
|---|---|---:|---:|---|
| tau PHF6* | `GSVQIVYKPK` | 4 | 4 | **1** (7QCQ/8OP0/8OPI/8PII, all VHH Z70) |
| α-syn C-term | `DYEPEA` | 237 (mostly α-syn itself) | 1 (2X6M) | **1** |
| Syt1 | `KEDAFSKLKEKFMNELH` | 6 (mostly Syt1 complexes) | 1 (8B8I) | **1** |
| **Aβ N-term** | **`DAEFRH`** | **164** | **17** | **≥5 candidate clusters** |

The tau family is a single VHH Z70 lineage even after adding 7QCQ (95.6–98.2%
VHH identity to 8OP0/8OPI/8PII). The Aβ N-terminal family is the only local
family with enough independent antibodies to reach the CSN confirmatory
threshold.

## 2. Aβ entries in SAbDab

Seventeen SAbDab entries contain `DAEFRH` (2006–2018):

```
2IPU 2R0W 3BAE 3BKJ 3IFL 3IFN 3IFO 3IFP 4F37 4HIX
4M1C 4OJF 4ONF 4ONG 5CSZ 5W3P 6CO3
```

Two are **not** Aβ-antibody complexes and must be excluded:

- `4F37`: antigen is colicin-E7 immunity protein; `DAEFRH` is a fusion/tag
  context, not the antibody epitope.
- `4M1C`: antigen is insulin-degrading enzyme; `DAEFRH` is a substrate
  context, not the antibody epitope.

Three Aβ entries are absent from the local archive and were fetched from RCSB
on 2026-11-11: `3IFL`, `4ONG`, `5CSZ` (`data/rcsb_abeta_extra/`). They are
full Fab structures (H/L ~214 aa with constant domains), so their VH/VL were
truncated to the first 125/115 residues for this first-pass comparison. An
ANARCI/IMGT run is still required.

After excluding 4F37 and 4M1C, **15 Aβ-relevant entries** enter the comparison
(12 archive-covered + 3 fetched).

## 3. Lineage clustering (VH, global identity)

Using the longest antibody chain (VH) and global alignment identity:

| Threshold | Clusters among 15 Aβ entries |
|---:|---:|
| ≥95% | 13 |
| ≥90% | **10** |
| ≥85% | 7 |

At 90% the clusters are:

```
{2IPU, 2R0W}
{3BAE, 3BKJ}
{3IFL, 3IFN}
{4HIX, 4OJF}
{4ONF, 4ONG}
{3IFO}
{3IFP}
{5CSZ}
{5W3P}
{6CO3}
```

The `3IF*` entries are from the same 2009 study and may merge under manual
curation; at 85% the family still has 7 clusters. This is a **sequence-level
lower bound**, not an ANARCI/germline call. Some pairs (for example
`2IPU`/`2R0W`, `3BAE`/`3BKJ`, `4HIX`/`4OJF`) are the same antibody or crystal
variants and should be merged.

## 4. What this establishes

- The Aβ N-terminal family is the first local family with **≥5 independent
  VH clusters** (10 at 90%, 7 at 85%), satisfying the CSN expansion
  protocol's confirmatory floor.
- The tau and α-syn families do not reach that floor locally; a full RCSB
  search is still required for them.
- The result is **lineage evidence only**. It does not establish that the
  bound peptide conformer is shared, that the antibodies bind the same
  chemical epitope, or that the peptide is disordered in the free state.

## 5. Required before a confirmatory claim

1. Verify the fetched `3IFL`, `4ONG`, `5CSZ` chain identity and peptide
   chemistry; `4ONG` does not expose `DAEFRH` as a 5–50-mer in the first
   parse, so its peptide may be modified or clipped.
2. Run ANARCI/IMGT numbering and cluster by species, germline, VH/VL pairing
   and variable-domain identity; merge same-study variants.
3. Manually curate construct boundaries, tags, fusion partners and crystal
   forms.
4. Freeze the candidate list and lineage calls **before** any RMSD scoring.
5. Run the preregistered peptide-backbone RMSD analysis with the NCS floor
   (same-entry copies ≤0.5 Å); tiers ≤0.5 / 0.5–1.0 / >1.0 Å.
6. Report the Aβ result as a separate CSN expansion, not as a modification of
   the frozen C0/C1/C2 results.

## 6. Boundary

`DAEFRH` is a short N-terminal Aβ peptide. Some entries bind longer Aβ
constructs (`DAEFRHDSGYEVHHQK`, fibrils), so the shared core must be defined
before RMSD. The family's biological meaning is Aβ N-terminal recognition, not
the tau PHF6* question that motivated CSN v1. A positive result would
generalize the **method** (independent-lineage conformer sharing), not the
original tau claim.
