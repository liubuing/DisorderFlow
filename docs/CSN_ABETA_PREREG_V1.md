# CSN Aβ N-terminal RMSD preregistration v1

Status: **frozen_before_rmsd**, 2026-11-11. The machine-readable copy is
`results/csn_abeta_v1/preregistration.json`. No RMSD value was inspected
before this file was written.

## Question

Do independent antibody lineages that bind the Aβ N-terminal core `AEFRH`
share the same bound peptide backbone conformer?

## Cohort

Nine independent lineages, fourteen entries (ANARCI/IMGT calls in
`docs/CSN_ABETA_LINEAGES_20261111.md`):

| Lineage | Entries |
|---|---|
| L1 | 2IPU, 2R0W |
| L2 | 3BAE, 3BKJ |
| L3 | 3IFL, 3IFN |
| L4 | 3IFO |
| L5 | 3IFP |
| L6 | 4HIX, 4OJF, 4ONF |
| L7 | 5CSZ |
| L8 | 5W3P |
| L9 | 6CO3 |

Excluded: `4F37` (colicin-E7, not Aβ), `4M1C` (insulin-degrading enzyme, not
Aβ, coordinates absent), `4ONG` (peptide `DAEFR`, 5 aa; lacks the shared H6;
descriptive note only).

## Core

- Primary: **`AEFRH`** (5 residues). It starts at index 0 in
  `AEFRHDS`/`AEFRHD` entries and at index 1 in
  `DAEFRH`/`DAEFRHD`/`DAEFRHDS…` entries. The script locates it by sequence.
- Secondary: **`AEFRHD`** (6 residues), available in 11 entries; the
  D1-start trio 4HIX/4OJF/4ONF is excluded from this sensitivity analysis.

## RMSD

- Atom: Cα.
- Superposition: row-convention Kabsch, the corrected implementation used by
  CSN C0 (`scripts/csn_c0_conformer_sharing_v1.py:38`). A perfect rigid motion
  returns zero RMSD.
- Pair unit: chain pair. Multiple peptide chains in one entry are all used.
- Entry pair: mean over its chain pairs.
- Tiers: `same_bound_conformer` ≤0.5 Å; `related` 0.5–1.0 Å; `different`
  >1.0 Å.
- Same-entry NCS copies are the measurement floor; mean >0.5 Å invalidates
  the analysis.

## Decision rules

| Condition | Verdict |
|---|---|
| Mean same-entry NCS RMSD >0.5 Å | `measurement_floor_invalid` |
| NCS valid and cross-lineage entry-pair median ≤0.5 Å | `shared_bound_conformer` |
| Cross-lineage median in (0.5, 1.0] Å | `related_conformers` |
| Cross-lineage median >1.0 Å | `different_conformers` |

Bootstrap: resample the nine lineages with replacement, 10,000 draws, seed
20261111; report the 2.5–97.5 percentile interval of the cross-lineage
entry-pair median. It is descriptive; the cohort has nine lineages and a
five-residue primary core.

## Falsification

Independent lineages binding the same core in conformers >1.0 Å apart falsify
the shared-conformer hypothesis.

## Claim boundary

- Tests bound peptide backbone conformer sharing only.
- Does not test binding affinity, specificity, or free-state disorder.
- Does not modify the frozen CSN C0/C1/C2 results.
- A positive result generalizes the method to Aβ N-terminal recognition, not
  the tau PHF6* claim.
- The five-residue primary core is short; the six-residue secondary analysis
  is a sensitivity check, not a second confirmatory test.

## Frozen hashes

```text
lineage doc   f3879ca4121133c4707b21a78da3c897d21786fae3fcd65cbebe21e7481f3e95
entries csv   166a029ddd0e64fb50885e85c83469a12c188048f6c6e8208a3062e9262bdffe
RMSD script   0c9f09c1460a41cccd8cd0b8b8899447fd03b7feccc809828baef08ba485e985

> Pre-execution revision: the script's archive-member path handling for
> `3BKJ` was fixed before any RMSD value was computed. The cohort, core,
> tiers and decision rules are unchanged.
```
