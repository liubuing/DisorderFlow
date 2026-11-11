# CSN Aβ N-terminal construct curation (2026-11-11)

Read-only curation of the 17 SAbDab entries that contain `DAEFRH`. Antibody
names are taken from the PDB titles; peptide sequences are from the CIF
chains. No RMSD is recomputed here.

## Included entries

| PDB | Antibody / construct | Peptide | Register | Lineage | Note |
|---|---|---|---|---|---|
| 2IPU | PFA1 Fab + Aβ 1-8 | `AEFRHDS` | A2 | L1 | Fab–peptide complex |
| 2R0W | PFA2 Fab + Aβ 1-8 | `AEFRHDS` | A2 | L1 | Same lineage as 2IPU, different Fab |
| 3BAE | WO2 Fab + Aβ N-terminal | `AEFRHDS` | A2 | L2 | — |
| 3BKJ | WO2 Fab + Aβ N-terminal | `AEFRHDS` | A2 | L2 | Same antibody as 3BAE |
| 3IFL | 12A11 Fab + Aβ 1-7 | `DAEFRHD` | D1 | L3 | Same antibody as 3IFN |
| 3IFN | 12A11 Fab + Aβ 1-40 | `AEFRHD` | A2 | L3 | Same antibody, longer antigen; only A2-D7 resolved |
| 3IFO | 10D5 Fab + Aβ 1-7 | `AEFRHD` | A2 | L4 | Same 2009 study as 3IFL/3IFN/3IFP, different antibody |
| 3IFP | 12B4 Fab + Aβ 1-7 | `AEFRHD` | A2 | L5 | Same study, different antibody |
| 4HIX | Humanized 3D6 Fab + Aβ | `DAEFRH` | D1 | L6 | 3D6 family |
| 4OJF | Humanized 3D6 Fab + Aβ 1-8 | `DAEFRH` | D1 | L6 | Same antibody as 4HIX |
| 4ONF | 3D6 Fab + Aβ 1-7 | `DAEFRH` | D1 | L6 | Murine 3D6; same CDR3 as 4HIX/4OJF |
| 5CSZ | Gantenerumab Fab + Aβ 1-11 | `DAEFRHDSGYE` / `DAEFRHDSGY` | D1 | L7 | Fibril-binding Fab |
| 5W3P | C706 Fab + Aβ 1-16 | `DAEFRHDSGYEVHHQK` | D1 | L8 | Longer peptide |
| 6CO3 | Aducanumab Fab + Aβ | `AEFRHD` | A2 | L9 | A2-D7 core |

## Excluded entries

| PDB | Reason |
|---|---|
| 4F37 | Antigen is colicin-E7 immunity protein; `DAEFRH` is a fusion/tag context. Same VH CDR3 as L2, but not an Aβ complex. |
| 4M1C | Antigen is insulin-degrading enzyme; `DAEFRH` is a substrate context. Coordinates are not in the local archive. |
| 4ONG | 3D6 Fab + Aβ 1-40; only `DAEFR` (5 aa) is resolved, so it lacks the shared His6 and cannot enter the 5-residue `AEFRH` core. Retained as a descriptive L6 note. |

## Curation decisions

- **L3 is the only lineage represented in both registers**: 3IFN is A2-start,
  3IFL is D1-start. They are the same antibody (12A11) and differ by only
  0.274 Å on the shared `AEFRHD` core, which is the internal register control.
- **3IFO and 3IFP are separate lineages**, not 12A11 variants. ANARCI assigns
  different VH germlines/CDR3s (`IGHV8-12*01`/`VRRPITPVDAMDYVL` for 3IFO;
  `IGHV8-12*02`/`ARRRIIYEDYFDYDV` for 3IFP).
- **4HIX, 4OJF and 4ONF are one clonotype** (3D6 family): they share the VH
  CDR3 `VRYDHYSGSSDY` and VL CDR3 `WQGTHFPRT`; the humanized and murine
  frameworks are merged into L6.
- **4ONG is L6 but excluded from the primary core** because its resolved
  peptide is only `DAEFR`. It is not evidence for a different conformation;
  it is a construct limitation.
- **5CSZ and 5W3P are independent lineages** (gantenerumab and C706) and are
  retained despite longer peptides; their core `AEFRH` is resolved.
- **6CO3 (aducanumab)** is retained at the A2-start register; it is the
  boundary lineage in the RMSD matrix.

## Code N-terminal boundary

`AEFRH` is present in all included entries. The two offset forms are:

- A2-start: `AEFRHDS` (2IPU, 2R0W, 3BAE, 3BKJ) or `AEFRHD` (3IFN, 3IFO,
  3IFP, 6CO3);
- D1-start: `DAEFRH` (4HIX, 4OJF, 4ONF), `DAEFRHD` (3IFL),
  `DAEFRHDSGYE`/`DAEFRHDSGY` (5CSZ), `DAEFRHDSGYEVHHQK` (5W3P).

The 5-residue core is therefore the maximal shared unit. Adding D1 or D7
introduces terminal flexibility and inflates the 6-mer RMSD, as shown in the
register-aware sensitivity analysis.
