# CSN local peptide scan (2026-11-11)

Read-only scan of the local SAbDab2 structure archive for the CSN expansion
protocol. No model was run.

## Method

- Input: `data/sabdab2_current/splits.tar.gz` (15,641 CIF members, 8,641
  unique PDB IDs).
- Parser: `gemmi` 0.7.5, streaming; model 0 only.
- Antibody chains: `H` and `L` (the archive renames the variable domains).
- Candidate peptide: any other chain with 5–50 standard amino-acid residues;
  waters, ions and other heteroatoms are dropped; chains with >20% `X` are
  skipped.
- Output: `results/csn_expansion_v1/peptide_inventory.csv` and
  `peptide_inventory_manifest.json` (SHA-256 recorded).

## Counts

| Metric | Value |
|---|---:|
| CIF members scanned | 15,641 |
| Peptide rows | 483 |
| Unique PDBs | 475 |
| Unique sequences | 394 |
| Sequences in ≥2 PDBs | 47 |
| Max PDBs per sequence | 14 |
| Rows in repeated sequences | 138 |
| Parse failures | 0 |

## Repeated sequences (top of the local list)

| Sequence | PDBs | Likely identity | Comment |
|---|---:|---|---|
| `AVGIGAVF` | 14 | HIV gp41 fusion peptide | Same epitope across many entries; lineage independence unknown |
| `NPDPNANPNVDPN` | 5 | Circumsporozoite protein repeat | Repeat antigen, not an IDP |
| `NPDPNANPNVDPNA` | 5 | Circumsporozoite protein repeat | Same family |
| `AEFRHD` / `AEFRHDS` | 4 / 4 | Amyloid-β N-terminal variants | Candidate CSN-like family |
| `DAEFRH` | 4 | Amyloid-β N-terminal | Candidate CSN-like family (4hix, 4ojf, 4onf, 5w3p) |
| `GIVEQCCTSICSLYQLENYCN` | 4 | Insulin A chain | Hormone, not an IDP |
| `LLELDKWASLW` | 4 | HIV gp41 | Peptide epitope |
| `NWFDITNWLWYIKKK` | 4 | HIV gp41/gp120 | Peptide epitope |
| `QLINTNGSWHIN` | 4 | HIV gp120 | Peptide epitope |
| `CGCGCGCGCGCG` | 2 | Synthetic Cys repeat | Not biological |

These are **sequence repeats, not independent lineages**. The same antibody,
the same construct or the same crystal form can produce multiple PDBs.

## CSN-specific entries missing from this archive

| Entry | Peptide | Status |
|---|---|---|
| 8OP0 / 8OPI / 8PII | tau PHF6* `GSVQIVYKPK` | **Absent** from the local archive |
| 2X6M | α-synuclein C-terminal `DYEPEA` | **Absent** |
| 8B8I | Syt1 `KEDAFSKLKEKFMNELH` | Present only as a 14-mer `KEDAFSKLKEKFMN` |
| 4HIX | Aβ `DAEFRH` | Present |

The archive is therefore a **local lower bound**, not the CSN coordinate
source. The CSN C0 analysis used separate local coordinates; those entries
must be fetched from RCSB for any expansion that includes tau/α-syn.

## Extended local scan (additional CIF directories)

The archive is not the only local source. A second scan covered
`data/sabdab2_abag_project_pool_v2/cif`, `data/multiscaffold_confirmatory_v1/cif`,
`results/flex_a1_scaffold_audit_v1/cif` and
`data/peptide_h3_temporal_split_v3/cif`:

| Metric | Value |
|---|---:|
| CIF files scanned | 1,417 |
| Peptide rows | 935 |
| Parse failures | 0 |

Key lineage checks:

| Entry | Peptide | Antibody comparison | Lineage call |
|---|---|---|---|
| 7QCQ | `GSVQIVYKPK` | VHH 95.6–98.2% identical to 8OP0/8OPI/8PII | **Same VHH Z70 lineage**, not independent |
| 8OP0 / 8OPI / 8PII | `GSVQIVYKPK` | 97.3–98.2% among themselves | Same lineage |
| 4HIX / 4OJF | `DAEFRH` | H/L 100% | Same antibody |
| 4ONF | `DAEFRH` | H 86.6%, L 88.5% vs 4HIX | Close lineage, not independent by the <90% rule |
| 5W3P | `DAEFRHDSGYEVHHQK` | H 47.5%, L 48.7% vs 4HIX | **Independent lineage** |
| 2X6M | `DYEPEA` | single entry | No second lineage |
| 8B8I | `KEDAFSKLKEKFMNELH` | single entry (8 copies) | No second lineage |

**Local conclusion:** the tau PHF6* family remains single-lineage after adding
7QCQ; the Aβ N-terminal family has two independent lineages (4HIX/4OJF/4ONF
vs 5W3P), which is descriptive only. No local family reaches the ≥5-lineage
confirmatory threshold. A full RCSB/PDB search is still required for tau and
α-syn; the local archive cannot answer the question.

## What to do next

1. Run full RCSB/PDB sequence searches for `GSVQIVYKPK`, `DYEPEA` and the
   Syt1 peptide; the local sources do not contain a second independent
   lineage for any of them.
2. Expand the Aβ N-terminal family with a full PDB search. It currently has
   two independent lineages (4HIX/4OJF/4ONF vs 5W3P), so it is descriptive
   until more lineages are found.
3. Run ANARCI/IMGT numbering on the candidate families before calling
   lineage independence; the current calls use global sequence identity and
   are a first pass.
4. Apply the stop rules in `docs/CSN_EXPANSION_SEARCH_PROTOCOL_V1.md`: 0
   families → `coordinate_starvation`; 1–2 → descriptive; ≥5 → confirmatory.

## Boundary

This scan counts peptide-like chains. It does not establish that a peptide is
an IDP epitope, that the antibody lineage is independent, or that the bound
conformer is shared. Those require parent mapping, lineage clustering and the
preregistered RMSD analysis.
