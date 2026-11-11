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

## What to do next

1. Cluster each repeated sequence by antibody lineage (ANARCI/IMGT, species,
   germline, VH/VL pairing, variable-domain identity).
2. Manually curate the Aβ family (`DAEFRH`, 4 PDBs) as a test case; if it has
   ≥2 independent lineages, it is the first confirmatory candidate.
3. Fetch 8OP0/8OPI/8PII and 2X6M coordinates from RCSB (not present locally)
   and add them to the expansion inventory.
4. Apply the stop rules in `docs/CSN_EXPANSION_SEARCH_PROTOCOL_V1.md`: 0
   families → `coordinate_starvation`; 1–2 → descriptive; ≥5 → confirmatory.

## Boundary

This scan counts peptide-like chains. It does not establish that a peptide is
an IDP epitope, that the antibody lineage is independent, or that the bound
conformer is shared. Those require parent mapping, lineage clustering and the
preregistered RMSD analysis.
