# Data inventory for the controlled-replication model and CSN expansion

Date: 2026-11-11. Read-only inventory of the local SAbDab2 snapshot and the
structure archive. No model was run.

## 1. SAbDab2 summary table

Source: `data/sabdab2_current/all-summary.csv` (12 MB, 21,741 rows, 45
columns). Deposition dates run from 1984-07-09 to 2026-06-16.

| Slice | Rows | Unique PDBs |
|---|---:|---:|
| All | 21,741 | 11,360 |
| Deposition ≤ 2021-08-02 | 10,859 | 5,695 |
| Deposition > 2021-08-02 | 10,882 | 5,665 |

Pre-cutoff detail (≤ 2021-08-02):

| Filter | Rows | Unique PDBs |
|---|---:|---:|
| Peptide antigen (`antigen_type` contains `PEPTIDE`) | 1,062 | 628 |
| Has an antigen chain | 9,002 | 4,795 |
| No light chain (VHH/single-domain) | 1,828 | 946 |
| Resolution ≤ 3.5 Å | 8,893 | — |
| Method `XRAY` | 8,626 | — |

Post-cutoff detail (> 2021-08-02):

| Filter | Rows | Unique PDBs |
|---|---:|---:|
| Peptide antigen | 550 | 307 |
| Has an antigen chain | 10,391 | 5,419 |
| No light chain | 2,710 | 1,457 |
| Resolution ≤ 3.5 Å | 8,714 | — |

## 2. Structure archive

| Asset | Count | Note |
|---|---:|---|
| `data/sabdab2_current/splits.tar.gz` | 15,641 CIF members, 8,641 unique PDB IDs | SAbDab2 structure archive, 836 MB compressed |
| `data/sabdab2_abag_project_pool_v2/cif/` | 1,146 | Curated antibody–antigen project pool |
| `data/peptide_h3_temporal_split_v3/cif/` | 103 | Temporal candidates; 31 sealed final structures |
| `data/sabdab2_abag_project_split_v2/` | 1,146 LMDB records | adaptation 728 / dev 187 / final 230 |

### Archive coverage caveat

The archive is **not** a complete mirror of the pre-2021 summary:

| Slice | Unique PDBs |
|---|---:|
| Pre-2021 summary | 5,695 |
| Pre-2021 PDBs present in the archive | **4,530 (79.5%)** |
| Pre-2021 PDBs absent from the archive | **1,165** |

Four archive PDB IDs are absent from the summary (newer entries). For M1, the
4,530 covered PDBs are sufficient for a first fine-tuning arm; the 1,165
missing entries can be fetched from RCSB later if the split needs them. For
CSN, the archive scan is a **local lower bound**, not a full-PDB census.

## 3. Implications

### Controlled-replication model (M1)

The pre-2021 slice is large enough:

- 5,695 unique PDBs, 10,859 instance rows;
- 8,893 rows at ≤3.5 Å;
- 628 peptide-antigen PDBs if the arm is restricted to antibody–peptide
  complexes.

Recommended training set: **all pre-2021 antibody structures with a heavy
chain and resolution ≤3.5 Å** (not only peptide complexes), because the model
must learn antibody sequence–structure grammar. The ECLS evaluation stays on
the frozen post-2021 panel. The peptide-only arm is a sensitivity analysis.

### CSN expansion

The local peptide-antigen slice is 628 pre-cutoff + 307 post-cutoff PDBs.
This is enough to start sequence clustering and lineage curation, but the
summary table has **no peptide sequence column**; sequences must be read from
the CIF archive. The full `splits.tar.gz` is the right input, not the 1,146
CIF project pool (which was selected for a different purpose).

The expected yield of independent lineages is unknown and must not be assumed.
The stop rules in `docs/CSN_EXPANSION_SEARCH_PROTOCOL_V1.md` apply: 0
families → `coordinate_starvation`; 1–2 → descriptive only; ≥5 → confirmatory.

## 4. What this does not establish

- It does not create a train/validation split.
- It does not verify chain pairing or structure quality beyond the summary
  fields.
- It does not establish that the pre-2021 set is free of the ECLS panel; the
  panel is post-2021 by construction, but an exact PDB/H3 overlap check is
  still required.
- It does not find independent antibody lineages; that is the CSN expansion
  work.
