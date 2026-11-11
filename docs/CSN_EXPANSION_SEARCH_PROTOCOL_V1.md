# CSN expansion search protocol v1 (proposal)

Status: **proposal, not executed.** Goal: turn the CSN C0 observation
(`single_or_few_family_descriptive`) into a testable cohort by finding
**independent antibody lineages** that bind the same or overlapping disordered
peptide epitope.

The current evidence is three deposits of one VHH Z70 lineage (8OP0/8OPI/8PII).
That is not independence, and 0.1 Å RMSD may partly reflect the same
construct/crystal rather than convergence. A confirmatory claim needs
independent lineages, not more deposits of the same lineage.

## 1. Target cohort

A family is eligible when:

1. The bound peptide is 5–50 residues and maps to an intrinsically disordered
   region of a parent protein (or is a validated IDP-derived epitope).
2. At least **two independent antibody lineages** bind the same or overlapping
   peptide (≥80% sequence identity over the shared core).
3. Each component has a deposited experimental structure (X-ray, cryo-EM,
   NMR) at ≤3.5 Å (X-ray/cryo-EM) or a deposited ensemble (NMR).
4. The antibody is a conventional Fab/scFv or a VHH/nanobody; lineage
   independence is established by species, germline, VH/VL pairing, or
   sequence identity <90% over the variable domain.
5. Construct tags, cleavage remnants and carrier proteins are excluded from
   the peptide definition.

Excluded: the same lineage with point mutations, the same Fab in different
crystal forms, NCS copies, synthetic tag-only peptides, and predicted
complexes.

## 2. Sources and search order

| Order | Source | What to query | Notes |
|---|---|---|---|
| 1 | Local SAbDab2 archive | All antibody–peptide complexes; peptide sequence; PDB entry; heavy/light chains | Primary; already used by FLEX |
| 2 | SAbDab current snapshot | 21,142 summary rows; peptide antigens; affinity annotations | Metadata-level curation |
| 3 | RCSB PDB Search API | mmCIF/SEQRES polymer entities with peptide length 5–50 in antibody complexes | Full PDB, not just SAbDab |
| 4 | AbMPNN dataset (`paulvstrashnov/ab-dataset`) | 3,391 SAbDab experimental structures; metadata CSV with heavy/light sequences | Cross-check and expand |
| 5 | AntiFold / InverseFoldingEvaluation repos | PDB lists and antibody-antigen examples | Method reference and possible structures |
| 6 | HuggingFace datasets | Search terms: antibody structure, SAbDab, peptide antigen, inverse folding | API was unreachable on 2026-11-11; retry before the freeze |
| 7 | GitHub | Search: SAbDab curation, antibody-antigen dataset, peptide epitope structures | Use repository metadata and README, not bulk scraping |
| 8 | Literature / IEDB | Papers reporting the same epitope bound by different antibodies | Use to seed PDB searches, not as structural evidence |

## 3. Procedure

### Step 1 — Peptide inventory

Extract every peptide chain (5–50 aa) from antibody-containing structures.
Record: PDB ID, chain, sequence, length, parent accession (if any), resolution,
method, deposition date.

### Step 2 — Peptide clustering

Cluster exact and near-identical peptides:

- exact match;
- ≥80% identity over the shared core with no more than one gap;
- same parent protein (UniProt exact substring or ≥90% identity).

Map each peptide to its parent with a local Swiss-Prot/isoform exact scan (the
CSN C1 pipeline already has this; reuse `csn_peptidematch_local_v1.py`).

### Step 3 — Lineage clustering

Number antibodies with ANARCI/IMGT. Cluster by:

- species and germline (V gene, J gene);
- VH/VL pairing;
- CDR-H3 identity and length;
- variable-domain identity <90% to call a new lineage.

A family with two entries from the same lineage is **not** independent.

### Step 4 — Manual curation

For every candidate family, inspect the paper/PDB header and record:

- construct boundaries, tags, carriers;
- crystal form and NCS copies;
- whether the peptide is the same chemical entity;
- whether the antibody was raised against the same antigen.

### Step 5 — Freeze the candidate list

Before any RMSD scoring, freeze:

- the candidate list;
- the inclusion/exclusion decisions;
- the peptide-family definition;
- the lineage-independence calls;
- the RMSD tiers and the NCS floor.

## 4. Power and decision rules

- **Confirmatory threshold**: ≥5 independent families with ≥2 lineages each.
  With fewer, the result stays descriptive.
- **Effect**: pairwise Cα RMSD of the bound peptide across independent
  lineages, compared with the same-entry NCS floor.
- **Tiers** (same as CSN C0): `same_bound_conformer` ≤0.5 Å; `related` 0.5–1.0 Å;
  `different` >1.0 Å.
- **Floor control**: same-entry NCS copies must be ≤0.5 Å; otherwise the
  measurement floor is invalid and the analysis is downgraded.
- **Falsification**: independent lineages binding the same epitope in
  different conformers (>1.0 Å) falsifies the shared-conformer hypothesis.
- **No AF3 inflation**: predicted complexes do not enter the cohort. They may
  be used only as a hypothesis-generation step, clearly separated.

## 5. Functional extension (optional, via ESEL)

If the structural cohort reaches ≥5 families, add a presentation-state arm to
the ESEL RFQ:

- monomeric peptide vs pre-aggregated/fibrillar peptide for the same epitope;
- readout: enrichment fold by state;
- claim: state-dependent enrichment, not cross-lineage convergence.

This is an experimental complement, not a substitute for independent lineages.

## 6. Deliverables

```
results/csn_expansion_v1/
├── preregistration.json          # frozen before scoring
├── peptide_inventory.json        # all peptide chains and parents
├── peptide_clusters.json         # exact/near clusters
├── lineage_calls.json            # ANARCI/IMGT clusters and independence
├── candidates.json               # eligible families and components
├── curation.csv                  # manual decisions and reasons
├── power_table.json              # families vs detectable effect
└── completion.json               # hashes and stop-rule status
```

## 7. Timeline

| Week | Work |
|---|---|
| 1 | Local SAbDab2 inventory; peptide clustering; parent mapping |
| 2 | RCSB full-PDB query; lineage clustering; first candidate list |
| 3 | Manual curation; HuggingFace/GitHub cross-check; freeze candidates |
| 4 | Power table; go/no-go; if ≥5 families, preregister the RMSD analysis |

## 8. Stop rules

- **0 eligible families** after full search: record `coordinate_starvation`
  and stop; do not relax independence to manufacture a cohort.
- **1–2 families**: complete descriptively; do not pool into a population
  claim.
- **≥5 families**: proceed to the preregistered confirmatory analysis.
- **NCS floor >0.5 Å**: downgrade the entire analysis to "measurement floor
  not validated".

## 9. Relationship to existing artifacts

- Read-only reuse: `results/csn_c0_v2/`, `results/flex_a1b_isolation_families_v2/`,
  `results/flex_a1_scaffold_audit_v1/`, `results/esel_e0b_v1/`.
- Do not modify the frozen CSN C0/C1/C2 artifacts.
- New results go to `results/csn_expansion_v1/` with write-once disclosure.
