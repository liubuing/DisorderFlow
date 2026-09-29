# Analogue-Scope Feasibility Harvest v22

Date: 2026-09-28. This is a dry data-eligibility harvest, not a new validation
result. No ANARCII numbering, no training-overlap review, no model or AF2
scoring has been run, and no frozen decision is modified.

## Why this exists

`PAE_NEW_FAMILY_FEASIBILITY_V21.md` closed the paired-H/L antibody short-peptide
scope at zero new qualified families (SAbDab 22,308 rows; RCSB direct search;
4,005 sequences numbered; 32 identity hits all TCR/MHC/photosystem). The same
audit flagged, without quantifying, that TCR and nanobody systems "cannot
directly substitute for the paired-H/L task but may constitute separate
schemes". This harvest quantifies those two separate schemes with the same
gate style so the option is on the table with real numbers.

## Fresh global re-verification (2026-09-28)

- SAbDab all-summary re-downloaded: byte-identical to the v21 snapshot
  (22,308 rows). Zero new antibody entries since 2026-09-25.
- RCSB short-chain space (5-50 aa polymer entity, experimental): 41,071
  entries, unchanged since the v21 audit.
- RCSB index horizon: newest publicly released entry was deposited 2026-09-10.
  Nothing released after 2026-09-18 exists yet.
- Antibody-keyword + short-chain depositions since 2026-06-01: exactly three
  (33CG, 33CH, 33CI) - all MAGE-A4/HLA-A*02:01 pMHC complexes, all already in
  the project's eligible pool and already scored in prior AF2 runs.
- Historical deposition rate of antibody-keyword + short-chain entries:
  43-77 per year (2018-2025). Expected new *qualified independent antibody
  families* per year after all gates: ~0-1. Waiting is not a supply strategy.

## TCR-pMHC cohort

Query: title contains any(T-cell receptor | T cell receptor | TCR) AND a
5-50 aa polymer entity, experimental structures. 340 entries fetched; 272
pass the harvest gates (>= 1 TCR chain, >= 1 candidate peptide chain 5-50 aa
excluding Ig/MHC/TCR by description keywords, resolution <= 4.0 A).

- **272 candidate entries; 175 unique peptide sequences.**
  For scale: the frozen ECLS final used 31 structures / 15 antigen clusters.
- Methods: 265 X-ray, 7 EM. Resolution range from 1.4 A.
- Sanity check of the classifier: the most repeated identified peptides are
  canonical immunology peptides (influenza NP ASNENMETM, influenza M1
  GILGFVFTL, Melan-A ELAGIGILTV, NY-ESO-1 SLLMWITQV, HTLV-1 Tax LLFGYPVYV),
  so chain classification is behaving.
- Files: `data/pae_analog_scope_v22/tcr_candidates.json` (per-entry: method,
  resolution, dates, identified TCR and peptide chains), plus raw
  `tcr_ids/tcr_meta/tcr_rows.json` and a checksummed
  `HARVEST_RECEIPT.json`.

## Nanobody/VHH cohort

Query: title contains any(nanobody | VHH | single-domain) AND a 5-50 aa
polymer entity. 54 entries; **52 pass the gates; 41 unique peptide
sequences.** Notably the cohort contains multiple Tau peptide structures
(PHF6 PGGGSVQIVYKPKK, Tau 301-312, Tau-F isoform peptides) - i.e. structures
directly relevant to the project's second disease target - plus gp41 and
microcystin-LR complexes. Files: `data/pae_analog_scope_v22/nano_passed.json`.

## What these cohorts would enable, and what they would not claim

Enabled (all still to be gated and preregistered before any run):

1. Real bound geometries and AF2-PAE teacher labels at cohort scale
   (272 + 52 structures) for an analogue-transfer validation of the
   composition-driven prescreening recipe.
2. A same-recipe retrain with mapped features: CDR3-beta composition for TCR,
   CDR3 composition for VHH, in place of H3 composition.

Not claimable:

- This is **analogue transfer, not antibody validation**. MHC-groove-anchored
  peptides are conformationally constrained at both ends, unlike free
  disordered epitopes; the interface geometry differs. Any writeup must state
  the transfer explicitly.
- The frozen antibody-trained models are not reused for scoring; the recipe
  is retrained on mapped features. Model-reuse claims are out of scope.
- Per the standing protocol, before any generation or AF2 run: coordinates
  and chemistry checks, actual training-overlap review of the frozen models'
  sources against these PDB entries, and upstream provenance review. Metadata
  classification here is title/description level; chain-level identity
  verification (numbering) is still pending.

## Immediate data verdict

Within the paired-H/L antibody short-peptide scope, global public structure
data remains exhausted (three independent audits + this fresh re-check). The
fillable path from public data is the analogue scope, and it now has concrete
numbers: 272 TCR entries / 175 unique peptides and 52 VHH entries / 41 unique
peptides, checksummed and stored locally under `data/pae_analog_scope_v22/`.
