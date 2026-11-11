# CSN Aβ N-terminal RMSD results (preregistered, 2026-11-11)

Status: **development, descriptive, not confirmatory.** The preregistration
was frozen before any RMSD value was computed
(`docs/CSN_ABETA_PREREG_V1.md`, `results/csn_abeta_v1/preregistration.json`).
The frozen CSN C0/C1/C2 results are untouched.

## 1. Cohort and measurement floor

- 14 entries / 9 independent lineages / 15 peptide chains (ANARCI/IMGT
  lineage calls in `docs/CSN_ABETA_LINEAGES_20261111.md`).
- Primary core: `AEFRH` (5 residues). Secondary core: `AEFRHD` (6 residues,
  11 entries / 8 lineages; the D1-start L6 trio is excluded).
- Same-entry NCS floor: 0.064 Å (primary, one pair: 5CSZ chains D/E) and
  0.078 Å (secondary). The floor is valid but based on a single pair.
- Within-lineage mean: 0.380 Å (primary), 0.274 Å (secondary).

## 2. Preregistered verdicts

| Analysis | Entries / lineages | Cross-lineage median (chain pairs) | Entry-pair median | Tiers same / related / different | Verdict |
|---|---:|---:|---:|---:|---|
| Primary `AEFRH` | 14 / 9 | 1.266 Å | **1.385 Å** | 35 / 6 / 57 | **`different_conformers`** |
| Secondary `AEFRHD` | 11 / 8 | 0.791 Å | **0.738 Å** | 13 / 26 / 23 | **`related_conformers`** |

Primary bootstrap (lineages, 10,000 draws, seed 20261111): median entry-pair
RMSD 95% interval **[0.364, 2.942] Å**. Secondary: **[0.499, 2.332] Å**.
The intervals are wide because the lineage-pair distribution is bimodal (next
section).

The two preregistered verdicts differ because the 5-residue core includes the
offset residue that varies between the `AEFRHDS` and `DAEFRH` forms; the
6-residue core is more homogeneous but excludes L6.

## 3. Post hoc observation: a dominant shared conformer in 6/9 lineages

The lineage-pair matrix is not unimodal. Median cross-lineage RMSDs:

| Group | Lineages | Typical RMSD |
|---|---|---|
| Tight cluster | L1, L2, L3, L4, L5, L7 | **0.13–0.66 Å** |
| Boundary | L9 | 1.11–1.30 Å vs the tight cluster |
| Distinct | L8 | 2.27–2.59 Å vs the tight cluster |
| Distinct | L6 | **2.95–3.13 Å** vs the tight cluster |

The tight cluster contains six independent lineages:
L1 (2IPU/2R0W), L2 (3BAE/3BKJ), L3 (3IFL/3IFN), L4 (3IFO), L5 (3IFP), L7
(5CSZ). Its members are mutually ≤0.66 Å, at or below the 0.5–1.0 Å
`related` tier.

L6 (4HIX/4OJF/4ONF) is the humanized/murine anti-Aβ N-terminal family; its
peptide is the 6-mer `DAEFRH` only, and it sits ~3 Å from the tight cluster.
L8 (5W3P) binds the longer `DAEFRHDSGYEVHHQK` peptide and sits ~2.3–2.6 Å
away. L9 (6CO3) binds `AEFRHD` and sits at the related/different boundary.

**This post hoc cluster is hypothesis-generating, not a preregistered
confirmatory endpoint.** It must not be reported as "the Aβ family shares one
conformer" without a new preregistration that fixes the subset definition and
the register/construct handling.

## 4. Interpretation

- The family is **heterogeneous**: a global "all nine lineages share one
  conformer" claim is not supported by the primary core.
- The data do support a narrower descriptive statement: **six independent
  lineages share a bound N-terminal conformer within ~0.7 Å**, while three
  lineages adopt distinct conformers or peptide registers.
- The L6 outlier is the most biologically interesting: the same N-terminal
  epitope can be bound in a substantially different backbone conformation by
  a different antibody family. That is the opposite of universal conformer
  selection and is worth a dedicated follow-up.
- The L7 (5CSZ) result is notable: a fibril-binding Fab sits inside the tight
  cluster, so the shared conformer is not limited to soluble peptide
  complexes.

## 5. Caveats

- The primary core is only five residues; a 5-Cα RMSD is sensitive to
  register and to the offset residue.
- The NCS floor is one pair; the measurement-floor check is weak.
- L6/L8/L9 may differ because of peptide length, register or construct, not
  because of a different selection mechanism. Register-aware alignment is the
  next analysis.
- Lineage calls are ANARCI/IMGT sequence-level; a second numbering scheme and
  manual construct curation are still required.
- No free-state disorder, affinity, specificity or experimental binding data
  enter this analysis.

## 6. What would change the claim

1. A preregistered **tight-cluster** analysis with the six lineages and a
   register-aware alignment; if it holds at ≤0.5 Å, the shared-conformer
   statement becomes confirmatory for that subset.
2. A preregistered **outlier** analysis for L6/L8/L9; if their peptides are in
   different registers, the "different conformer" reading weakens.
3. New independent lineages from a full RCSB search; the current nine are a
   local lower bound.

## 7. Data and reproduction

```text
results/csn_abeta_v1/
├── abeta_entries.csv                 # curation table
├── abeta_vh_anarci.csv_H.csv         # ANARCI VH output
├── abeta_vl_anarci.csv_KL.csv        # ANARCI VL output
├── preregistration.json              # frozen before RMSD
├── rmsd_pairs_primary.csv
├── rmsd_pairs_secondary.csv
├── summary_primary.json
├── summary_secondary.json
└── completion.json
```

Reproduce:

```bash
python scripts/build_csn_abeta_inventory_v1.py
# ANARCI is run in the WSL bioconda environment; see docs/CSN_ABETA_LINEAGES_20261111.md
python scripts/csn_abeta_rmsd_v1.py
```

The RMSD script refuses to run unless the preregistration status is
`frozen_before_rmsd`.

## 8. Claim boundary

- This is a development descriptive result, not a confirmatory claim.
- It does not modify the frozen CSN C0/C1/C2 results.
- A positive subset result would generalize the **method** to Aβ N-terminal
  recognition, not the original tau PHF6* claim.
- No binding, affinity, specificity or therapeutic claim is made.
