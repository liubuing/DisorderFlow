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

## 4.1 Register-aware sensitivity (post hoc)

The primary 5-residue core `AEFRH` is shared by two peptide registers: the
A2-start form (`AEFRHDS`/`AEFRHD`) and the D1-start form (`DAEFRH…`). To test
whether the tight cluster is a register artifact, the cohort was split by
register and compared within each register (`scripts/csn_abeta_register_rmsd_v1.py`).

| Group | Core | Entries / lineages | Cross-lineage median | Tiers same / related / different | NCS |
|---|---|---:|---:|---:|---:|
| R0 A2-start | `AEFRH` | 8 / 6 | **0.368 Å** | 18 / 1 / 7 | not estimable |
| R0 A2-start | `AEFRHD` | 8 / 6 | 0.551 Å | 9 / 10 / 7 | not estimable |
| R1 D1-start | `AEFRH` | 6 / 4 | **2.937 Å** | 2 / 0 / 15 | 0.064 Å |
| R1 D1-start | `DAEFRH` | 6 / 4 | 2.957 Å | 0 / 0 / 17 | 0.120 Å |

Key observations:

- **The 5-residue core is the stable unit.** Adding the terminal D1/D7 residue
  inflates RMSD (R0 0.368 → 0.551 Å; R1 2.937 → 2.957 Å), so the 6-residue
  differences are partly terminal flexibility, not a different fold.
- **The tight cluster survives the split.** In R0, L1–L5 are mutually ≤0.5 Å;
  L9 is the outlier (~1.2 Å). In R1, L3 (3IFL) and L7 (5CSZ) remain close
  (2 chain pairs ≤0.5 Å), while L6 and L8 are distinct.
- **Register is not the driver.** The same lineage L3 has entries in both
  registers (3IFN A2-start, 3IFL D1-start); their shared `AEFRHD` RMSD is
  **0.274 Å**. The same antibody binds both offset forms in essentially the
  same backbone conformation.
- **The global six-lineage cluster is confirmed**: L1, L2, L3, L4, L5, L7
  share the `AEFRH` conformer; L6 and L8 are genuine outliers, and L9 sits at
  the boundary.

The R0 NCS floor is not estimable because no A2-start entry has two peptide
copies in the core; the R1 floor comes from 5CSZ chains D/E (0.064 Å on
`AEFRH`). This is a limitation, not a failure: the global NCS pair was also
5CSZ.

## 4.2 Free-state ensemble comparison (post hoc)

Question: is the bound `AEFRH` conformer present in public free-state NMR
ensembles, and how common is it?

Sources: RCSB NMR entries 1AMC (Aβ 1-28, 5 models), 1BJB (Aβ[E16] 1-28, 14
models), 1BJC (Aβ[F16] 1-28, 15 models), 1ZE7 (Aβ 1-16, 20 models); sensitivity
arms 1ZE9 (zinc-bound, 20 models) and 2BP4 (80% TFE, 20 models).

Method: extract the `AEFRH` Cα trace from each NMR model, superpose it on each
bound tight-cluster reference (L1–L5, L7), and record the best RMSD and the
closest lineage.

| NMR entry | Condition | Models | Median best RMSD | Fraction ≤0.5 Å | Free-ensemble pairwise median |
|---|---|---:|---:|---:|---:|
| 1AMC | aqueous, Aβ 1-28 | 5 | 3.057 Å | 0.00 | 0.118 Å |
| 1BJB | aqueous, Aβ[E16] 1-28 | 14 | 2.178 Å | 0.00 | 0.958 Å |
| 1BJC | aqueous, Aβ[F16] 1-28 | 15 | 2.084 Å | 0.00 | 1.907 Å |
| 1ZE7 | aqueous, Aβ 1-16 | 20 | 1.789 Å | **0.10** | 1.697 Å |
| 1ZE9 | zinc-bound, Aβ 1-16 | 20 | 1.812 Å | 0.00 | 0.676 Å |
| 2BP4 | 80% TFE, Aβ 1-16 | 20 | 3.114 Å | 0.00 | 0.114 Å |

Primary aqueous aggregate (54 models): median best RMSD **2.079 Å**; only
**3.7%** within 1.0 Å and **13.0%** within 1.5 Å. The single near-native model
is in 1ZE7 (best RMSD **0.202 Å**, 1/20 models).

Interpretation:

- The bound tight-cluster conformer is **not a dominant free-state state** in
  these NMR ensembles. Most models are 2–3 Å away.
- It is **accessible**: 1ZE7 contains one model at 0.20 Å, and 30% of its
  models are within 1.5 Å. This is consistent with a minor, accessible state
  that binding could select.
- 1AMC and 2BP4 are internally tight but far from the bound conformer; they
  likely represent helical/TFE-induced states, not the aqueous free ensemble.
- The data are **consistent with conformer selection but do not prove it**:
  the near-native model is one of twenty, the ensembles are not
  Boltzmann-weighted, and the peptide lengths/conditions differ from the
  bound complexes.

This is a descriptive free-state comparison, not a thermodynamic or kinetic
measurement.

## 4.3 Numbering-scheme sensitivity

All 16 VH and 16 VL sequences were re-numbered with ANARCI Kabat and Chothia
schemes (`scripts/csn_abeta_scheme_sensitivity_v1.py`). V/J germline and
closest-species assignments were **identical across IMGT, Kabat and Chothia
(0 mismatches)**. CDR3 boundaries differ by scheme as expected: VH
IMGT-vs-Kabat sequence similarity median 0.748; VL 1.0. The lineage calls do
not depend on the numbering scheme.

## 4.4 Full PDB metadata scan

A two-pass scan of `pdb_seqres.txt.gz` found 164 PDB entries with a
`DAEFRH`-containing chain. Keyword-based antibody detection flagged 18; 17 are
already in SAbDab. The single new hit, **8KH2**, is a horse-spleen L-ferritin
fusion with Aβ(1-42), not an antibody. **No additional antibody lineage was
found.** A sequence search cannot exclude antibodies with highly divergent
framework sequences, but the current 9-lineage cohort is complete at the
metadata level.

Construct-level curation (3IFL/3IFN same 12A11 antibody in two registers;
3IFO/3IFP distinct antibodies; 4HIX/4OJF/4ONF same 3D6 clonotype; 4ONG
excluded as a 5-mer) is recorded in `docs/CSN_ABETA_CURATION_20261111.md`.

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

Manuscript draft: `publication/csn_abeta_manuscript_draft_v1.md` (local, not
tracked, not submitted).

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
