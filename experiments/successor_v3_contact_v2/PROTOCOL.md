# Successor-v3 Contact-v2 SPR/BLI Protocol

Status: planned, not executed. This document contains no experimental results.

## Objective

Test whether alanine substitutions at model-predicted H3 contact residues weaken
binding more than substitutions at low-contact H3 residues. The primary endpoint
is the target-level difference between the mean effect of the two high-contact
mutations and the low-contact mutation control.

## Panel And Blinding

- Use the 64 constructs in `constructs.csv`: 16 independent target components,
  one WT, two predicted-contact disruptions, and one low-contact control each.
- A scientist not running the assay assigns opaque sample IDs and keeps the key
  inaccessible until all QC decisions and kinetic fits are frozen.
- Express and purify all four constructs for a target in the same production batch.
- Confirm identity by intact mass or peptide mapping and report purity by SEC-HPLC.
- Exclude only by the predeclared QC criteria below; retain every exclusion in
  `measurements_template.csv` with its reason.

## Assay Setup

- Preferred platform: SPR with monomeric antigen immobilized at low density.
  BLI is acceptable when the same orientation and analysis rules are used for all
  constructs of a target.
- Run at 25 C in one validated buffer formulation. Record composition, pH,
  detergent, DMSO, chip or sensor lot, and antigen immobilization level.
- Include a reference surface or reference sensor and buffer blanks.
- Use a preliminary WT range-finding run only to set a target-specific dilution
  series. Do not inspect mutant effects during range finding.
- Use at least eight analyte concentrations spanning approximately `0.1x` to
  `10x` the WT KD, plus zero concentration. Use a two-fold or three-fold series.
- Randomize constructs within each target and randomize concentration order when
  instrument carryover permits. Place one repeated midpoint concentration at the
  end of each cycle to monitor drift.
- Collect three independent technical runs on separately prepared analyte
  dilutions. Technical runs do not count as additional biological mutations.

## Kinetic Analysis

- Apply reference and blank subtraction before fitting.
- Fit the simplest justified model. Use 1:1 Langmuir globally across
  concentrations unless residual structure, mass transport, or avidity invalidates
  it. Any alternative model must be selected while blinded and recorded.
- Freeze inclusion, model choice, and fitted values before unblinding.
- Primary value: median valid-run `log10(KD)` per construct.
- For each target, calculate both high-contact mutant effects as mutant minus WT
  `log10(KD)`, average those two effects, and subtract the low-contact mutant
  minus WT effect. This yields exactly one independent contrast per target.
- Primary test: two-sided one-sample test of the 16 target-level contrasts
  against zero. Report a target bootstrap 95% CI, an exact sign-flip test, and
  all individual construct effects. Do not analyze 32 high-contact mutations as
  32 independent observations.
- Censored KD values use the preregistered interval-censored sensitivity analysis;
  they must not be replaced by assay limits as if exactly measured.
- Report effect sizes and confidence intervals regardless of significance.

## QC And Exclusions

A run is valid only when all applicable criteria pass:

- Construct identity confirmed and SEC-HPLC monomer fraction at least 90%.
- Reference-subtracted response is concentration ordered over at least five
  non-zero concentrations.
- Replicate midpoint response differs by no more than 20% from its first value.
- Fitted KD is within the tested dynamic range; otherwise report a bound such as
  `KD > upper_limit` and do not substitute a numeric value.
- Fitted `Rmax` is physically plausible and residuals show no systematic phase
  pattern. Record numeric fit diagnostics supplied by the instrument software.
- No visible carryover after the declared regeneration procedure.
- At least two of three technical runs pass. Report all failed runs and reasons.

Do not exclude a construct because its direction disagrees with the model. Do not
replace a failed target after unblinding.

## Power And Interpretation

The frozen panel has 16 independent target contrasts. With a mean high-minus-low
effect of `0.5 log10(KD)`, target-level SD `0.5 log10(KD)`, and two-sided alpha
`0.05`, planned target-level t-test power is recorded in `panel.json`. The SD and
effect are planning assumptions, not observations. Technical replicates do not
increase the independent sample size.

SPR/BLI can test binding and kinetic effects for these constructs. It does not by
itself establish therapeutic specificity, developability, efficacy, or safety.
