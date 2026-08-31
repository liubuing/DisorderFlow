# DisorderFlow iBEC Evidence Verification

## Package Scope

This lightweight ZIP contains reviewer-facing reports, compact frozen result
artifacts, and SHA256 manifests. It does not contain the full source repository,
model weights, licensed datasets, AF2 parameters, or work directories. It can
verify the integrity and provenance fields of included evidence; it cannot
independently rerun model inference or recompute every analysis.

## Integrity Check

From the extracted `AnalyzingDisorder` directory, run:

```powershell
python VERIFY_PACKAGE.py
```

The command verifies every non-self entry in `PACKAGE_MANIFEST.json` and every
non-self entry in `Supplementary_Materials/MANIFEST.json`. Manifest files omit
their own hashes because a file cannot contain a stable hash of itself.

## Evidence Map

- `ecls_final_decision.json` contains the immutable temporal-final decision.
- `ecls_evidence_manifest.json` inventories 68 checksummed files in the full
  ECLS reviewer release and records excluded large asset classes.
- `abeta_candidate_summary.json` and `abeta_final_shortlist.csv` contain the
  Aβ computational funnel and final hypotheses.
- `caid3_external_analysis.json` contains the frozen external disorder-head
  discrimination and calibration analysis.
- `e1_generator_behavior_summary.json` and `e2_idp_evidence_summary.json`
  contain the iBEC-specific generator and evidence audits.
- `tau_sanity_status.json` and `tau_candidate_results.json` contain the
  correctly masked Tau second-target evidence.
- `binder_confidence_status.json` records abstention with no experimental
  labels and a null probability output.

## Full Reproduction Boundary

The complete repository is required for tests and saved-result recomputation.
Full model reproduction additionally requires separately distributed datasets,
checkpoints, ProteinMPNN/ESM-IF weights, and AF2 parameters matching the hashes
recorded by each frozen contract. The temporal ECLS final is terminal and must
not be rerun or replaced by the authors.
