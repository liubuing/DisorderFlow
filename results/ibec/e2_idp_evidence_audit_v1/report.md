# iBEC E2 IDP Evidence Audit

## Dataset Identity

- Source LMDB: 1301 records.
- Frozen conformation LMDB: 1289 records.
- Every retained record contains exactly five conformations.
- Twelve final source records (length 495-500) were mechanically omitted because the build stopped at 1,289; this was not biological filtering.

## Evidence Tiers

| Tier | Count |
|---|---:|
| experimentally_curated_fully_disordered | 23 |
| experimentally_curated_partially_disordered | 36 |
| experimentally_curated_contains_idr | 99 |
| experimentally_curated_short_disorder | 61 |
| disprot_exact_without_disorder_region | 1 |
| prediction_only_disorder | 9 |
| conformational_proxy_only | 5 |
| no_disorder_evidence | 1055 |

## Corrected Competition Statement

> We built a 1,289-record, five-conformation protein dataset. 219 records have exact-sequence DisProt disorder-region evidence; the remaining records are separated into prediction-only, conformational-proxy, and no-evidence tiers.

The previous phrase '800+ natural IDPs' is not supported and must not be used.

## Claim Boundary

MobiDB-lite predictions and AF2 pLDDT fallbacks are not experimental disorder annotations. Exact DisProt membership without a disorder region is also not counted as experimental disorder evidence.
