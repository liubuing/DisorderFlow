# Candidate-Interface Confidence V1 Development Checkpoint

## Classification

Development-only. `deployment_gate_passed` is false. This checkpoint must not
replace the default DisorderFlow checkpoint or be used as evidence of calibrated
binder probability, affinity, specificity, or efficacy.

## Selected Artifact

- Checkpoint: `logs/bfn_candidate_interface_confidence_v1_2026_08_30__13_27_28_candidate_interface_v1b_ranked_s2041/checkpoints/200.pt`
- SHA-256: `94817ed8b8d06d8ec7a5708416df8e1930f3896d0001cc833396bdb413e935c3`
- Size: 45,982,657 bytes
- Iteration: 200, raw weights
- Head: `candidate_interface_v1`, 24/24 successor tensors present
- Frozen initializer SHA-256: `9bc6d00a80ca07faaba2176a6b7cbdd48b24072b9640aab763c4ffe36867f2e7`
- Dataset manifest SHA-256: `b1a000db65eef5e004d2e3f43e643975bd66cc2cdee8c289712d32c286a227bb`

## Data Contract

The checkpoint was trained on 345 frozen complete-complex ColabFold bundles:
315 from 8B9V/alpha-synuclein and 30 from 3STB/A beta 42. Validation used 52
5MP3/Tau bundles in 23 complete protocol/pose/seed groups. No 5MP3 record was
used for optimization.

Only the 25,155 `candidate_interface_*` parameters were trainable. The legacy
BFN backbone remained frozen.

## Held-Out Fixed-Score Results

| Output | MAE | Mean group Spearman | Pair accuracy |
|---|---:|---:|---:|
| Candidate pLDDT | 0.1036 | 0.3913 | 0.5714 |
| Candidate ipTM | 0.1153 | 0.8849 | 0.9333 |
| Candidate-to-antigen PAE | 0.2810 | -0.0217 | 0.5429 |

PAE values are normalized by 31 for these metrics. The checkpoint resolves the
architectural sequence-invariance problem and learns useful held-out ipTM
ordering, but pLDDT and PAE ranking do not pass a deployment standard. Absolute
calibration also requires another independent scaffold.

Full per-checkpoint and per-record results are in
`evaluation_v1b_ranked.json`. The earlier objective-wiring baseline is retained
in `evaluation.json`.
