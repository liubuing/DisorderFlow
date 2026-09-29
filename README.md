# DisorderFlow — ECLS Structural Sequence Scoring

DisorderFlow is a research repository for antibody–peptide sequence scoring
and related BFN design experiments. **The current primary manuscript and
release are ECLS v1**, targeting a computational structural sequence-scoring
paper in Bioinformatics. The manuscript has been rewritten around native-sequence recognition versus candidate ranking. Scientific text and a current local reanalysis archive are prepared for author review; author declarations, journal submission and remote publication remain unconfirmed.

## Primary paper: ECLS

Epitope-conditioned likelihood shift compares the same H3 sequence in two
coordinate contexts:

```text
ECLS = H3 NLL(complex backbone)
     - H3 NLL(peptide-stripped, complex-derived Fab backbone)
advantage = mean ECLS(composition-matched shuffled H3) - ECLS(native H3)
```

The frozen primary claim is that deposited native H3 sequences have a more
favorable contrast than composition-matched shuffles on the temporal structure
panel. It is not a binding/affinity prediction or a general generated-candidate
reranking claim.

| Frozen temporal-final endpoint | Result |
|---|---:|
| Structures / antigen-cluster inference units | 31 / 15 |
| Mean ECLS advantage | 0.172281 |
| Cluster-bootstrap 95% CI | [0.059156, 0.291920] |
| Positive antigen clusters | 12 / 15 |

The result passed the prespecified internal gates. This does not mean journal
acceptance. The final evaluation is terminal and must not be rerun or retuned.

Current entry points:

- [Publication protocol](PUBLICATION_PROTOCOL.md): claim, exclusions and target venue.
- [Frozen ECLS scope](publication/ECLS_SCOPE_FREEZE.yml): immutable evidence contract.
- [Publication map](docs/PUBLICATION_MAP.md): primary and separate research branches.
- [ECLS reproducibility](release/ecls_v1/REPRODUCIBILITY.md): local verification and release status.
- Local manuscript: `publication/MANUSCRIPT_DRAFT.md` (unpublished; not tracked in Git).
- Local upload preparation: `release/zenodo_v1/DEPOSIT_INSTRUCTIONS.md`.

## Other research branches

| Branch | Status | Relationship to ECLS v1 |
|---|---|---|
| PAE surrogate | Separate revision; input-provenance and baseline issues audited; complex-model advantage not established | Not the ECLS manuscript or its primary evidence |
| Successor/contact-v2 | Frozen development candidate awaiting at least 12 independent components | Supplementary/development provenance only |
| Disorder-aware BFN design | Generation and scoring infrastructure; no experimentally validated binder | Research platform, not evidence of successful antibody design |
| T1/T2 and multiscaffold design | Bounded exploratory or negative computational findings | Limitations/supporting history only |

Current PAE entry points are the [v20 audit](docs/PAE_EVIDENCE_AUDIT_V20.md),
[v21 data audit](docs/PAE_NEW_FAMILY_FEASIBILITY_V21.md), and local
`publication/pae_screening_evidence_v20_draft.md`. The
[v4 corrections](docs/PAE_SURROGATE_REVISION_V4.md) remain mandatory context;
the v3 manuscript/PDF are superseded for submission purposes. The old evaluation consumed AF2 output coordinates; its results
do not establish pre-AF2 screening. That branch's limitations do not invalidate
the separate frozen ECLS result.

## Software and verification

`disorderflow/` holds the core models and datasets, `modules/` the runtime
interfaces, `scripts/` the research workflows, and `tests/` the scientific
contracts. Related fixed-backbone BFN sequence generation remains available via
`modules.bfn_loader.run_bfn_design`; feature availability does not establish
binding, specificity or efficacy.

```bash
python scripts/validate_release_lineage.py --source-only
python scripts/validate_publication_alignment.py
```

The local publication bundle has not been verified as remotely published. No
wet-lab binding measurements are claimed. License: [MIT](LICENSE.md).

## Repository layout & maintenance (2026-09-27 engineering-debt pass)

Entry points and post-split structure — full details in the Chinese guide
below; the publication claims above are unaffected.

- `app.py` is now a 65-line launcher; all web implementation lives in the
  `webapp/` package (`state/sequtils/design/baselines/af2/pipelines/ui/facade`).
  The venv is installed editable (`pip install -e . --no-deps`), so
  `modules/` scripts no longer need `PYTHONPATH=.`. Note: `C:\biological` is
  a directory **junction** to `D:\biological` (same files, same inode — not a
  copy); the venv was created through it, and on 2026-09-29 the activate
  scripts were repointed to the `D:` spelling and editable install redone.
  Prefer `D:` paths in new references.
- `manage.py` retains `start|stop|restart|status|batch|config|test`.
- One-off v5 build-campaign scripts moved to `scripts/oneoff/v5_build_campaign/`
  with their inputs in `inputs/oneoff_v5/` and logs in `_archive_local/`.
- Checkpoint truth table: [configs/CHECKPOINT_TRUTH.md](configs/CHECKPOINT_TRUTH.md)
  (current main = `bfn_disorder_v4_afdb_supervised_2026_08_05_..._s2032`;
  `configs/demo_design.yml` was re-pointed there from the 2026-05-18 v6_phase2).

仓库导览（中文）：

| 入口 | 说明 |
|---|---|
| `app.py` | Web 入口薄壳（65 行），实现全在 `webapp/` 包；**新功能不再写进 app.py** |
| `manage.py` | 服务管理 start/stop/restart/status/batch/config/test |
| `train*.py` | BFN 主训练与无序头训练谱系（`train.py` 为主入口） |
| `build_design_variant_dataset.py` | 置信度真标签数据集构建（"假标签"根因修复后的版本） |
| `webapp/state.py` | 配置加载、checkpoint 解析、模型单例（含 `_app_config` 等全局） |
| `webapp/ui.py` | `create_ui()`：全部 Gradio 组件与事件绑定 |
| `webapp/facade.py` | 旧式 `from app import X` 的兼容再导出层 |
| `configs/CHECKPOINT_TRUTH.md` | checkpoint 真相表（唯一权威索引） |
| `scripts/oneoff/` | 一次性战役档案（v5 建库脚本、本次拆分器），勿重跑 |

环境三层：Windows venv（日常）／WSL `Ubuntu-24.04-D`（AF2、PAE 推理）／
`D:\DisorderFlowRuntime\`（ColabFold 缓存）。冻结资产红线不变：
`publication/ECLS_SCOPE_FREEZE.yml`、一次性 holdout 评估脚本、
`publication/candidate_interface_pae_deployment_v1.json` 及配套 checkpoint。
