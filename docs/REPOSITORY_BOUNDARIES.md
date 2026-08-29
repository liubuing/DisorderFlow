# Repository Boundaries

DisorderFlow contains production-like runtime code, active research workflows,
frozen publication evidence, generated outputs, and third-party source. They
must not be treated as one undifferentiated Python package.

| Zone | Location | Rule |
|---|---|---|
| Core package | `disorderflow/` | Reusable model, dataset, loss, and utility code |
| Stable runtime | `app.py`, `manage.py`, `run_disorderflow.ps1` | User-facing local service |
| Active workflows | `scripts/`, `modules/` | Versioned research orchestration |
| Tests | `tests/` | Behavioral and scientific contracts |
| Frozen evidence | `publication/`, `release/`, publication results | Write-once; preserve hashes |
| Generated outputs | `reviewer_outputs/`, `outputs/`, `logs/` | Data, never imported as source |
| Third party | `ProteinMPNN/`, `diffab/` | Keep upstream provenance and focused tests |

New research entrypoints belong under `scripts/`. Existing root scripts are
retained only because moving them would break recorded commands and historical
provenance.

The active StateContrast-v2 command surface is:

```powershell
D:\DisorderFlowRuntime\Python314\python.exe scripts\statecontrast_v2_cli.py --help
```

Repository audit:

```powershell
D:\DisorderFlowRuntime\Python314\python.exe scripts\audit_repository_boundaries.py
```
