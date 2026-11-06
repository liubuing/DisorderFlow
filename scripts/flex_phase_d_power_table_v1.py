"""FLEX-BINDER Phase D power table (frozen; data-independent mathematics).

Phase D redefines the task as binder-vs-nonbinder discrimination. The
family-level consistency gate follows the campaign's sign-test convention:
count antigen families whose discrimination AUC exceeds 0.5. This script
freezes, before any Phase D data exists, the critical counts and power for
planned family counts - so the feasibility gate can state honestly how many
families the expansion must deliver.

Pure binomial arithmetic; no data, no network, no model.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import freeze, digest

OUT = ROOT / 'results/flex_phase_d_v1'
ALPHA = 0.05
P_ALTS = (0.65, 0.70, 0.75, 0.80, 0.90)


def tail(n, k, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = {}
    for families in (12, 16, 20, 24, 30, 40):
        critical = None
        for k in range(families + 1):
            if tail(families, k, 0.5) <= ALPHA:
                critical = k
                break
        row = {
            'critical_count': critical,
            'realized_alpha': tail(families, critical, 0.5),
            'family_positive_fraction_required': critical / families,
            'power': {f'p_alt_{p}': tail(families, critical, p) for p in P_ALTS},
        }
        table[f'families_{families}'] = row
    result = {
        'classification': 'frozen_power_mathematics_no_data_claims',
        'endpoint_convention': 'family-level sign test: family counts as positive '
                               'when its binder-vs-nonbinder AUC > 0.5; one-sided '
                               'alpha = 0.05',
        'feasibility_reading': 'the Phase D data-feasibility gate must deliver enough '
                               'families that the detectable family-positive '
                               'fraction is scientifically meaningful; compare with '
                               'the J0/C1 experience (8-9 families -> detectable '
                               'only at >= 8/9 positive)',
        'table': table,
        'script_sha256': digest(Path(__file__)),
    }
    freeze(OUT / 'power_table.json', result)
    compact = {k: {'critical': v['critical_count'],
                   'min_fraction': round(v['family_positive_fraction_required'], 3),
                   'power@0.75': round(v['power']['p_alt_0.75'], 3)}
               for k, v in table.items()}
    print(json.dumps(compact, indent=1))


if __name__ == '__main__':
    main()
