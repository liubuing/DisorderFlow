"""Train and evaluate a small pairwise ranking head on frozen features (v1).

Supervision: within-complex pairs (i, j): label = 1 if variant i binds better
than j.  Features are frozen (no model training): 20-dim composition,
the corrected-role BFN scores (ECLS / complex NLL / stripped NLL), and simple
biophysical descriptors.

Evaluation: leave-one-complex-out (LOCO). For each held-out complex, train on
the other four and report within-group pairwise accuracy and Spearman between
the predicted score and the measured label.

The decisive comparison is BFN+composition vs composition alone: if the
corrected judge carries binding information beyond composition, the held-out
Spearman must improve by a pre-declared margin (>= 0.05).

Output: results/pairwise_v1/head_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FEATURES = ROOT / 'results/pairwise_v1/bfn_features.json'
OUT = ROOT / 'results/pairwise_v1'
AA = 'ACDEFGHIKLMNPQRSTVWY'
MAX_PAIRS_PER_GROUP = 20_000
SEED = 20261101


def biophysical(h3):
    n = len(h3)
    return [
        sum(h3.count(a) for a in 'KR') - sum(h3.count(a) for a in 'DE'),
        sum(h3.count(a) for a in 'AVILMFWC') / n,
        sum(h3.count(a) for a in 'FWY') / n,
        h3.count('P') / n,
        h3.count('G') / n,
    ]


def composition(h3):
    n = len(h3)
    return [h3.count(a) / n for a in AA]


def build_matrix(records):
    import numpy as np
    rows = []
    for record in records:
        h3 = record['h3']
        rows.append({
            'group': record['group'],
            'h3': h3,
            'label': record['label'],
            'composition': composition(h3),
            'bfn': [record['ecls'], record['complex_nll'], record['stripped_nll']],
            'biophys': biophysical(h3),
        })
    return rows


def feature_vector(row, kind):
    if kind == 'composition':
        return row['composition']
    if kind == 'bfn':
        return row['bfn']
    if kind == 'biophys':
        return row['biophys']
    if kind == 'bfn+composition':
        return row['bfn'] + row['composition']
    if kind == 'bfn+composition+biophys':
        return row['bfn'] + row['composition'] + row['biophys']
    raise ValueError(kind)


def pairs_for_group(rows, rng):
    n = len(rows)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if rows[i]['label'] == rows[j]['label']:
                continue
            pairs.append((i, j, 1.0 if rows[i]['label'] > rows[j]['label'] else 0.0))
    if len(pairs) > MAX_PAIRS_PER_GROUP:
        idx = rng.choice(len(pairs), MAX_PAIRS_PER_GROUP, replace=False)
        pairs = [pairs[k] for k in idx]
    return pairs


def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3:
        return float('nan')

    def rank(v):
        order = np.argsort(v, kind='mergesort')
        ranks = np.empty(len(v), float)
        ranks[order] = np.arange(len(v), dtype=float)
        return ranks
    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom else float('nan')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', default=str(FEATURES))
    parser.add_argument('--out', default=str(OUT / 'head_report.json'))
    parser.add_argument('--margin', type=float, default=0.05)
    args = parser.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    data = json.loads(Path(args.features).read_text(encoding='utf-8'))
    rows = build_matrix(data['records'])
    groups = sorted({r['group'] for r in rows})
    rng = np.random.default_rng(SEED)

    kinds = ['composition', 'bfn', 'biophys', 'bfn+composition', 'bfn+composition+biophys']
    results = {k: {'per_group': {}, 'mean_spearman': None, 'mean_accuracy': None}
               for k in kinds}

    for held_out in groups:
        train_rows = [r for r in rows if r['group'] != held_out]
        test_rows = [r for r in rows if r['group'] == held_out]
        train_pairs = []
        for group in sorted({r['group'] for r in train_rows}):
            group_rows = [r for r in train_rows if r['group'] == group]
            local = pairs_for_group(group_rows, rng)
            train_pairs.extend((group_rows[i], group_rows[j], label)
                               for i, j, label in local)
        x_train = {k: [] for k in kinds}
        y_train = []
        for a, b, label in train_pairs:
            for k in kinds:
                fa = np.asarray(feature_vector(a, k), float)
                fb = np.asarray(feature_vector(b, k), float)
                x_train[k].append(fa - fb)
            y_train.append(label)
        y_train = np.asarray(y_train, float)
        test_pairs = pairs_for_group(test_rows, rng)

        for kind in kinds:
            x = np.asarray(x_train[kind], float)
            y = y_train
            if len(np.unique(y)) < 2:
                continue
            scaler = StandardScaler().fit(x)
            model = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
            model.fit(scaler.transform(x), y)

            # per-variant score = w . f  (differences cancel the intercept)
            def score(row):
                f = np.asarray(feature_vector(row, kind), float)
                return float(model.coef_[0] @ (f / scaler.scale_))

            scores = [score(r) for r in test_rows]
            truth = [r['label'] for r in test_rows]
            acc = float(np.mean([
                1.0 if (score(test_rows[i]) - score(test_rows[j])) *
                (1.0 if test_rows[i]['label'] > test_rows[j]['label'] else -1.0) > 0
                else 0.0
                for i, j, _ in test_pairs
            ])) if test_pairs else float('nan')
            rho = spearman(scores, truth)
            results[kind]['per_group'][held_out] = {
                'n_test': len(test_rows),
                'n_pairs': len(test_pairs),
                'pairwise_accuracy': acc,
                'spearman': rho,
            }
        print('held out', held_out, 'done', flush=True)

    for kind in kinds:
        rhos = [v['spearman'] for v in results[kind]['per_group'].values()
                if v['spearman'] is not None and not np.isnan(v['spearman'])]
        accs = [v['pairwise_accuracy'] for v in results[kind]['per_group'].values()
                if v['pairwise_accuracy'] is not None and not np.isnan(v['pairwise_accuracy'])]
        results[kind]['mean_spearman'] = float(np.mean(rhos)) if rhos else None
        results[kind]['mean_accuracy'] = float(np.mean(accs)) if accs else None

    comp = results['composition']['mean_spearman']
    both = results['bfn+composition']['mean_spearman']
    delta = None if comp is None or both is None else both - comp
    verdict = {
        'margin_required': args.margin,
        'delta_spearman_bfn_plus_composition_minus_composition': delta,
        'passes': bool(delta is not None and delta >= args.margin),
        'interpretation': (
            'BFN adds binding information beyond composition' if delta is not None and delta >= args.margin
            else 'no evidence that the BFN judge adds binding information beyond composition'),
    }
    report = {
        'schema_version': 'pairwise_head_v1',
        'n_groups': len(groups),
        'groups': groups,
        'records': len(rows),
        'kinds': kinds,
        'results': results,
        'verdict': verdict,
        'classification': 'development_test_not_confirmatory',
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'results': {k: {'mean_spearman': results[k]['mean_spearman'],
                                      'mean_accuracy': results[k]['mean_accuracy']}
                                  for k in kinds},
                      'verdict': verdict}, indent=1))


if __name__ == '__main__':
    main()
