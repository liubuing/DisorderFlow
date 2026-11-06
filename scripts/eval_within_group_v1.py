"""Within-complex cross-validated evaluation of ranking signals (v1).

The deployment question is: given one antibody-antigen complex and a set of
H3 variants, which ranking signal orders them best?  Cross-group transfer is a
different (and much harder) question and is reported separately in
head_report.json.

Here, for every complex independently, run 5-fold cross-validation over pairs:
train the pairwise head on 4 folds, score the held-out fold, and report

  - pairwise accuracy on held-out pairs
  - Spearman between the predicted per-variant score and the measured label

for: composition (20-dim), BFN scores (ECLS / complex NLL / stripped NLL),
both, and biophysical descriptors.

Output: results/pairwise_v1/within_group_report.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / 'results/pairwise_v1/bfn_features.json'
OUT = ROOT / 'results/pairwise_v1'
AA = 'ACDEFGHIKLMNPQRSTVWY'
SEED = 20261101
N_FOLDS = 5


def composition(h3):
    n = len(h3)
    return [h3.count(a) / n for a in AA]


def biophys(h3):
    n = len(h3)
    return [sum(h3.count(a) for a in 'KR') - sum(h3.count(a) for a in 'DE'),
            sum(h3.count(a) for a in 'AVILMFWC') / n,
            sum(h3.count(a) for a in 'FWY') / n,
            h3.count('P') / n, h3.count('G') / n]


KINDS = {
    'composition': lambda r: composition(r['h3']),
    'bfn': lambda r: [r['ecls'], r['complex_nll'], r['stripped_nll']],
    'bfn+composition': lambda r: [r['ecls'], r['complex_nll'], r['stripped_nll']] + composition(r['h3']),
    'biophys': lambda r: biophys(r['h3']),
}


def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float('nan')

    def rank(v):
        order = np.argsort(v, kind='mergesort')
        ranks = np.empty(len(v), float)
        ranks[order] = np.arange(len(v), dtype=float)
        return ranks
    rx, ry = rank(x), rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    data = json.loads(FEATURES.read_text(encoding='utf-8'))['records']
    report = {'schema_version': 'within_group_v1', 'groups': {}}
    for group in sorted({r['group'] for r in data}):
        rows = [r for r in data if r['group'] == group]
        n = len(rows)
        labels = np.asarray([r['label'] for r in rows])
        fold_results = {k: {'accuracy': [], 'spearman': []} for k in KINDS}
        kf = KFold(n_splits=min(N_FOLDS, max(2, n // 10)), shuffle=True,
                   random_state=SEED)
        rng = np.random.default_rng(SEED)
        for train_idx, test_idx in kf.split(np.arange(n)):
            train = [rows[i] for i in train_idx]
            test = [rows[i] for i in test_idx]
            # training pairs from the train fold
            pairs = []
            for i in range(len(train)):
                for j in range(i + 1, len(train)):
                    if train[i]['label'] == train[j]['label']:
                        continue
                    pairs.append((i, j, 1.0 if train[i]['label'] > train[j]['label'] else 0.0))
            if len(pairs) > 20000:
                idx = rng.choice(len(pairs), 20000, replace=False)
                pairs = [pairs[k] for k in idx]
            for kind, fn in KINDS.items():
                x = np.asarray([np.asarray(fn(train[i]), float) - np.asarray(fn(train[j]), float)
                                for i, j, _ in pairs])
                y = np.asarray([p[2] for p in pairs])
                if len(np.unique(y)) < 2:
                    continue
                scaler = StandardScaler().fit(x)
                model = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
                model.fit(scaler.transform(x), y)

                def score(row):
                    return float(model.coef_[0] @ (np.asarray(fn(row), float) / scaler.scale_))

                # test pairs
                test_pairs = []
                for i in range(len(test)):
                    for j in range(i + 1, len(test)):
                        if test[i]['label'] == test[j]['label']:
                            continue
                        test_pairs.append((i, j))
                if test_pairs:
                    acc = float(np.mean([
                        1.0 if (score(test[i]) - score(test[j])) *
                        (1.0 if test[i]['label'] > test[j]['label'] else -1.0) > 0
                        else 0.0 for i, j in test_pairs]))
                    fold_results[kind]['accuracy'].append(acc)
                fold_results[kind]['spearman'].append(
                    spearman([score(r) for r in test], [r['label'] for r in test]))
        report['groups'][group] = {
            'n': n,
            'label_range': [float(labels.min()), float(labels.max())],
            'signals': {
                k: {'mean_accuracy': float(np.mean(v['accuracy'])) if v['accuracy'] else None,
                    'mean_spearman': float(np.nanmean(v['spearman'])) if v['spearman'] else None}
                for k, v in fold_results.items()},
        }
        print(group, json.dumps(report['groups'][group]['signals'], indent=1), flush=True)

    # aggregate: delta of bfn+composition over composition
    deltas = []
    for group, payload in report['groups'].items():
        comp = payload['signals']['composition']['mean_spearman']
        both = payload['signals']['bfn+composition']['mean_spearman']
        if comp is not None and both is not None and not (np.isnan(comp) or np.isnan(both)):
            deltas.append(both - comp)
            payload['delta_bfn_plus_composition_minus_composition'] = both - comp
    report['summary'] = {
        'n_groups': len(report['groups']),
        'mean_delta_spearman': float(np.mean(deltas)) if deltas else None,
        'median_delta_spearman': float(np.median(deltas)) if deltas else None,
        'groups_bfn_helps': int(sum(1 for d in deltas if d > 0)),
        'verdict': ('bfn adds within-complex ranking information'
                    if deltas and np.mean(deltas) >= 0.05 else
                    'no evidence that bfn adds within-complex ranking information'),
    }
    (OUT / 'within_group_report.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], indent=1))


if __name__ == '__main__':
    main()
