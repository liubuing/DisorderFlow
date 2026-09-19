import json
import numpy as np

from scripts.benchmark_aayl_learnability import OUT, load, metrics, mutations, top_weights


def test_tied_screen_is_order_invariant_and_random_expectation():
    y = np.arange(10, dtype=float)
    tied = np.ones(10)
    assert np.isclose(metrics(y, tied)["top20_recall"], .2)
    assert np.isclose(metrics(y, tied)["enrichment"], 1)
    assert metrics(y, tied)["pair_accuracy"] == .5
    p = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    assert np.isclose(top_weights(p, 2).sum(), 2)
    assert metrics(y, p) == metrics(y[::-1], p[::-1])
    assert metrics(y, y)["top20_recall"] == 1


def test_coverage_requires_exact_substitution_not_just_position():
    seen = mutations("CAAA", "AAAA") | mutations("ADAA", "AAAA")
    assert mutations("CDAA", "AAAA") <= seen
    assert not mutations("CEAA", "AAAA") <= seen


def test_frozen_splits_have_no_candidate_leakage_and_exact_budgets():
    _, rows, changes = load()
    manifest = json.loads((OUT / "splits.json").read_text())
    assert manifest["rows"] == [r["poi"] for r in rows]
    assert len(manifest["splits"]) == 22
    for s in manifest["splits"]:
        train, test = s["train"], s["test"]
        assert not set(train) & set(test)
        assert not {rows[i]["heavy_sequence"] for i in train} & {rows[i]["heavy_sequence"] for i in test}
        assert all(len(changes[i]) == 1 for i in train)
        assert all(len(changes[i]) in (2, 3) for i in test)
        assert all(rows[i]["family"] == s["family"] for i in train + test)
        if isinstance(s["budget"], int):
            assert len(train) == s["budget"]
        else:
            assert len(train) == (89 if s["family"] == "AAYL49" else 55)
        seen = frozenset().union(*(changes[i] for i in train))
        assert s["covered"] == [i for i in test if changes[i] <= seen]
