"""Parity tests between the standalone ecls-scorer tool and the canonical scripts.

Kept in the root test tree so the default pytest run (testpaths = tests) also
guards the tool's arithmetic, without requiring torch.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOL_ROOT = ROOT / "tools" / "ecls_scorer"
sys.path.insert(0, str(TOOL_ROOT))

from ecls_scorer.scoring import (  # noqa: E402
    MPNN_ALPHABET as TOOL_ALPHABET,
    composition_shuffles as tool_shuffles,
    percentile_lower_is_better as tool_percentile,
    sequence_nll as tool_sequence_nll,
)

sys.path.insert(0, str(ROOT))

from scripts.benchmark_h3_epitope_delta import (  # noqa: E402
    MPNN_ALPHABET as SCRIPT_ALPHABET,
    percentile_lower_is_better as script_percentile,
    sequence_nll as script_sequence_nll,
)
from scripts.benchmark_h3_scorer_sanity import (  # noqa: E402
    composition_shuffles as script_shuffles,
)


def test_alphabets_match():
    assert TOOL_ALPHABET == SCRIPT_ALPHABET == "ACDEFGHIKLMNPQRSTVWYX"


def test_sequence_nll_parity_with_canonical_script():
    rng = np.random.default_rng(7)
    log_probs = rng.uniform(-20.0, 0.0, size=(40, len(TOOL_ALPHABET)))
    sequence = "CARRYYDHYSGSSDYWG"
    indices = list(range(5, 5 + len(sequence)))
    assert tool_sequence_nll(log_probs, sequence, indices) == \
        script_sequence_nll(log_probs, sequence, indices)


def test_composition_shuffles_parity_with_canonical_script():
    sequence = "ARSGYSSSWYFDY"
    assert tool_shuffles(sequence, 200, seed=2309) == \
        script_shuffles(sequence, 200, seed=2309)
    assert tool_shuffles(sequence, 200, seed=2281) == \
        script_shuffles(sequence, 200, seed=2281)


def test_percentile_parity_with_canonical_script():
    native, controls = 1.5, [0.2, 1.4, 1.6, 9.0]
    assert tool_percentile(native, controls) == script_percentile(native, controls)


def test_tool_package_importable_without_torch():
    spec = importlib.util.find_spec("ecls_scorer")
    assert spec is not None
