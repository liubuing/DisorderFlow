"""ECLS arithmetic and composition-matched shuffle controls.

The arithmetic here is ported verbatim from the canonical publication scripts so
that per-sequence numbers are directly comparable with the published definition:
- sequence_nll: scripts/benchmark_h3_epitope_delta.py::sequence_nll
- composition_shuffles: scripts/benchmark_h3_scorer_sanity.py::composition_shuffles
"""

from __future__ import annotations

import math
import random

# ProteinMPNN conditional-probability alphabet (21 tokens, X = unknown).
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def sequence_nll(log_probs, sequence: str, indices) -> float:
    """Mean per-residue NLL of ``sequence`` over the given 0-based indices."""
    if len(indices) != len(sequence):
        raise ValueError("index count must match sequence length")
    if not sequence:
        raise ValueError("empty sequence")
    return -sum(
        float(log_probs[index, MPNN_ALPHABET.index(aa)])
        for index, aa in zip(indices, sequence)
    ) / len(sequence)


def _multiset_permutations(sequence: str) -> int:
    total = math.factorial(len(sequence))
    for aa in set(sequence):
        total //= math.factorial(sequence.count(aa))
    return total


def composition_shuffles(sequence: str, n: int, seed: int) -> list[str]:
    """Up to ``n`` unique composition-matched shuffles with a fixed RNG stream.

    Identical seeding and uniqueness handling to the publication scripts: a
    ``random.Random(seed)`` stream, native excluded, capped by the multiset
    permutation count minus one, with the same attempt budget.
    """
    rng = random.Random(seed)
    native = str(sequence)
    seen = {native}
    controls: list[str] = []
    attempts = 0
    max_unique = _multiset_permutations(native) - 1
    target = min(n, max_unique)
    while len(controls) < target and attempts < max(n * 100, 1000):
        attempts += 1
        chars = list(native)
        rng.shuffle(chars)
        shuffled = "".join(chars)
        if shuffled not in seen:
            seen.add(shuffled)
            controls.append(shuffled)
    return controls


def percentile_lower_is_better(native: float, controls) -> float:
    """Fraction of controls strictly worse (higher NLL) than native."""
    return sum(value > native for value in controls) / len(controls)


def ecls_delta(complex_nll: float, apo_nll: float) -> float:
    """Per-sequence ECLS: NLL(complex) - NLL(peptide-stripped Fab)."""
    return complex_nll - apo_nll


def advantage_summary(native_ecls: float, shuffle_ecls: list[float]) -> dict:
    """advantage = mean(shuffle ECLS) - native ECLS; positive favours native."""
    if not shuffle_ecls:
        raise ValueError("advantage requires at least one shuffle control")
    shuffle_mean = sum(shuffle_ecls) / len(shuffle_ecls)
    return {
        "shuffle_count": len(shuffle_ecls),
        "shuffle_mean_ecls": round(shuffle_mean, 6),
        "ecls_advantage": round(shuffle_mean - native_ecls, 6),
        "native_ecls_percentile": round(
            percentile_lower_is_better(native_ecls, shuffle_ecls), 6),
    }
