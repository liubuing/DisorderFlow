#!/usr/bin/env python
"""Autoregressive multi-conformer sampling primitives.

The model adapter must return next-residue log probabilities conditioned on the
current prefix for one pose. No fixed conditional-NPZ shortcut is accepted.
"""

from __future__ import annotations

import numpy as np


def aggregate_pose_log_probs(pose_log_probs, weights=None):
    values = np.asarray(pose_log_probs, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("At least two pose-conditioned distributions are required")
    if not np.isfinite(values).all():
        raise ValueError("Pose-conditioned log probabilities must be finite")
    if weights is None:
        weights = np.full(values.shape[0], 1.0 / values.shape[0])
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (values.shape[0],) or np.any(weights < 0):
        raise ValueError("Invalid pose weights")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Pose weights must sum to one")
    return np.average(values, axis=0, weights=weights)


def sample_autoregressive(native, pose_models, rng, max_substitutions, alphabet):
    """Sample one H3 sequence by recomputing every pose distribution per prefix.

    Each adapter receives the same prefix and position. It must implement
    ``next_log_probs(prefix, position)`` using the actual model, not cached
    per-position probabilities. A single adapter may itself aggregate a pose
    batch; two or more adapters are aggregated here with equal weights.
    """
    if len(pose_models) < 1:
        raise ValueError("Sampling requires at least one pose model")
    sequence = []
    substitutions = 0
    for position, native_aa in enumerate(native):
        pose_log_probs = [
            np.asarray(model.next_log_probs("".join(sequence), position),
                       dtype=np.float64)
            for model in pose_models
        ]
        if len(pose_log_probs) == 1:
            log_probs = pose_log_probs[0]
            if not np.isfinite(log_probs).all():
                raise ValueError("Pose-conditioned log probabilities must be finite")
        else:
            log_probs = aggregate_pose_log_probs(pose_log_probs)
        log_probs = log_probs - np.max(log_probs)
        probabilities = np.exp(log_probs)
        probabilities /= probabilities.sum()
        if substitutions >= max_substitutions:
            choice = alphabet.index(native_aa)
        else:
            choice = int(rng.choice(len(alphabet), p=probabilities))
            substitutions += choice != alphabet.index(native_aa)
        sequence.append(alphabet[choice])
    if substitutions > max_substitutions:
        raise RuntimeError("Autoregressive sampler exceeded substitution budget")
    return "".join(sequence)
