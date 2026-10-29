#!/usr/bin/env python
"""Autoregressive multi-conformer sampling primitives.

The model adapter must return next-residue log probabilities conditioned on the
current prefix for one pose. No fixed conditional-NPZ shortcut is accepted.
"""

from __future__ import annotations

import numpy as np


def aggregate_pose_log_probs(pose_log_probs, weights=None):
    """把多个构象体条件的逐位置分布加权聚合为一个混合分布。

    输入校验刻意严格（≥2 个分布、全部有限、权重非负且和为 1）——
    聚合是采样正确性的关键点，坏输入必须在入口失败而非静默传播。
    """
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
    """逐位置自回归采样一条 H3 序列：每个前缀都重新查询各构象体分布。

    契约：adapter 必须用真实模型对 (prefix, position) 计算 next_log_probs，
    禁止使用缓存的逐位置概率 shortcuts（文件 docstring 的"No fixed
    conditional-NPZ shortcut"即指此）。多 adapter 等权聚合；
    max_substitutions 用尽后余下位置锁定天然残基——限制突变数保证
    采样序列仍贴近天然参考。
    """
    if len(pose_models) < 1:
        raise ValueError("Sampling requires at least one pose model")
    sequence = []
    substitutions = 0
    for position, native_aa in enumerate(native):
        # 每个位置对每个构象体重算分布——这是"真自回归"而非查表的关键
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
        # 减最大值做数值稳定的 softmax
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
