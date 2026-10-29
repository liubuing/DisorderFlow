#!/usr/bin/env python
"""Generate matched positionwise PoE and single-state H3 pilot candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
# ProteinMPNN 21 字母表（含 X 占位）；抽样与评分均在该表内进行。
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def sha256(path):
    # 计算文件 SHA-256；NPZ/配置/评分工件全部先验哈再使用（不匹配即抛错）。
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_npz(record):
    # 加载单构象的条件对数概率 NPZ 并校验哈希、有限性；
    # 返回 (log_p, H3 设计位点索引)。
    path = ROOT / record["npz"]
    if sha256(path) != record["npz_sha256"]:
        raise ValueError(f"Conditional NPZ hash mismatch: {path}")
    payload = np.load(path)
    indices = [
        int(index) for index in np.flatnonzero(np.asarray(payload["design_mask"]) > 0)
    ]
    log_p = np.asarray(payload["log_p"][0], dtype=np.float64)
    if not np.isfinite(log_p[indices]).all():
        raise ValueError(f"Non-finite conditional probabilities: {path}")
    return log_p, indices


def conditional_distribution(component, arm):
    # 构建两臂的逐位点条件分布（对比实验的核心）:
    #   - single_state: 只用第一个构象（pose 0）的 complex 态 log_p;
    #   - ensemble_product_of_experts: 对全部构象的 log_p 取逐点平均
    #     （对数域平均即概率乘积——"专家乘积"集成）。
    # 末尾做 max 减法后 softmax 归一化，保证数值稳定且每行和为 1。
    rows = component["conformers"]
    states = [row["complex"] for row in rows]
    log_probs = []
    indices = None
    for row in states:
        log_p, observed = load_npz(row)
        if indices is None:
            indices = observed
        elif indices != observed:
            raise ValueError("H3 design masks differ across conformers")
        log_probs.append(log_p[indices])
    if arm == "single_state":
        combined = log_probs[0]
    elif arm == "ensemble_product_of_experts":
        combined = np.mean(np.stack(log_probs, axis=0), axis=0)
    else:
        raise ValueError(f"Unknown arm: {arm}")
    combined = combined - np.max(combined, axis=-1, keepdims=True)
    probabilities = np.exp(combined)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return probabilities, indices


def sample_h3(native, probabilities, rng, substitutions):
    # 以原生 H3 为骨架，随机选 substitutions 个位置做条件采样替换;
    # 原生氨基酸在该位点的概率被置 0，强制保证产生真实突变
    # （否则候选与原生相同，会污染"突变 vs 原生"的对照设计）。
    if substitutions < 1 or substitutions > len(native):
        raise ValueError("Invalid substitution count")
    sequence = list(native)
    positions = rng.choice(len(native), size=substitutions, replace=False)
    for position in positions:
        row = probabilities[position].copy()
        row[ALPHABET.index(native[position])] = 0.0
        row /= row.sum()
        sequence[position] = ALPHABET[int(rng.choice(len(ALPHABET), p=row))]
    return "".join(sequence)


def candidate_score(sequence, native, probabilities):
    # 生成器侧评分: 候选序列在该臂条件分布下的联合对数概率
    # （log sum over positions）。仅用于生成端记账，不是性能端点。
    return float(sum(
        np.log(probabilities[index, ALPHABET.index(aa)])
        for index, aa in enumerate(sequence)
    ))


def generate_component(component, seeds, substitution_schedule):
    # 对单个组件在两臂上按冻结预算生成候选:
    #   - 每个 seed 用 default_rng(seed) 独立播种（固定种子保证可复现）;
    #   - substitution_schedule 给出每个样本的目标突变数;
    #   - 重复候选直接重采（最多 10000 次尝试，超出即失败——预算网格
    #     预期可被满足，失败说明分布过窄，fail-closed 而非静默重复）;
    #   - 每臂附原生对照（substitutions=0）作为零突变参照。
    output = {}
    for arm in ("ensemble_product_of_experts", "single_state"):
        probabilities, indices = conditional_distribution(component, arm)
        if len(indices) != len(component["native_h3"]):
            raise ValueError("H3 design mask length mismatch")
        arm_rows = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            accepted = []
            for sample_index, substitutions in enumerate(substitution_schedule):
                attempts = 0
                while attempts < 10000:
                    attempts += 1
                    candidate = sample_h3(
                        component["native_h3"], probabilities, rng, substitutions
                    )
                    if candidate not in accepted:
                        accepted.append(candidate)
                        arm_rows.append({
                            "seed": seed,
                            "sample_index": sample_index,
                            "substitutions": substitutions,
                            "sequence": candidate,
                            "generation_score": candidate_score(
                                candidate, component["native_h3"], probabilities
                            ),
                        })
                        break
                else:
                    raise RuntimeError(
                        f"Unable to fill {component['component_id']} {arm} seed {seed}"
                    )
        output[arm] = arm_rows
        output[f"{arm}_native_control"] = {
            "sequence": component["native_h3"],
            "substitutions": 0,
            "generation_score": candidate_score(
                component["native_h3"], component["native_h3"], probabilities
            ),
        }
    return output


def generate(config_path, score_path=None, out_path=None):
    # 主生成流程: 拒绝从 superseded 契约生成 → 校验条件评分工件哈希 →
    # 只保留注册表中非 sensitivity_only 的"干净"组件并核对数量与
    # matched_budget 一致 → 生成候选并落盘（输出已存在则拒绝覆盖）。
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    if str(config.get("status", "")).startswith("superseded"):
        raise ValueError("Refusing to generate from a superseded pilot contract")
    score_item = config["conditional_score_artifact"]
    score_path = Path(score_path or ROOT / score_item["path"])
    if sha256(score_path) != score_item["sha256"]:
        raise ValueError("Conditional score artifact hash mismatch")
    score = json.loads(score_path.read_text(encoding="ascii"))
    clean_ids = {
        row["component_id"]
        for row in yaml.safe_load(
            (ROOT / config["component_registry"]["path"]).read_text(encoding="ascii")
        )["components"]
        if not row.get("sensitivity_only", False)
    }
    components = [row for row in score["component_scores"] if row["component_id"] in clean_ids]
    expected = config["matched_budget"]["components"]
    if len(components) != expected:
        raise ValueError(f"Expected {expected} clean components, found {len(components)}")
    budget = config["matched_budget"]
    generated = {
        component["component_id"]: generate_component(
            component,
            budget["seeds"],
            budget["substitutions_per_seed"],
        )
        for component in components
    }
    output = {
        "schema_version": 1,
        "status": "complete",
        "classification": config["classification"],
        "generator": "positionwise_conditional_product_of_experts_vs_pose_0_single_state",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "score_artifact": {"path": str(score_path.relative_to(ROOT)), "sha256": sha256(score_path)},
        "budget": budget,
        "components": generated,
        "claim_boundary": config["claim_boundary"],
    }
    output_path = Path(out_path or ROOT / config["outputs"]["raw_candidates"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite candidates: {output_path}")
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    return output


def main():
    # 命令行入口：默认使用冻结的前瞻性 pilot v2 配置。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmarks/idp_ensemble_prospective_pilot_v2.yml")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    print(json.dumps(generate(ROOT / args.config, out_path=args.out), indent=2))


if __name__ == "__main__":
    main()
