#!/usr/bin/env python
"""Build the calibration-extension generation/AF2/LMDB pipeline configs with hash chaining."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

MERGED = ROOT / "data/candidate_interface_external_calibration_v1/merged_structural_manifest.json"
AUDIT = ROOT / "data/candidate_interface_external_calibration_v1/isolation_audit_consolidated.json"
EXT_NS = "data/candidate_interface_multiscaffold_calibration_ext_v1"
RES = "results/candidate_interface_multiscaffold_calibration_ext"

HOLDOUT = ROOT / EXT_NS / "holdout_manifest.json"
LINEAGE = ROOT / "publication/multiscaffold_v2_calibration_ext_checkpoint_lineage.json"
PROTOCOL = ROOT / "configs/benchmarks/multiscaffold_confirmatory_v2_calibration_ext.yml"
GENCFG = ROOT / "configs/benchmarks/multiscaffold_confirmatory_v2_calibration_ext_generation.yml"
SELCFG = ROOT / "configs/benchmarks/multiscaffold_confirmatory_v2_calibration_ext_selection.yml"
SPLIT = ROOT / EXT_NS / "split_manifest.json"
AF2C1 = ROOT / "configs/benchmarks/candidate_interface_multiscaffold_calibration_ext_af2.yml"
AF2C2 = ROOT / "configs/benchmarks/candidate_interface_multiscaffold_calibration_ext_af2_model2.yml"

RAW_GEN = ROOT / RES / "raw_generation.json"
RAW_AUDIT = ROOT / RES / "raw_generation_audit.json"
SELECTION = ROOT / RES / "pre_af2_selection.json"

# 四条生成臂：当前 BFN（disorder 开/关）、stage-A BFN 与 ProteinMPNN 基线
ARMS = ["current_bfn_disorder_on", "current_bfn_disorder_off",
        "stage_a_antibody_bfn", "proteinmpnn"]
SEEDS = [2609, 2617, 2621]  # 固定生成种子，保证可复现
CANDIDATES_PER_SEED = 8  # 每种子候选数
SLOTS_PER_ARM_COMPONENT = 3  # 每臂每组件入选的候选槽数


def sha256(path: Path) -> str:
    # 流式计算文件 SHA256（1 MiB 分块，适配大文件）
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_once(path: Path, payload: dict) -> None:
    # 一次性写出 JSON：已存在则拒绝覆盖，保证冻结产物不被篡改
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")


def write_yaml_once(path: Path, payload: dict) -> None:
    # 一次性写出 YAML：已存在则拒绝覆盖
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(yaml.safe_dump(payload, sort_keys=False) + "\n", encoding="utf-8")


def component_specs() -> list[dict]:
    # 从合并结构清单与隔离审计中提取每个独立组件及其代表记录
    merged = json.loads(MERGED.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    by_instance = {str(r["instance"]): r for r in merged["records"]}
    specs = []
    for component in audit["components"]:
        rep = by_instance[component["representative_id"]]
        specs.append({
            "component_id": component["component_id"],
            "members": component["members"],
            "representative_id": component["representative_id"],
            "representative": {
                "instance": rep["instance"],
                "pdb_id": rep["pdb_id"],
                "heavy_chain": rep["original_chain_ids"]["heavy"],
                "light_chain": rep["original_chain_ids"]["light"],
                "antigen_chains": [rep["original_chain_ids"]["antigen"]],
                "antigen_chain": rep["original_chain_ids"]["antigen"],
                "antigen_name": rep["title"],
                "resolution": rep["resolution"],
                "cif_path": rep["source_cif"],
                "heavy_sequence": rep["heavy_sequence"],
                "light_sequence": rep["light_sequence"],
                "antigen_sequence": rep["antigen_sequence"],
                "vh_sequence": rep["vh_sequence"],
                "vl_sequence": rep["vl_sequence"],
                "cdr_sequences": rep["cdr_sequences"],
                "paired_cdr_sequence": rep["paired_cdr_sequence"],
                "cdr_h3_sequence": rep["cdr_h3_sequence"],
                "h3_heavy_indices_zero_based": rep["h3_heavy_indices_zero_based"],
                "h3_definition": rep["h3_definition"],
            },
        })
    return specs


def build_holdout() -> dict:
    # 阶段1：冻结校准扩展的 holdout manifest（隔离阈值方案 vhvl_90_pairedcdr_70）
    specs = component_specs()
    manifest = {
        "schema_version": 1,
        "status": "frozen_before_calibration_extension_generation",
        "classification": "prospective_internal_calibration_extension; not external confirmation",
        "source_isolation_audit": AUDIT.relative_to(ROOT).as_posix(),
        "source_isolation_audit_sha256": sha256(AUDIT),
        "scheme": "vhvl_90_pairedcdr_70",
        "thresholds": {
            "vh": 0.9, "vl": 0.9, "cdr_h3": 0.5, "paired_cdr": 0.7, "antigen": 0.3,
        },
        "coverage": 0.8,
        "n_all_axis_components": len(specs),
        "n_representatives": len(specs),
        "representative_ids": [s["representative_id"] for s in specs],
        "components": [
            {"component_id": s["component_id"], "members": s["members"],
             "representative_id": s["representative_id"],
             "representative": s["representative"]}
            for s in specs
        ],
    }
    write_json_once(HOLDOUT, manifest)
    return manifest


def build_lineage() -> dict:
    # 阶段2：冻结 checkpoint 血缘（stage-A 随机初始化 -> stage-B disorder 监督）
    lineage = {
        "schema_version": 1,
        "status": "frozen_before_calibration_extension_generation",
        "holdout_manifest_sha256": sha256(HOLDOUT),
        "stage_a": {
            "initialization": "random",
            "config": "configs/train/bfn_multiscaffold_v2_stage_a_random.yml",
            "config_sha256": "57935148ea0b65234436302827a62b280c2311e8e51a86b1e00e9bace29ba7b6",
            "checkpoint": "logs/bfn_multiscaffold_v2_stage_a_random_2026_08_06__08_43_37_multiscaffold_v2_random_s2647/checkpoints/best.pt",
            "checkpoint_sha256": "00ae022962bb35bd1ab65e4fea912715d3e99d09c0858ed6b8452ece32e9678f",
            "iteration": 100000,
            "validation_loss": 2.4305,
        },
        "stage_b": {
            "initialization": "stage_a_only",
            "config": "configs/train/bfn_multiscaffold_v2_stage_b_disorder.yml",
            "config_sha256": "426da178df61d8d11ab554a937154128aca36d3b302178b39a2755c2973443c2",
            "source_checkpoint_sha256": "00ae022962bb35bd1ab65e4fea912715d3e99d09c0858ed6b8452ece32e9678f",
            "checkpoint": "logs/bfn_multiscaffold_v2_stage_b_disorder_2026_08_06__19_00_50_multiscaffold_v2_stagea100k_s2657/checkpoints/best.pt",
            "checkpoint_sha256": "5cf0516f076492c46a73e0c5d53814d8cd9cf427472cc4e76aff42f5a584a8c3",
            "iteration": 2200,
            "validation_disorder_roc_auc": 0.9051453809079083,
        },
        "prohibited_initializers": ["balanced_v4", "parent_bfn", "legacy_disorderflow"],
        "claim_boundary": "checkpoint provenance only; no generation or benchmark result",
    }
    write_json_once(LINEAGE, lineage)
    return lineage


def build_protocol() -> dict:
    # 阶段3：冻结隔离协议（阈值/CDR 定义/生成与选择约束）
    protocol = {
        "schema_version": 2,
        "status": "frozen_before_calibration_extension_generation",
        "classification": "prospective_internal_calibration_extension",
        "isolation_audit": AUDIT.relative_to(ROOT).as_posix(),
        "isolation_audit_sha256": sha256(AUDIT),
        "holdout_manifest": HOLDOUT.relative_to(ROOT).as_posix(),
        "holdout_manifest_sha256": sha256(HOLDOUT),
        "isolation": {
            "coverage": 0.8,
            "hard_axes": {
                "pdb_id": "exact",
                "antigen": {"minimum_identity": 0.3, "coverage_mode": 0},
                "cdr_h3": {"minimum_identity": 0.5, "coverage_mode": 0},
                "paired_cdr": {"minimum_identity": 0.7, "coverage_mode": 0},
                "vh": {"minimum_identity": 0.9, "coverage_mode": 0},
                "vl": {"minimum_identity": 0.9, "coverage_mode": 0},
            },
            "cdr_definition": {
                "scheme": "chothia", "h1": "26-32", "h2": "52-56",
                "h3": "93-102_anchor_bounded", "l1": "24-34", "l2": "50-56",
                "l3": "89-97",
            },
        },
        "holdout": {
            "reference_independent_records": len(component_specs()),
            "all_axis_components": len(component_specs()),
            "representatives": len(component_specs()),
            "minimum_required_components": 5,
        },
        "training_lineage": {
            "stage_a": "random_initialization_filtered_antibody_pretraining",
            "stage_b": "stage_a_only_initialization_disorder_supervision",
            "prohibited_initializers": ["balanced_v4", "parent_bfn", "legacy_disorderflow"],
        },
        "generation": {
            "seeds": SEEDS,
            "candidates_per_seed": CANDIDATES_PER_SEED,
            "arms": ARMS,
            "failed_candidates_remain_failures": True,
        },
        "selection": {
            "freeze_before_af2": True,
            "candidates_per_arm_per_component": SLOTS_PER_ARM_COMPONENT,
        },
        "claim_boundary": "calibration extension: AF2-derived development labels only; not binding or external confirmation",
    }
    write_yaml_once(PROTOCOL, protocol)
    return protocol


def build_genconfig() -> dict:
    # 阶段4：冻结原始生成配置（四臂 x 三种子 x 每种子 8 条，保留失败与重复）
    config = {
        "schema_version": 1,
        "status": "frozen_before_calibration_extension_candidate_generation",
        "protocol": PROTOCOL.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256(PROTOCOL),
        "holdout_manifest": HOLDOUT.relative_to(ROOT).as_posix(),
        "holdout_manifest_sha256": sha256(HOLDOUT),
        "checkpoint_lineage": LINEAGE.relative_to(ROOT).as_posix(),
        "generation": {
            "seeds": SEEDS,
            "candidates_per_seed": CANDIDATES_PER_SEED,
            "preserve_native_h3_length": True,
            "canonical_chain_map": {"heavy": "H", "light": "L", "antigen": "P"},
            "raw_attempts_expected": len(component_specs()) * len(ARMS) * len(SEEDS) * CANDIDATES_PER_SEED,
            "retain_native_candidates": True,
            "retain_duplicate_candidates": True,
            "failed_candidates_remain_failures": True,
            "arms": {
                "current_bfn_disorder_on": {
                    "type": "bfn",
                    "checkpoint": "logs/bfn_multiscaffold_v2_stage_b_disorder_2026_08_06__19_00_50_multiscaffold_v2_stagea100k_s2657/checkpoints/best.pt",
                    "checkpoint_sha256": "5cf0516f076492c46a73e0c5d53814d8cd9cf427472cc4e76aff42f5a584a8c3",
                    "stochastic": True,
                    "disorder_profile": "stage_b_head_raw_sigmoid_latent_routing",
                    "disorder_guided_strength": 1.0,
                },
                "current_bfn_disorder_off": {
                    "type": "bfn",
                    "checkpoint": "logs/bfn_multiscaffold_v2_stage_b_disorder_2026_08_06__19_00_50_multiscaffold_v2_stagea100k_s2657/checkpoints/best.pt",
                    "checkpoint_sha256": "5cf0516f076492c46a73e0c5d53814d8cd9cf427472cc4e76aff42f5a584a8c3",
                    "stochastic": True,
                    "disorder_profile": "none",
                },
                "stage_a_antibody_bfn": {
                    "type": "bfn",
                    "checkpoint": "logs/bfn_multiscaffold_v2_stage_a_random_2026_08_06__08_43_37_multiscaffold_v2_random_s2647/checkpoints/best.pt",
                    "checkpoint_sha256": "00ae022962bb35bd1ab65e4fea912715d3e99d09c0858ed6b8452ece32e9678f",
                    "stochastic": True,
                    "disorder_profile": "none",
                },
                "proteinmpnn": {
                    "type": "proteinmpnn",
                    "script": "ProteinMPNN/protein_mpnn_run.py",
                    "weights": "ProteinMPNN/vanilla_model_weights",
                    "model_name": "v_48_020",
                    "model_weights_sha256": "c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd",
                    "temperature": 0.1,
                },
            },
        },
        "output": {
            "path": RAW_GEN.relative_to(ROOT).as_posix(),
            "schema": "one_row_per_attempt",
            "candidate_selection_permitted": False,
        },
        "claim_boundary": "raw prospective internal calibration-extension generation; no candidate selection or confirmatory endpoint",
    }
    write_yaml_once(GENCFG, config)
    return config


def build_selconfig() -> dict:
    # 阶段5：冻结 AF2 前的候选选择配置（仅基于序列多样性，禁用生成器/置信度等输入）
    config = {
        "schema_version": 1,
        "status": "frozen_before_calibration_extension_selection",
        "generation_config": GENCFG.relative_to(ROOT).as_posix(),
        "generation_config_sha256": sha256(GENCFG),
        "raw_generation": RAW_GEN.relative_to(ROOT).as_posix(),
        "raw_generation_sha256": sha256(RAW_GEN),
        "raw_generation_audit": RAW_AUDIT.relative_to(ROOT).as_posix(),
        "raw_generation_audit_sha256": sha256(RAW_AUDIT),
        "selection": {
            "slots_per_arm_component": SLOTS_PER_ARM_COMPONENT,
            "require_unique_h3_within_arm_component": True,
            "insufficient_unique_candidates": "retain_failed_selection_slots",
            "objective_priority": [
                "maximize_distinct_generation_seeds",
                "maximize_minimum_pairwise_hamming_distance",
                "maximize_total_pairwise_hamming_distance",
                "minimize_sum_sample_indices",
                "lexicographic_attempt_id_tiebreak",
            ],
            "prohibited_inputs": [
                "generator_score", "bfn_confidence", "af2_outputs",
                "interface_scores", "native_recovery",
            ],
        },
        "output": SELECTION.relative_to(ROOT).as_posix(),
        "expected_slots": len(component_specs()) * len(ARMS) * SLOTS_PER_ARM_COMPONENT,
        "claim_boundary": "sequence-diversity selection only; no structural or performance evidence",
    }
    write_yaml_once(SELCFG, config)
    return config


def build_split() -> dict:
    # 阶段6：冻结划分 manifest（train/test 为空，全部组件归入 calibration）
    manifest = {
        "schema_version": "candidate_interface_multiscaffold_split_v1",
        "status": "frozen_before_calibration_extension_af2",
        "classification": "calibration extension split; training and test are empty",
        "source_holdout_manifest": HOLDOUT.relative_to(ROOT).as_posix(),
        "source_holdout_manifest_sha256": sha256(HOLDOUT),
        "source_selection": SELECTION.relative_to(ROOT).as_posix(),
        "source_selection_sha256": sha256(SELECTION),
        "split_policy": "calibration-only external scaffold extension",
        "mask_contract": "H3-only using h3_heavy_indices_zero_based from the frozen source manifest",
        "train": [],
        "calibration": [s["component_id"] for s in component_specs()],
        "test": [],
        "test_usage_policy": "no test components exist in the calibration extension",
    }
    write_json_once(SPLIT, manifest)
    return manifest


def build_af2_configs() -> None:
    # 阶段7：生成两个 AF2 multimer 模型的配置（含 native/组合打乱对照与实体数量预期）
    selection_sha = sha256(SELECTION)
    holdout_sha = sha256(HOLDOUT)
    split_sha = sha256(SPLIT)
    selcfg_sha = sha256(SELCFG)
    for model_number in (1, 2):
        params = f"C:/Users/34342/.cache/colabfold/params/params_model_{model_number}_multimer_v3.npz"
        params_sha = "611da8fc7478928f68de12e8b226260ef1f4ce62bcc29b008572e52f4f212959" if model_number == 1 else "51362b0844382ae0f5720c59b81dd13a43eea40fbf9995dd2573bdab88865378"
        config = {
            "schema_version": 1,
            "status": "frozen_before_calibration_extension_af2",
            "protocol_id": f"alphafold2_multimer_v3_model_{model_number}_single_sequence",
            "selection_config": SELCFG.relative_to(ROOT).as_posix(),
            "selection_config_sha256": selcfg_sha,
            "selection": SELECTION.relative_to(ROOT).as_posix(),
            "selection_sha256": selection_sha,
            "holdout_manifest": HOLDOUT.relative_to(ROOT).as_posix(),
            "holdout_manifest_sha256": holdout_sha,
            "split_manifest": SPLIT.relative_to(ROOT).as_posix(),
            "split_manifest_sha256": split_sha,
            "controls": {
                "per_component": ["native", "composition_shuffle"],
                "composition_shuffle_seed": 7307,
                "require_non_native_shuffle": True,
            },
            "af2": {
                "backend": "modules.af2_jax_runner.run_multimer_prediction",
                "wsl_distribution": "Ubuntu-24.04-D",
                "environment": "venv_wsl",
                "model_type": "alphafold2_multimer_v3",
                "model_number": model_number,
                "model_parameters": params,
                "model_parameters_sha256": params_sha,
                "seeds": [7103, 7111, 7121],
                "recycles": 3,
                "msa_mode": "single_sequence_two_identical_rows",
                "templates": "disabled_zero_features",
                "models_per_seed": 1,
                "chunk_size": 24,
                "return_structures": True,
                "retain_full_pae_matrix": True,
            },
            "expected": {
                "selected_candidates": len(component_specs()) * len(ARMS) * SLOTS_PER_ARM_COMPONENT,
                "selection_failed_slots": 0,
                "controls": len(component_specs()) * 2,
                "prediction_entities": len(component_specs()) * len(ARMS) * SLOTS_PER_ARM_COMPONENT + len(component_specs()) * 2,
                "prediction_slots": (len(component_specs()) * len(ARMS) * SLOTS_PER_ARM_COMPONENT + len(component_specs()) * 2) * 3,
            },
            "output": {
                "results": (f"{RES}/af2/results.json" if model_number == 1
                            else f"{RES}/af2_model2/results.json"),
                "structures": f"{RES}/af2/pdb" if model_number == 1 else f"{RES}/af2_model2/pdb",
                "pae_sidecars": f"{RES}/af2/pae" if model_number == 1 else f"{RES}/af2_model2/pae",
            },
            "claim_boundary": "AF2-derived development labels for the calibration extension; not binding or external confirmation",
        }
        target = AF2C1 if model_number == 1 else AF2C2
        write_yaml_once(target, config)


def main():
    # 分阶段构建冻结产物；每阶段写出前打印路径与哈希，哈希链贯穿各阶段
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=["holdout", "lineage", "protocol", "genconfig",
                                 "selconfig", "split", "af2"])
    args = parser.parse_args()
    stage = args.stage
    targets = {
        "holdout": HOLDOUT, "lineage": LINEAGE, "protocol": PROTOCOL,
        "genconfig": GENCFG, "selconfig": SELCFG, "split": SPLIT,
    }
    if stage == "holdout":
        build_holdout()
    elif stage == "lineage":
        build_lineage()
    elif stage == "protocol":
        build_protocol()
    elif stage == "genconfig":
        build_genconfig()
    elif stage == "selconfig":
        build_selconfig()
    elif stage == "split":
        build_split()
    else:
        build_af2_configs()
        print(json.dumps({"stage": "af2", "configs": [
            AF2C1.relative_to(ROOT).as_posix(), AF2C2.relative_to(ROOT).as_posix()]}, indent=2))
        return
    print(json.dumps({"stage": stage, "path": targets[stage].relative_to(ROOT).as_posix(),
                      "sha256": sha256(targets[stage])}, indent=2))


if __name__ == "__main__":
    main()
