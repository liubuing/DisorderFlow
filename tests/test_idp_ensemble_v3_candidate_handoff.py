"""IDP 集成 v3 候选交接测试：验证 finalize 汇总与 blind 盲法的数量与脱敏契约。

守护 v3 候选交接管线（finalize_idp_ensemble_v3_candidates + blind_idp_ensemble_v3_candidates）：
12 组件 × 2 臂 × 4 突变桶 = 96 条候选必须全部汇总、
盲法清单 108 条（96 候选 + 12 天然对照）且不携带 sequence/arm 字段（脱敏）、
密钥文件完整保存全部构体映射——防止实验执行者看到臂别或序列导致盲法失效。
"""
import json

from scripts.blind_idp_ensemble_v3_candidates import blind
from scripts.finalize_idp_ensemble_v3_candidates import finalize


# 交接契约：finalize 必须收集全部 96 条候选；blind 清单必须 108 条、
# 公开侧零序列/零臂信息、密钥侧完整对应——防止盲化不彻底或候选丢失。
def test_finalize_and_blind_v3_candidates(tmp_path):
    panel = {
        "components": [{"component_id": f"C{i:02d}"} for i in range(12)],
        "seeds": [12001, 12011, 12021, 12031],
        "mutation_space": {"substitution_buckets": [2, 4, 6, 8]},
    }
    panel_path = tmp_path / "panel.json"
    panel_path.write_text(json.dumps(panel), encoding="ascii")
    candidate_dir = tmp_path / "components"
    candidate_dir.mkdir()
    native = "AAAAAAAAAA"
    for component in panel["components"]:
        component_id = component["component_id"]
        arms = {}
        for arm in ("ensemble", "single_state"):
            rows = []
            for index, bucket in enumerate([2, 4, 6, 8]):
                sequence = "C" * bucket + "A" * (len(native) - bucket)
                rows.append({
                    "seed": panel["seeds"][index],
                    "substitution_bucket": bucket,
                    "substitutions": bucket,
                    "sequence": sequence,
                })
            arms[arm] = {
                "pose_count": 2 if arm == "ensemble" else 1,
                "candidates": rows,
                "native_control": {"sequence": native, "substitutions": 0},
            }
        (candidate_dir / f"{component_id}.json").write_text(json.dumps({
            "status": "component_candidates_complete",
            "component_id": component_id,
            "target": "target",
            "native_h3": native,
            "seeds": panel["seeds"],
            "substitution_buckets": [2, 4, 6, 8],
            "arms": arms,
        }), encoding="ascii")
    candidates_path = tmp_path / "candidates.json"
    result = finalize(panel_path, candidate_dir, candidates_path)
    assert result["candidate_count"] == 96
    manifest_path = tmp_path / "manifest.json"
    key_path = tmp_path / "key.json"
    manifest = blind(candidates_path, panel_path, manifest_path, key_path)
    assert manifest["construct_count"] == 108
    assert all("sequence" not in row and "arm" not in row for row in manifest["constructs"])
    assert len(json.loads(key_path.read_text(encoding="ascii"))["constructs"]) == 108
