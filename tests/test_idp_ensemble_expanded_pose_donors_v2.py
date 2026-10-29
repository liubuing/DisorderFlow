"""IDP 集成扩展 pose 供体 v2 测试：验证仅实验就绪组件进入供体清单。

守护 scripts.materialize_idp_ensemble_expanded_pose_donors_v2.donor_ids：
只有 experimental_pose_source_ready=True 的组件的同源谱系 PDB 进入供体列表，
且 ID 必须排序（确定性输出）——防止未验证来源的 pose 混入材料化流程。
"""
from scripts.materialize_idp_ensemble_expanded_pose_donors_v2 import donor_ids


# 供体准入：未就绪组件的 PDB 必须被排除，输出按字典序排序保证可复现——
# 防止就绪闸门失效或输出顺序不稳定。
def test_donor_ids_only_include_experimental_ready_components():
    audit = {
        "components": [
            {
                "experimental_pose_source_ready": True,
                "same_lineage_target_pdb_ids": ["2bbb", "1aaa"],
            },
            {
                "experimental_pose_source_ready": False,
                "same_lineage_target_pdb_ids": ["3ccc"],
            },
        ]
    }
    assert donor_ids(audit) == ["1aaa", "2bbb"]
