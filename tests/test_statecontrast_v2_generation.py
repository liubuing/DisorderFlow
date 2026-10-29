"""StateContrast v2 候选生成测试：验证 H3 注入只改设计区与替换计数精确性。

守护 scripts.generate_statecontrast_v2_candidates：
inject_h3 只覆写 design_region_flag 为真的位置且不得就地修改输入 batch、
substitutions 必须返回精确替换数——防止生成时污染上下文或预算计数漂移。
"""
import torch

from scripts.generate_statecontrast_v2_candidates import inject_h3, substitutions


# 注入隔离：只有设计区位置被替换（上下文 9 保持），且输入 batch 不被就地修改——
# 防止生成器污染共享 batch 或越界覆写。
def test_h3_injection_changes_only_design_region():
    batch = {
        "aa": torch.tensor([[9, 9, 9, 9]]),
        "design_region_flag": torch.tensor([[False, True, True, False]]),
    }
    output = inject_h3(batch, "AC")
    assert output["aa"].tolist() == [[9, 0, 1, 9]]
    assert batch["aa"].tolist() == [[9, 9, 9, 9]]


# 替换计数：逐位比较的替换数必须精确（2 处）——防止计数器把相同位也算入突变预算。
def test_substitution_count_is_exact():
    assert substitutions("AAAA", "ACCA") == 2
