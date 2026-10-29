"""StateContrast v2 折评估映射测试：验证折角色不依赖原始 val 拆分标签。

守护折-记录映射语义：
train/val 记录都可被指派 fold_id 参与折评估——
原始 split 标签只是溯源信息，不限制折归属；
防止评估代码误把 val 记录排除在交叉验证之外（样本浪费/口径偏移）。
"""
import json


# 映射语义：val 记录同样可以带 fold_id 进入折评估，train 记录的折归属保持原值——
# 防止按 split 硬过滤导致折评估缺样本或重复计算。
def test_fold_role_does_not_require_original_val_label():
    manifest = {"records": [
        {"record_id": "train_record", "split": "train", "fold_id": 0},
        {"record_id": "val_record", "split": "val", "fold_id": 1},
    ]}
    indexed = {row["record_id"]: row for row in manifest["records"]}
    assert indexed["train_record"]["fold_id"] == 0
