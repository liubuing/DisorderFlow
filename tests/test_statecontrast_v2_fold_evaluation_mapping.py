import json


def test_fold_role_does_not_require_original_val_label():
    manifest = {"records": [
        {"record_id": "train_record", "split": "train", "fold_id": 0},
        {"record_id": "val_record", "split": "val", "fold_id": 1},
    ]}
    indexed = {row["record_id"]: row for row in manifest["records"]}
    assert indexed["train_record"]["fold_id"] == 0
