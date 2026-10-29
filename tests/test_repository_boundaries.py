"""仓库分区边界的契约测试：守护分区分类优先按声明的路径前缀匹配，
防止新增目录被误归入错误分区而绕过边界审计。"""
from scripts.audit_repository_boundaries import classify


# 前缀优先：disorderflow/ 归 core、scripts/ 归 scripts——分区归类必须与声明前缀一致。
def test_repository_zone_classification_prefers_declared_prefix():
    zones = {"core": ["disorderflow/"], "scripts": ["scripts/"]}
    assert classify(__import__("pathlib").Path("disorderflow/models/x.py"), zones) == "core"
    assert classify(__import__("pathlib").Path("scripts/x.py"), zones) == "scripts"
