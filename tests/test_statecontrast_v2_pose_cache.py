"""状态对比 v2 姿态解析缓存的契约测试：守护同一坐标文件的解析结果被缓存复用（二次调用命中），
防止重复解析大 PDB 拖慢训练或缓存键设计错误导致命中失效。"""
from disorderflow.datasets.statecontrast_v2_pose_manifest import _parse_pose


# 缓存命中：同路径同链参数的两次 _parse_pose 调用必须命中缓存一次——防止缓存键含不可控成分。
def test_pose_parser_cache_is_reused_for_same_coordinate(tmp_path):
    path = tmp_path / "pose.pdb"
    path.write_text(
        "ATOM      1  N   ALA H   1       0.000   0.000   0.000  1.00 10.00           N\n"
        "ATOM      2  CA  ALA H   1       1.000   0.000   0.000  1.00 10.00           C\n"
        "ATOM      3  C   ALA H   1       2.000   0.000   0.000  1.00 10.00           C\n"
        "TER\nEND\n", encoding="ascii")
    _parse_pose.cache_clear()
    _parse_pose(str(path), ("H",), ())
    _parse_pose(str(path), ("H",), ())
    assert _parse_pose.cache_info().hits == 1
