from disorderflow.datasets.statecontrast_v2_pose_manifest import _parse_pose


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
