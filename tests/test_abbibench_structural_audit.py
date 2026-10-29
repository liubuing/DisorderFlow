"""AbBiBench 结构审计工具测试：验证路径解析与短序列无缺口比对阈值行为。

守护 AbBiBench 审计管线的输入完整性不变量：
链名路径解析必须显式化（大小写修正需唯一、歧义一律拒绝），
短序列筛查同时施加查询覆盖与同一性阈值（边界值取含）——防止静默错配进入下游统计。
"""
from scripts.prepare_abbibench_structural_audit import resolve_path
from scripts.audit_abbibench_short_sequences import ungapped_hit


# 路径解析只允许精确匹配与唯一大小写修正；缺失或多义候选必须显式拒绝，
# 防止审计脚本静默绑定到错误文件。
def test_case_correction_is_explicit_and_ambiguous_paths_are_rejected():
    assert resolve_path('./data/A.pdb', ['A.pdb']) == ('A.pdb', 'exact')
    assert resolve_path('./data/a.pdb', ['A.pdb']) == ('A.pdb', 'unique_case_correction')
    assert resolve_path('a.PDB', ['A.pdb', 'a.pdb']) == (None, 'missing_or_ambiguous')
    assert resolve_path('absent.csv', ['A.pdb']) == (None, 'missing_or_ambiguous')


# 无缺口比对必须同时检查查询覆盖率与同一性，并计入末端偏移：
# 短命中因覆盖率不足应被拒绝——防止截短的错误匹配通过筛查。
def test_ungapped_screen_uses_both_coverages_and_terminal_offsets():
    hit = ungapped_hit('ACDEF', 'CDEF', .9)
    assert hit['query_coverage'] == .8
    assert hit['identity'] == 1
    assert ungapped_hit('ACDEFGHIKL', 'CDEF', .9) is None


# 阈值边界语义：恰好等于阈值的命中应通过、低于阈值与空查询必须拒绝——
# 防止边界比较符（>=/>）写反导致筛选口径漂移。
def test_short_sequence_threshold_includes_boundary_but_not_lower_identity():
    assert ungapped_hit('AAAA', 'AACC', .5) is not None
    assert ungapped_hit('AAAA', 'ACCC', .5) is None
    assert ungapped_hit('', 'AAAA', .5) is None
