"""PAE 公共审计 v8 工具测试：验证 PDB ID 归一化与短序列对齐的双重覆盖要求。

守护 PAE 公共来源/序列审计（v8）：
pdb_00006HHD / 6HHD_2 等变体必须归一到同一 ID（去重保映射）、
无缺口对齐必须同时满足查询与目标两侧覆盖率、末端偏移须被计入——
防止错误匹配或重复 PDB 条目混入公共审计。
"""
from scripts.audit_pae_public_sequences_v8 import ungapped
from scripts.audit_pae_public_sources_v8 import normalize


# ID 归一化：带前缀/副本后缀的 PDB ID 必须映射到同一小写键——防止重复条目漏去重。
def test_pdb_id_dedup_preserves_case_mapping():
    assert normalize('pdb_00006HHD') == normalize('6HHD_2') == '6hhd'


# 双侧覆盖防护：短查询对长目标、低同一性、空查询都必须拒绝——
# 防止局部/低质匹配被当作序列同一性证据。
def test_short_alignment_requires_both_coverages():
    assert ungapped('ACDEFGHIKL', 'ACDEFGHIKL', .5)
    assert not ungapped('ACDEF', 'ACDEFGHIKLMNPQRSTVWY', .5)
    assert not ungapped('AAAAA', 'CCCCC', .3)
    assert not ungapped('', 'ACDEF', .3)


# 末端偏移：目标内部（非开头）的对齐必须被发现——防止只做起点对齐而漏检真实匹配。
def test_terminal_offset_is_considered():
    assert ungapped('ACDEFGHIKL', 'XXACDEFGHIK', .9)
