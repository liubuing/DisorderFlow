"""PAE 公共溯源 v9 测试：验证修饰残基审计与双侧覆盖的重叠判定。

守护 scripts.audit_pae_public_provenance_v9 的溯源化学审计：
修饰残基（SEP/NLE）必须被标记且不得当作未修饰肽支持、
canonical 序列用未知标记 X 表示非标准残基、
best_overlap 施加双侧覆盖率（短片段即使完全包含于目标也可能不合格）——
防止修饰或部分覆盖的公共结构被静默当作标准肽证据。
"""
import numpy as np
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from scripts.audit_pae_public_provenance_v9 import chemistry, best_overlap


# 修饰残基防护：SEP（磷酸丝氨酸）/NLE 必须进 nonstandard_residues，
# 肽支持标志为 False，canonical 序列以 X 标记——防止修饰残基被静默删除或洗白。
def test_modified_residue_cannot_be_silently_deleted():
    chain = Chain('P')
    for i, name in enumerate(['MET','SEP','LEU','NLE'],1):
        r = Residue((' ' if name in ['MET','LEU'] else 'H_'+name,i,' '),name,' ')
        r.add(Atom('CA',np.zeros(3),1,1,' ',' CA ',i,element='C'))
        chain.add(r)
    result = chemistry(chain)
    assert not result['supported_unmodified_peptide']
    assert result['resolved_ca_residues']==4
    assert [r['residue'] for r in result['nonstandard_residues']]==['SEP','NLE']
    assert result['canonical_sequence_with_unknown_markers']=='MXLX'


# 双侧覆盖语义：子串关系成立但查询侧覆盖率不足时仍判 None；
# 高覆盖且完全一致的片段才通过——防止短片段碰瓷真实肽段。
def test_two_sided_coverage_can_miss_true_fragment():
    assert best_overlap('ANPNANP','ANPNANPNANPNANPNANPN') is None
    assert 'ANPNANP' in 'ANPNANPNANPNANPNANPN'
    assert best_overlap('AVGIGAVFL','AVGIGAVF')['identity']==1
