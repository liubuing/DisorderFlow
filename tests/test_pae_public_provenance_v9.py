import numpy as np
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from scripts.audit_pae_public_provenance_v9 import chemistry, best_overlap


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


def test_two_sided_coverage_can_miss_true_fragment():
    assert best_overlap('ANPNANP','ANPNANPNANPNANPNANPN') is None
    assert 'ANPNANP' in 'ANPNANPNANPNANPNANPN'
    assert best_overlap('AVGIGAVFL','AVGIGAVF')['identity']==1
