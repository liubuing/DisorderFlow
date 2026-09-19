"""Apply chemical eligibility to all members and preserve biological strata."""
import sys
from pathlib import Path
from Bio.PDB import MMCIFParser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest
from scripts.audit_pae_public_provenance_v9 import chemistry, best_overlap
from scripts.audit_successor_v3_isolation import select_representative

SOURCE = ROOT / 'data/pae_public_crosscheck_v8'
OUT = ROOT / 'data/pae_public_provenance_v9'
FAMILIES = {'PUB001':'HIV1_p24_epitope_mimotopes', 'PUB002':'HIV1_gp120_V3',
    'PUB003':'Plasmodium_CSP_NANP', 'PUB004':'phosphorylated_tau', 'PUB005':'Plasmodium_CSP_NANP',
    'PUB006':'HIV1_fusion_peptide', 'PUB007':'Plasmodium_CSP_NANP'}


def main():
    groups = read(SOURCE / 'retrospective_candidates.json')['components']
    records = {r['instance']:r for r in read(SOURCE / 'structures/structural_manifest.json')['records']}
    refs = read(ROOT / 'data/pae_independent_v7/reference_union.json')['records']
    inventory = read(OUT / 'generator_sequence_inventory.json')['records']
    parser = MMCIFParser(QUIET=True)
    checked, qualified = [], []
    for group in groups:
        valid = []
        for member in group['members']:
            r = records[member]
            assert digest(ROOT/r['source_cif']) == r['source_cif_sha256']
            chain = next(parser.get_structure(r['pdb_id'],str(ROOT/r['source_cif'])).get_models())[r['original_chain_ids']['antigen']]
            c = chemistry(chain)
            supported = c['supported_unmodified_peptide'] and c['canonical_sequence_with_unknown_markers']==r['antigen_sequence']
            checked.append({'component_id':group['component_id'], 'instance':member, 'chemistry':c, 'supported':supported})
            if supported:
                valid.append(r)
        if not valid:
            continue
        representative = select_representative(valid)
        query = representative['antigen_sequence']
        matches = []
        for ref in refs:
            target = ref.get('antigen_sequence','')
            hit = best_overlap(query,target)
            if target and ((hit and hit['identity']>=.3) or query in target or target in query):
                matches.append({'reference_id':ref['reference_id'], 'pdb_id':ref['pdb_id'],
                    'sources':ref.get('sources',[]), 'antigen_sequence':target,
                    'query_contained_in_reference':query in target, 'reference_contained_in_query':target in query,
                    'alignment':hit})
        matches.sort(key=lambda m: (not m['query_contained_in_reference'], -(m['alignment'] or {}).get('identity',0),m['reference_id']))
        qualified.append({'component_id':group['component_id'], 'family_stratum':FAMILIES[group['component_id']],
            'members':[r['instance'] for r in valid], 'representative':representative,
            'representative_changed_due_to_chemistry':representative['instance']!=group['representative']['instance'],
            'reference_antigen_matches':matches,
            'generator_antigen_substring_hits':{p:[{'id':r['id'],'antigen_sequence':r['antigen_sequence']} for r in rs if query in r['antigen_sequence']] for p,rs in inventory.items()},
            'strict_independent':False, 'allowed_scope':'retrospective_descriptive_screening_with_disclosed_overlap',
            'ready_for_scoring':False})
    freeze(OUT / 'qualified_candidates.json', {'classification':'chemistry_corrected_retrospective_candidates',
        'input_hashes':{p:digest(SOURCE/p) for p in ['retrospective_candidates.json','structures/structural_manifest.json']},
        'script_sha256':digest(Path(__file__)), 'member_chemistry':checked, 'components':qualified,
        'counts':{'checked_members':len(checked), 'supported_members':sum(r['supported'] for r in checked),
            'candidate_components':len(qualified), 'biological_family_strata':len({g['family_stratum'] for g in qualified}), 'strict_independent_components':0},
        'family_basis':'PDB source titles plus exact/contained peptide motifs; strata are biological annotations, not inferred evolutionary independence',
        'decision':'Suitable only for a separately frozen descriptive retrospective pilot; no Q1/Q2 novel-antigen proof. Generation, targets, endpoint, overlap strata and failure rules must be fixed before scores.',
        'do_not_use':['original PUB001 representative1CFN with deleted NLE','PUB004 5DMG with deleted SEP'],
        'af2_scoring_started':False})
    print([(g['component_id'],g['representative']['pdb_id'],g['family_stratum']) for g in qualified])


if __name__=='__main__':
    main()
